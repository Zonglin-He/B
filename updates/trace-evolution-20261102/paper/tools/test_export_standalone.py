"""Synthetic canonical metadata only; no active suite, payload, model or GPU access."""

import copy
import csv
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest import mock


SPEC = importlib.util.spec_from_file_location('export_standalone', Path(__file__).with_name('export_standalone.py'))
export = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False))


def artifacts(manifest):
    episodes = (128 + manifest['config']['batch_size'] - 1) // manifest['config']['batch_size']
    return ({f'predictions/{a}/{i:06d}.npy' for a in export.ARMS for i in range(128)}
            | {f'episodes/{e:06d}.{a}.json' for e in range(episodes) for a in export.ARMS}
            | {'prediction_protocol.json', 'initial_state.json', 'prediction_result.json', 'cost.jsonl'})


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'suite'
        self.paper = Path(self.tmp.name) / 'paper'
        self.generated = self.paper / 'generated'
        self.generated.mkdir(parents=True)
        for name in (*export.OUTPUTS, 'update_risk.tex'):
            (self.generated / name).write_text('KEEP ' + name)
        source = self.root / 'runtime/minimal.py'
        source.parent.mkdir(parents=True)
        source.write_text('# synthetic frozen source\n')
        self.runtime = {'minimal.py':export.sha(source)}
        self.prepared = dict(cells=[], smokes=[], artifact_sha256={}, runtime_hashes=self.runtime)
        self.scope = 'synthetic timing boundary'
        self.worker = types.SimpleNamespace(prediction_artifacts=artifacts, TIMING_SCOPE=self.scope,
                                             GPU_LOCK=Path('/synthetic/gpu.lock'))
        self.queue = types.SimpleNamespace(EXPOSURE='synthetic development exposed', worker=lambda:self.worker,
            verify_prepared=self.verify_prepared, strict_runtime=self.strict_runtime,
            validate_smoke=lambda root,e:export.sha(root/e['output']))
        self.canonical = []
        for d in export.DATASETS:
            smoke = dict(key='smoke/'+d, output='smoke/'+d+'.json')
            self.prepared['smokes'].append(smoke)
            write(self.root / smoke['output'], dict(deterministic_runtime={'deterministic':True},
                gpu_lock=dict(path='/synthetic/gpu.lock',inode=12,pid=1,exclusive=True)))
            for condition in export.CONDITIONS:
                self.make_cell(d, condition)
        self.seal_prepared()
        write(self.root / 'EXECUTION_STARTED.json', dict(suite_sha256=self.digest,pid=1234))
        write(self.root / 'ALL_PREDICTIONS_SEALED.json', dict(suite_sha256=self.digest,labels_opened_by_queue=False,
            smoke_sha256={e['key']:export.sha(self.root/e['output']) for e in self.prepared['smokes']},
            prediction_seal_sha256={e['key']:export.sha(self.root/e['output']/'PREDICTIONS_SEALED.json')
                                    for e in self.prepared['cells']}))
        write(self.root / 'execution_children.json', [dict(exit_code=0)]*14)
        write(self.root / 'queue_state.json', dict(phase='completed',passed_smokes=2,sealed_predictions=6,
                                               scored_cells=6,total_cells=6,error=None))
        self.csv_rows()
        self.seal_suite()

    def strict_runtime(self, value):
        export.require(value == {'deterministic':True}, 'Runtime drift')

    def verify_prepared(self, root, digest):
        export.require(export.sha(root/'SUITE_PREPARED.json') == digest, 'Prepared drift')
        export.require([(e['dataset'],e['condition']) for e in self.prepared['cells']]
                       == [(d,c) for d in export.DATASETS for c in export.CONDITIONS], 'Plan drift')
        return export.read(root/'SUITE_PREPARED.json')

    def seal_prepared(self):
        write(self.root / 'SUITE_PREPARED.json', self.prepared)
        self.digest = export.sha(self.root/'SUITE_PREPARED.json')

    def make_cell(self, dataset, condition):
        key = dataset+'/'+condition
        entry = dict(key=key,dataset=dataset,condition=condition,output='cells/'+key,
                     manifest='manifests/'+dataset+'.'+condition+'.json')
        directory = self.root / entry['output']
        cfg = dict(batch_size=2 if condition=='common_sorted' else 64,steps=10)
        manifest = dict(dataset=dataset,config=cfg,selection_seed=20261029,
            episode_order_seed=20261030 if condition=='inherited_regrouped' else None,
            panel=dict(classes=['a'],inputs=[dict(label='/synthetic/'+dataset+'/'+str(i)) for i in range(128)],
                       ordered_ids=list(range(128))), exposure_note=self.queue.EXPOSURE,claim_boundary='diagnostic')
        write(self.root / entry['manifest'], manifest)
        entry['manifest_sha256'] = export.sha(self.root/entry['manifest'])
        self.prepared['artifact_sha256'][entry['manifest']] = entry['manifest_sha256']
        self.prepared['cells'].append(entry)
        protocol = dict(manifest=manifest,manifest_sha256=entry['manifest_sha256'],
            manifest_path=str(self.root/entry['manifest']), selection_seed=manifest['selection_seed'],
            episode_order_seed=manifest['episode_order_seed'],ordered_ids=manifest['panel']['ordered_ids'],
            deterministic_runtime={'deterministic':True},gpu_lock=dict(path='/synthetic/gpu.lock',inode=12,pid=2,exclusive=True))
        write(directory/'prediction_protocol.json',protocol)
        write(directory/'initial_state.json',{})
        costs, score_rows = [], []
        batch = cfg['batch_size']
        episodes = (128+batch-1)//batch
        values = dict(zip(export.ARMS,(60,50,55,70,40)))
        for e in range(episodes):
            images=min(batch,128-e*batch)
            order=list(reversed(export.ARMS)) if e%2 else list(export.ARMS)
            for index,a in enumerate(export.ARMS):
                factor=0 if a in ('Uniform','Confidence') else 2 if a=='min' else 1
                n=factor*images
                dc=1 if n else 0
                costs.append(dict(episode=e,arm=a,order_position=order.index(a),images=images,warmup=e==0,
                    steps=10,candidate_count=images,candidate_sha256='candidate',episode_sha256='episode',
                    seconds=float(images*(index+1)),diagnostic_forward_inputs=n,diagnostic_encoder_inputs=n,
                    diagnostic_forward_calls=dc,diagnostic_encoder_calls=dc,eval_forward_inputs=images,
                    eval_forward_calls=1,adapt_forward_inputs=images*10,adapt_forward_calls=10,
                    encoder_inputs=11*images+n,encoder_calls=11+dc,total_forward_inputs=11*images+n,
                    total_forward_calls=11+dc))
            score_rows.append(dict(episode=e,warmup=e==0,arms={a:dict(intersection=[values[a]*images],
                                                      union=[100*images]) for a in export.ARMS}))
        directory.mkdir(parents=True,exist_ok=True)
        (directory/'cost.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in costs))
        timings = {a:dict(measured_images=128-batch,measured_episodes=episodes-1,seconds=float((128-batch)*(index+1)),
                          seconds_per_image=float(index+1),images_per_second=1/(index+1),warmup_episode=0,
                          accuracy_includes_warmup=True) for index,a in enumerate(export.ARMS)}
        write(directory/'prediction_result.json',dict(status='predicted',images=128,full_split=False,
              accuracy_includes_warmup=True,timings=timings,timing_scope=self.scope))
        hashes={n:(export.sha(directory/n) if (directory/n).exists() else '0'*64) for n in artifacts(manifest)}
        write(directory/'PREDICTIONS_SEALED.json',dict(complete=True,gt_contents_opened=False,images=128,
              arms=list(export.ARMS),artifact_sha256=hashes))
        pred_hash=export.sha(directory/'PREDICTIONS_SEALED.json')
        result=dict(status='completed',dataset=dataset,seed=0,images=128,full_split=False,
            prediction_seal_sha256=pred_hash,label_contents_first_opened_after_all_requested_arms_sealed=True,
            exposure_note=self.queue.EXPOSURE,claim_boundary='diagnostic',timing_scope=self.scope,timings=timings,
            arm_results={a:dict(miou_unrounded=float(values[a]),intersection=[values[a]*128],union=[12800])
                         for a in export.ARMS})
        write(directory/'result.json',result)
        write(directory/'SCORING_STARTED.json',dict(prediction_seal_sha256=pred_hash))
        write(directory/'score_inputs.json',dict(label_sha256={r['label']:'1'*64 for r in manifest['panel']['inputs']}))
        (directory/'scores_by_episode.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in score_rows))
        self.seal_score(directory)
        self.canonical.append((entry,result))

    def seal_score(self,directory):
        names=('SCORING_STARTED.json','scores_by_episode.jsonl','score_inputs.json','result.json')
        write(directory/'SCORED_SEALED.json',dict(prediction_seal_sha256=export.sha(directory/'PREDICTIONS_SEALED.json'),
              artifact_sha256={n:export.sha(directory/n) for n in names}))

    def seal_suite(self):
        entries=dict(self.prepared['artifact_sha256'])
        names=['SUITE_PREPARED.json','EXECUTION_STARTED.json','ALL_PREDICTIONS_SEALED.json',
               'rows.csv','execution_children.json','queue_state.json']
        names += [e['output'] for e in self.prepared['smokes']]
        names += [e['output']+'/'+n for e in self.prepared['cells']
                  for n in ('PREDICTIONS_SEALED.json','SCORED_SEALED.json','result.json')]
        entries.update({n:export.sha(self.root/n) for n in names})
        write(self.root/'SUITE_SEALED.json',dict(complete=True,formal_cells=6,arm_results=30,passed_gt_free_gates=2,
            full_split=False,suite_prepared_sha256=self.digest,runtime_hashes=self.runtime,artifact_sha256=entries,
            payload_verification='All cell prediction and score payloads rehashed via their linked seals',
            exposure_note=self.queue.EXPOSURE))

    def csv_rows(self):
        fields=['dataset','condition','arm','images','full_split','miou_unrounded','measured_images',
                'seconds','seconds_per_image','images_per_second']
        with (self.root/'rows.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader()
            for e,r in self.canonical:
                for a in export.ARMS:
                    writer.writerow(dict(dataset=e['dataset'],condition=e['condition'],arm=a,images=128,
                        full_split=False,miou_unrounded=r['arm_results'][a]['miou_unrounded'],
                        **{k:r['timings'][a][k] for k in fields[6:]}))

    def snapshot(self):
        return {p.name:p.read_bytes() for p in self.generated.iterdir()}

    def run_export(self,check_only=False):
        with mock.patch.object(export,'load_queue',return_value=self.queue), \
             mock.patch.object(export,'service_success',return_value={'verified':'synthetic success'}):
            return export.run(self.root,self.paper,self.digest,'synthetic.service',check_only)

    def test_complete_import_without_any_prediction_payload_reads(self):
        status,code=self.run_export()
        self.assertEqual((status['status'],code),('imported',0))
        self.assertEqual(set(status['deltas'].values()),{-10.})
        comparison=(self.generated/export.OUTPUTS[0]).read_text()
        self.assertIn('$-10.00$',comparison)
        self.assertIn(r'$\mathbf{70.00}$',comparison)
        self.assertNotIn(r'$\mathbf{40.00}$',comparison)
        self.assertNotIn('inherited_sorted',comparison)
        self.assertIn('development-exposed',comparison)
        cost=(self.generated/export.OUTPUTS[1]).read_text()
        self.assertIn('64 & 1000.00',cost)
        self.assertIn('126 & 1000.00',cost)
        self.assertIn('all 128 images',cost)
        self.assertEqual((self.generated/'update_risk.tex').read_text(),'KEEP update_risk.tex')
        p=export.read(self.generated/export.OUTPUTS[2])
        self.assertEqual(len(p['cells']),6)
        self.assertEqual(p['cells'][0]['costs']['min']['diagnostic_forward_inputs'],256)
        for n,h in p['output_sha256'].items():
            self.assertEqual(export.sha(self.generated/n),h)

    def test_partial_and_check_only_preserve_outputs(self):
        before=self.snapshot()
        self.assertEqual(self.run_export(True)[0]['status'],'ready')
        self.assertEqual(before,self.snapshot())
        (self.root/'SUITE_SEALED.json').unlink()
        with mock.patch.object(export,'verify_cell',side_effect=AssertionError('No cell import before seal')):
            self.assertEqual(self.run_export()[1],3)
        self.assertEqual(before,self.snapshot())

    def test_negative_null_nonfinite_and_bad_counts_rejected(self):
        for value in (None,-1.,float('nan'),float('inf')):
            with self.assertRaises(export.Invalid):
                export.pooled(dict(intersection=[1],union=[2],miou_unrounded=value),1)
        for i,u in (([3],[2]),([-1],[2]),([1.],[2]),([0],[0])):
            with self.assertRaises(export.Invalid):
                export.pooled(dict(intersection=i,union=u,miou_unrounded=50.),1)

    def test_resealed_false_score_and_scope_fail_closed(self):
        directory=self.root/self.prepared['cells'][0]['output']
        original=export.read(directory/'result.json')
        variants=[dict(original,images=127),dict(original,full_split=True)]
        wrong=copy.deepcopy(original)
        wrong['arm_results']['min']['miou_unrounded']=41.
        variants.append(wrong)
        for result in variants:
            write(directory/'result.json',result)
            self.seal_score(directory)
            self.seal_suite()
            before=self.snapshot()
            with self.assertRaises(export.Invalid): self.run_export()
            self.assertEqual(before,self.snapshot())

    def test_csv_duplicate_missing_and_wrong_link_rejected(self):
        original=(self.root/'rows.csv').read_text().splitlines()
        variants=[original[:-1],original[:-1]+[original[1]],
                  [original[0],original[1].replace(',128,False,',',127,False,'),*original[2:]]]
        for rows in variants:
            (self.root/'rows.csv').write_text('\n'.join(rows)+'\n')
            self.seal_suite()
            with self.assertRaises(export.Invalid): self.run_export()

    def test_source_drift_and_barrier_drift_rejected(self):
        (self.root/'runtime/minimal.py').write_text('changed')
        with self.assertRaisesRegex(export.Invalid,'hash mismatch'): self.run_export()

    def test_resealed_barrier_missing_prediction_link_rejected(self):
        path=self.root/'ALL_PREDICTIONS_SEALED.json'
        barrier=export.read(path)
        barrier['prediction_seal_sha256'].pop(next(iter(barrier['prediction_seal_sha256'])))
        write(path,barrier)
        self.seal_suite()
        with self.assertRaisesRegex(export.Invalid,'barrier population'): self.run_export()

    def test_partial_service_and_failed_queue_never_publish(self):
        before=self.snapshot()
        with mock.patch.object(export,'load_queue',return_value=self.queue), \
             mock.patch.object(export,'service_success',side_effect=export.Pending('still running')):
            with self.assertRaises(export.Pending): export.run(self.root,self.paper,self.digest)
        self.assertEqual(before,self.snapshot())
        (self.root/'SUITE_SEALED.json').unlink()
        write(self.root/'queue_state.json',dict(phase='failed'))
        with self.assertRaisesRegex(export.Invalid,'Queue failed'): self.run_export()
        self.assertEqual(before,self.snapshot())

    def test_unrounded_winners_and_ties_are_not_forced_to_trace(self):
        c=dict(dataset='pc60',condition='common_sorted',scores=dict(zip(export.ARMS,(50.,50.,50.,50.004,50.003))),
            delta=.003,timings={a:{'measured_images':126} for a in export.ARMS},
            costs={a:dict(ms_per_image=1.,diagnostic_forward_inputs_per_image=0.,encoder_inputs_per_image=2.)
                   for a in export.ARMS})
        text=export.render([c])[export.OUTPUTS[0]]
        row=next(v for v in text.splitlines() if v.startswith('PC60 &'))
        self.assertEqual(row.count(r'\mathbf'),1)
        self.assertEqual(row.split(' & ')[5],r'$\mathbf{50.00}$')
        c['scores']['min']=50.004
        text=export.render([c])[export.OUTPUTS[0]]
        row=next(v for v in text.splitlines() if v.startswith('PC60 &'))
        self.assertEqual(row.count(r'\mathbf'),2)

    def test_service_running_missing_failed_foreign_and_success(self):
        state=dict(Id='synthetic.service',LoadState='loaded',ActiveState='active',SubState='exited',Result='success',
            MainPID='0',ExecMainPID='1234',ExecMainCode='1',ExecMainStatus='0',
            ExecStart=f'python -m tools.evolution_followup.queue run --suite {self.root} --suite-sha256 {self.digest} ;')
        def check(s):
            response=types.SimpleNamespace(stdout='\n'.join(k+'='+v for k,v in s.items()))
            with mock.patch.object(export.subprocess,'run',return_value=response):
                return export.service_success('synthetic.service',self.root,self.digest,{'pid':1234})
        self.assertEqual(check(state),state)
        with self.assertRaises(export.Pending): check(dict(state,MainPID='1234',SubState='running'))
        for change in (dict(LoadState='not-found'),dict(Result='failed'),dict(ExecMainStatus='1'),
                       dict(ExecMainPID='999'),dict(ExecStart='foreign root')):
            with self.assertRaises(export.Invalid): check(dict(state,**change))

    def test_cost_incomplete_negative_denominator_and_null_fail(self):
        e,r=self.canonical[0]
        path=self.root/e['output']/'cost.jsonl'
        original=path.read_text()
        manifest=export.read(self.root/e['manifest'])
        rows=[json.loads(line) for line in original.splitlines()]
        for changed in ([*rows[:-1]], [dict(rows[0],seconds=-1.),*rows[1:]],
                        [dict(rows[0],diagnostic_forward_inputs=None),*rows[1:]]):
            path.write_text(''.join(json.dumps(v)+'\n' for v in changed))
            with self.assertRaises(export.Invalid): export.cost_summary(path,manifest,r['timings'])
        path.write_text(original)
        wrong=copy.deepcopy(r['timings'])
        wrong['min']['measured_images']=128
        with self.assertRaises(export.Invalid): export.cost_summary(path,manifest,wrong)

    def test_atomic_publication_rolls_back(self):
        before=self.snapshot()
        real=export.os.replace
        calls=0
        def fail(a,b):
            nonlocal calls
            calls+=1
            if calls==2: raise OSError('synthetic failure')
            return real(a,b)
        with mock.patch.object(export.os,'replace',side_effect=fail):
            with self.assertRaises(OSError): export.publish(self.paper,{n:'new' for n in export.OUTPUTS})
        self.assertEqual(before,self.snapshot())

    def test_bootstrap_rejects_wrong_prepared_hash_before_helpers(self):
        with mock.patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':''}):
            with self.assertRaisesRegex(export.Invalid,'Prepared-suite'):
                export.load_queue(self.root,'wrong')


if __name__=='__main__':
    unittest.main()
