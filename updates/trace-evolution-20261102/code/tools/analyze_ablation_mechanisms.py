"""Explain weak removal effects from sealed candidates; no model or GT access."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from tools import analyze_forward_budget as readers
from tools.analyze_selective_tracing import load_inputs


def dominance_allocation(targets, counterfactuals, residuals):
    """Separate raw attenuation from the change in normalized coefficients.

    Before EPS pruning, p_i=q_i/sum(q) and mu=sum(p_i*d_i). Then
    TV(p, p')=E_p[abs(d-mu)]/(2*(1-mu)). A common d cancels entirely.
    Actual EPS-pruned coefficients are reported independently below.
    """
    if not (len(targets) == len(counterfactuals) == len(residuals)):
        raise ValueError("Support arrays must have equal length")
    targets = list(map(readers.probability, targets))
    counterfactuals = list(map(readers.probability, counterfactuals))
    residuals = list(map(readers.probability, residuals))
    q = [min(t, c) for t, c in zip(targets, counterfactuals)]
    d = [max(r-t, 0.0) for t, r in zip(targets, residuals)]
    rho = [a*(1-b) for a, b in zip(q, d)]
    total = math.fsum(q)
    p = [a/total for a in q] if total else [0.0]*len(q)
    mu = math.fsum(a*b for a, b in zip(p, d))
    denominator = math.fsum(rho)
    exact_tv = (math.fsum(a*abs(b-mu) for a, b in zip(p, d))/(2*(1-mu))
                if total and denominator else None)
    direct_tv = (0.5*math.fsum(abs(a-b/denominator) for a, b in zip(p, rho))
                 if total and denominator else None)
    if exact_tv is not None:
        readers.require(math.isclose(exact_tv, direct_tv, abs_tol=1e-12),
                        "Dominance coefficient identity failed")
    areas = [1]*len(q)
    actual = readers.coefficient_distance(q, rho, areas, 2)
    active = [b > 0 for b in d]
    return dict(candidates=len(q), q_sum=total, rho_sum=denominator,
        dominance_positive=sum(active),
        dominance_positive_fraction=sum(active)/len(active) if active else None,
        dominance_positive_q_mass=math.fsum(a for a, b in zip(p, active) if b),
        raw_attenuation_fraction=mu if total else None,
        normalized_tv_before_eps=exact_tv,
        normalized_tv_after_eps=actual['tv'],
        eps_activity_changes=sum((a > readers.EPS) != (b > readers.EPS)
                                 for a, b in zip(q, rho)),
        active_to_empty=actual['active_to_empty'],
        empty_to_active=actual['empty_to_active'])


def run(cache_root, output):
    output = Path(output).resolve()
    readers.require(not output.exists() or not any(output.iterdir()),
                    "Choose a fresh output directory")
    evidence = readers.Evidence()
    root = Path(__file__).resolve().parents[1]
    for source in (Path(__file__), Path(readers.__file__),
                   root/'tools/analyze_selective_tracing.py'):
        evidence.read(source)
    panels, _, _, shared_seals = load_inputs(Path(cache_root), root, evidence)
    rows, summaries = [], []
    for panel in panels:
        local = []
        for cost in panel['costs']:
            items = panel['grouped'][cost['episode']]
            row = dict(dataset=panel['dataset'], episode=cost['episode'],
                images=cost['images'], **dominance_allocation(
                    [x.target for x in items], [x.counterfactual for x in items],
                    [x.residual for x in items]))
            local.append(row)
        rows.extend(local)
        summary = dict(dataset=panel['dataset'], images=panel['images'],
            episodes=len(local), candidates=sum(r['candidates'] for r in local),
            eps_activity_changes=sum(r['eps_activity_changes'] for r in local))
        summary['dominance_positive_fraction'] = (
            sum(r['dominance_positive'] for r in local)/summary['candidates'])
        q_sum = math.fsum(r['q_sum'] for r in local)
        summary['pooled_q_attenuation_fraction'] = (
            math.fsum(r['q_sum']-r['rho_sum'] for r in local)/q_sum if q_sum else None)
        for field in ('dominance_positive_q_mass', 'raw_attenuation_fraction',
                      'normalized_tv_before_eps', 'normalized_tv_after_eps'):
            values = [r[field] for r in local if r[field] is not None]
            summary[field+'_episode_mean'] = math.fsum(values)/len(values) if values else None
            summary[field+'_episode_max'] = max(values, default=None)
        summaries.append(summary)
    evidence.recheck()
    output.mkdir(parents=True, exist_ok=True)
    tables = {'episode_mechanisms.csv': rows, 'mechanism_summary.csv': summaries}
    for name, values in tables.items():
        readers.write_csv(output/name, values)
    provenance = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        status='completed', source='Sealed cached candidate responses; no new inference',
        normalization_unit='adaptation episode', shared_seals_verified=shared_seals,
        inputs=evidence.files,
        outputs={name: readers.sha256((output/name).read_bytes()) for name in tables},
        limitations=[
            'Coefficient TV is not a gradient, optimizer-trajectory or mIoU bound.',
            'Pooled attenuation is descriptive, not the actual cross-episode loss normalizer.',
            'No GT, model predictions, threshold search or new performance evidence used.',
            'Common attenuation cancels before EPS pruning; pruning can change activity.',
        ])
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.cache_root, args.output)
