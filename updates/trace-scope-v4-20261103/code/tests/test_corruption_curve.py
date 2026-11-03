import copy
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from tools.evolution_fifty import corruption_curve as curve


def same_rng(a,b):
    return a[0] == b[0] and np.array_equal(a[1],b[1]) and a[2:] == b[2:]


def test_exact_fixed_matrix_and_no_S5_generation():
    plan = curve.plan()
    assert plan["new_RGBs"] == 14490 and plan["future_cells"] == 30
    assert plan["future_image_method_evaluations"] == 43470
    assert [r["id"] for r in curve.conditions()] == [f"{kind}.s{s}" for kind in curve.CURVE_TYPES for s in (1,3)]
    assert not plan["executable_queue"] and not plan["GPU_allowed"] and not plan["scientific_completion"]


@pytest.mark.parametrize("condition",curve.conditions())
def test_original_BGR_corruption_before_resize_seed_and_lossless_PNG(condition):
    image = np.random.default_rng(22).integers(0,256,(37,53,3),dtype=np.uint8)
    original = image.copy()
    calls = []
    def corrupt(pixels,*,severity,corruption_name):
        calls.append((pixels.shape,severity,corruption_name))
        assert np.array_equal(pixels,original)
        return np.clip(pixels.astype(int)+np.random.randint(-18,19,pixels.shape),0,255).astype(np.uint8)
    state = np.random.get_state()
    encoded = curve.generate_image(SimpleNamespace(corrupt=corrupt),image,7,condition["corruption"],condition["severity"])
    assert same_rng(state,np.random.get_state()) and np.array_equal(image,original)
    expected = np.random.RandomState(7).randint(-18,19,original.shape)
    expected = cv2.resize(np.clip(original.astype(int)+expected,0,255).astype(np.uint8),(224,224),interpolation=cv2.INTER_LINEAR)
    assert np.array_equal(cv2.imdecode(np.frombuffer(encoded,np.uint8),cv2.IMREAD_COLOR),expected)
    assert calls == [(original.shape,condition["severity"],condition["corruption"])]
    np.random.seed(456)
    assert curve.generate_image(SimpleNamespace(corrupt=corrupt),image,7,condition["corruption"],condition["severity"]) == encoded


@pytest.mark.parametrize("fault",["raise","mutate","mutate_raise","float","shape","nonarray"])
def test_reference_fault_restores_rng_and_shared_clean_even_after_mutation(fault):
    image = np.zeros((33,41,3),np.uint8)
    def corrupt(pixels,**_):
        np.random.rand(5)
        if fault.startswith("mutate"):pixels[0,0,0]=255
        if fault.endswith("raise"):raise RuntimeError("synthetic upstream failure")
        if fault == "float":return pixels.astype(np.float32)
        if fault == "shape":return pixels[:32]
        if fault == "nonarray":return None
        return pixels
    state = np.random.get_state()
    with pytest.raises((ValueError,RuntimeError)):
        curve.generate_image(SimpleNamespace(corrupt=corrupt),image,0,"fog",1)
    assert same_rng(state,np.random.get_state()) and not image.any()


@pytest.mark.parametrize("index,kind,severity",[(True,"fog",1),(-1,"fog",1),(1449,"fog",1),
    (0,"snow",1),(0,"fog",5),(0,"fog",True),(0,"fog",1.),(0,"fog",0)])
def test_unplanned_conditions_reject_before_any_reference_or_rng(index,kind,severity):
    state = np.random.get_state()
    with pytest.raises(ValueError):curve.generate_image(SimpleNamespace(),np.zeros((32,32,3),np.uint8),index,kind,severity)
    assert same_rng(state,np.random.get_state())


@pytest.mark.parametrize("fault",["float","shape","empty","oversize"])
def test_invalid_original_pixels_reject_before_work(monkeypatch,fault):
    image = np.zeros((32,32,3),np.uint8)
    if fault == "float":image=image.astype(np.float32)
    elif fault == "shape":image=image[:,:,0]
    elif fault == "empty":image=image[:0]
    else:monkeypatch.setattr(curve,"MAX_SOURCE_BYTES",1)
    with pytest.raises(ValueError):curve.generate_image(SimpleNamespace(),image,0,"fog",1)


def manifests(monkeypatch):
    monkeypatch.setattr(curve,"SOURCES",5)
    sources = [dict(official_id=f"s{i}",image=f"/synthetic/clean/{i}.jpg",image_sha256=f"{i+1:064x}",
        label=f"/synthetic/GT/{i}.png",label_metadata=[123,i],label_metadata_details=dict(contents_read=False)) for i in range(5)]
    root="/synthetic/G03"
    def row(source,index,kind,severity):
        result=copy.deepcopy(source)
        prefix=root if severity in (1,3) else "/synthetic/sealedS5"
        result.update(image=f"{prefix}/rgb/{kind}/severity_{severity}/{index:06d}.png",image_sha256="a"*64,
            clean_image=source["image"],clean_image_sha256=source["image_sha256"],corruption=kind,severity=severity,
            original_sample_idx=index,preprocessing=curve.PREPROCESSING)
        return result
    manifest={c["id"]:[row(s,i,c["corruption"],c["severity"]) for i,s in enumerate(sources)] for c in curve.conditions()}
    reuse={kind:[row(s,i,kind,5) for i,s in enumerate(sources)] for kind in curve.CURVE_TYPES}
    return manifest,reuse,sources,root


def test_declared_manifest_and_S5_reuse_are_pure_metadata_not_payload_proof(monkeypatch):
    manifest,reuse,sources,root=manifests(monkeypatch)
    before=copy.deepcopy((manifest,reuse,sources))
    def deny(*_a,**_k):pytest.fail("Pure validator did file IO")
    monkeypatch.setattr(Path,"open",deny)
    monkeypatch.setattr(Path,"stat",deny)
    result=curve.validate_manifest(manifest,sources,root)
    old=curve.validate_S5_reuse(reuse,sources)
    assert result["new_RGBs"] == 50 and not result["payload_hashes_verified"] and not result["filesystem_identity_verified"]
    assert old["reference_RGBs"] == 25 and not old["S5_regenerated"] and not old["payload_or_seal_verified"]
    assert (manifest,reuse,sources) == before


@pytest.mark.parametrize("fault",["key","population","order","label","labelmetadata","cleanhash","outputhash",
    "cleanpath","index","floatindex","severity5","floatseverity","geometry","path","sourceID","extra"])
def test_manifest_faults_reject_no_silent_scope_or_GT_change(monkeypatch,fault):
    manifest,_,sources,root=manifests(monkeypatch)
    rows=manifest[curve.conditions()[-1]["id"]]
    row=rows[-1]
    if fault == "key":manifest["fog.s5"]=manifest.pop(curve.conditions()[0]["id"])
    elif fault == "population":rows.pop()
    elif fault == "order":rows.reverse()
    elif fault == "label":row["label"]="/synthetic/otherGT.png"
    elif fault == "labelmetadata":row["label_metadata"][0]+=1
    elif fault == "cleanhash":row["clean_image_sha256"]="b"*64
    elif fault == "outputhash":row["image_sha256"]="g"*64
    elif fault == "cleanpath":row["clean_image"]="/other.jpg"
    elif fault == "index":row["original_sample_idx"]=0
    elif fault == "floatindex":row["original_sample_idx"]=4.
    elif fault == "severity5":row["severity"]=5
    elif fault == "floatseverity":row["severity"]=3.
    elif fault == "geometry":row["preprocessing"]="second_resize"
    elif fault == "path":row["image"]=rows[0]["image"]
    elif fault == "sourceID":sources[-1]["official_id"]=sources[0]["official_id"]
    else:row["GT_used_for_selection"]=True
    with pytest.raises(ValueError):curve.validate_manifest(manifest,sources,root)


@pytest.mark.parametrize("root",["relative","/x/../y","/x//y","/x/./y",True])
def test_noncanonical_root_rejects(monkeypatch,root):
    manifest,_,sources,_=manifests(monkeypatch)
    with pytest.raises(ValueError):curve.validate_manifest(manifest,sources,root)


@pytest.mark.parametrize("fault",["missing","order","severity","index","GT","reusepath","relative","sourcepath"])
def test_existing_S5_metadata_must_match_original_population(monkeypatch,fault):
    _,reuse,sources,_=manifests(monkeypatch)
    rows=reuse[curve.CURVE_TYPES[-1]]
    if fault == "missing":reuse.pop(curve.CURVE_TYPES[0])
    elif fault == "order":rows.reverse()
    elif fault == "severity":rows[0]["severity"]=3
    elif fault == "index":rows[0]["original_sample_idx"]=1
    elif fault == "GT":rows[0]["label"]="/otherGT"
    elif fault == "reusepath":rows[1]["image"]=rows[0]["image"]
    elif fault == "relative":rows[0]["image"]="relative/path.png"
    else:rows[0]["image"]=sources[0]["label"]
    with pytest.raises(ValueError):curve.validate_S5_reuse(reuse,sources)


@pytest.mark.parametrize("fault",["floatmetadata","boolmetadata","duplicateimage","duplicatelabel","roles","relative"])
def test_clean_metadata_source_legality_is_shared_with_S5_reuse(monkeypatch,fault):
    manifest,reuse,sources,root=manifests(monkeypatch)
    if fault == "floatmetadata":sources[0]["label_metadata"][0]=123.
    elif fault == "boolmetadata":sources[0]["label_metadata"][0]=True
    elif fault == "duplicateimage":sources[1]["image"]=sources[0]["image"]
    elif fault == "duplicatelabel":sources[1]["label"]=sources[0]["label"]
    elif fault == "roles":sources[0]["image"]=sources[1]["label"]
    else:sources[0]["image"]="relative"
    with pytest.raises(ValueError):curve.validate_manifest(manifest,sources,root)
    with pytest.raises(ValueError):curve.validate_S5_reuse(reuse,sources)


@pytest.mark.parametrize("validator",["generated","reuse"])
@pytest.mark.parametrize("fault",["sizefloat","mtimefloat","flagint","details_tuple","extra_bool_int","severityfloat","indexfloat"])
def test_copied_annotation_provenance_types_are_exact_at_every_depth(monkeypatch,validator,fault):
    manifest,reuse,sources,root=manifests(monkeypatch)
    for source in sources:source["extra_provenance"]={"nested":[True,None,1.5]}
    for rows in [*manifest.values(),*reuse.values()]:
        for row in rows:row["extra_provenance"]={"nested":[True,None,1.5]}
    row=(manifest[curve.conditions()[-1]["id"]] if validator == "generated" else reuse[curve.CURVE_TYPES[-1]])[-1]
    if fault == "sizefloat":row["label_metadata"][0]=float(row["label_metadata"][0])
    elif fault == "mtimefloat":row["label_metadata"][1]=float(row["label_metadata"][1])
    elif fault == "flagint":row["label_metadata_details"]["contents_read"]=0
    elif fault == "details_tuple":row["extra_provenance"]["nested"]=tuple(row["extra_provenance"]["nested"])
    elif fault == "extra_bool_int":row["extra_provenance"]["nested"][0]=1
    elif fault == "severityfloat":row["severity"]=float(row["severity"])
    else:row["original_sample_idx"]=float(row["original_sample_idx"])
    def denied(*_a,**_k):pytest.fail("Typed provenance validator did file IO")
    monkeypatch.setattr(Path,"open",denied)
    monkeypatch.setattr(Path,"stat",denied)
    with pytest.raises(ValueError):
        if validator == "generated":curve.validate_manifest(manifest,sources,root)
        else:curve.validate_S5_reuse(reuse,sources)


def test_legal_extra_JSON_provenance_is_preserved_in_both_validators(monkeypatch):
    manifest,reuse,sources,root=manifests(monkeypatch)
    extra={"bool":False,"int":0,"float":0.,"null":None,"list":["x",{"nested":2.5}]}
    for source in sources:source["extra"]=copy.deepcopy(extra)
    for rows in [*manifest.values(),*reuse.values()]:
        for row in rows:row["extra"]=copy.deepcopy(extra)
    assert curve.validate_manifest(manifest,sources,root)["valid"]
    assert curve.validate_S5_reuse(reuse,sources)["valid"]
