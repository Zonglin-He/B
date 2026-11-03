import copy
import io

import numpy as np
from PIL import Image
import pytest

from tools.evolution_fifty import case_payload as core


def png(array, palette=False):
    image = Image.fromarray(array)
    if palette:
        image = image.convert("P")
        image.putpalette([c for i in range(256) for c in (255-i,i,i//2)])
    buffer = io.BytesIO();image.save(buffer,format="PNG");return buffer.getvalue()


def npy(array):
    buffer = io.BytesIO();np.save(buffer,array,allow_pickle=False);return buffer.getvalue()


def fixture(index=0, *, empty=False):
    spec = dict(dataset="coco_object",images=2,modes=list(core.MODES),preprocessing="native_PIL_bilinear",
                config=dict(batch_size=2,resize=[224,224],trace_max_regions=2),
                vocabulary=dict(classes=["background","one","two"],queries=["background","one","two"],reduce_zero_label=False))
    case = dict(dataset="coco_object",index=index,official_id=f"source-{index}",episode=0,batch_idx=index)
    rgb = png(np.full((3,5,3),100,dtype=np.uint8))
    gt = png(np.array([[0,1,255],[1,0,1]],dtype=np.uint8),palette=True)
    predictions = {m:npy(np.full((224,224),i%3,dtype=np.uint16)) for i,m in enumerate(core.MODES)}
    mask = np.zeros((16,16),dtype=bool);mask[0,0]=True
    candidate = dict(batch_idx=index,image_idx=index,class_idx=1,component_idx=1,area=1,confidence=.8,reliability=.4,
                     region=dict(shape=[16,16],encoding="packbits-little-hex",true_pixels=1,data=np.packbits(mask.ravel(),bitorder="little").tobytes().hex()))
    _,geometry = core.translated_candidate(np.repeat(np.repeat(mask,14,axis=0),14,axis=1))
    records = {mode:dict(GT_opened=False,scientific_identity="M02_spatial_"+mode,episode=0,
        ordered_ids=["source-0","source-1"],candidate_sha256="a"*64,candidates=[] if empty else [copy.deepcopy(candidate)],
        preservation_geometry=[] if empty else [dict(geometry,candidate_index=0,batch_idx=index,mode=mode)]) for mode in core.MODES}
    rebind(records)
    return rgb,gt,predictions,records,spec,case


def rebind(records):
    for record in records.values():
        decoded = [dict(c,region=core.candidate_mask(c)[::14,::14].copy()) for c in record["candidates"]]
        record["candidate_sha256"] = core.candidate_identity(decoded)


@pytest.mark.parametrize("batch",[1,2,13])
def test_selection_is_exact_first_two_original_sources_without_outcomes(batch):
    source = dict(rows={},specs=[])
    for dataset in core.DATASETS:
        source["rows"][dataset] = [dict(official_id=f"{dataset}-{i}",forbidden_outcome=None) for i in range(5)]
        source["specs"].append(dict(dataset=dataset,images=5,modes=list(core.MODES),config=dict(batch_size=batch,resize=[224,224])))
    result = core.select_cases(source)
    assert len(result) == 4
    assert [(r["dataset"],r["index"],r["episode"],r["batch_idx"]) for r in result] == [(d,i,i//batch,i%batch) for d in core.DATASETS for i in (0,1)]
    assert core.protocol()["no_new_scores"] and core.protocol()["no_inference"]
    assert "not population" not in str(result)  # Selection is only exact source metadata.


@pytest.mark.parametrize("index,empty",[(0,False),(1,False),(0,True)])
def test_eight_authentic_panels_known_pixels_and_no_input_mutation(index,empty,monkeypatch):
    args = fixture(index,empty=empty)
    before = copy.deepcopy(args)
    # No filepath IO is needed to render the supplied, separately admitted bytes.
    from pathlib import Path
    monkeypatch.setattr(Path,"open",lambda *a,**kw:pytest.fail("Pure renderer opened a path"))
    data,metadata = core.render_case(*args)
    assert args == before
    with Image.open(io.BytesIO(data)) as image:
        assert image.size == tuple(metadata["PNG_size"]) and image.width == 960 and image.mode == "RGB"
        pixels = np.asarray(image)
        top = metadata["tile_top"]
        assert (pixels[top+18,16] == 100).all()
        colors = core.palette(3)
        assert (pixels[top+18,16+240] == colors[0]).all()
        assert (pixels[top+18,16+480] == colors[0]).all()  # Confidence=mode index3 -> class0.
        assert (pixels[top+18,16+720] == colors[0]).all()
        assert (pixels[top+256+18,16+720] == colors[2]).all()  # Global=mode index2.
    assert metadata["case"]["index"] == index and metadata["new_inference"] is False
    assert metadata["new_scores"] is False and metadata["population_effect_or_causality_claim"] is False
    assert metadata["panel_order"] == core.protocol()["panels"]
    assert len(metadata["class_legend"]) == 3
    if empty:
        assert metadata["candidate"] is None and "no substitute" in metadata["candidate_status"]
    else:
        assert metadata["candidate"]["confidence"] == .8 and metadata["candidate"]["C"] == .4
        assert metadata["candidate"]["geometry"]["original_area"] == 196
        assert metadata["candidate"]["geometry"]["shifted_area"] == 196


@pytest.mark.parametrize("fault",["caseindex","boolindex","episode","sourceid","mode","digest","maskarea","maskencoding","geometry","score","query","pred_dtype","pred_shape","pred_class","RGB_GT"])
def test_invalid_supplied_bindings_and_pixels_failclosed(fault):
    rgb,gt,preds,records,spec,case = fixture()
    if fault == "caseindex":case["index"] = 2
    elif fault == "boolindex":case["index"] = False
    elif fault == "episode":records["native"]["episode"] = True
    elif fault == "sourceid":case["official_id"] = "other"
    elif fault == "mode":records["native"]["scientific_identity"] = "wrong"
    elif fault == "digest":records["native"]["candidate_sha256"] = "g"*64
    elif fault == "maskarea":records["native"]["candidates"][0]["area"] = 2
    elif fault == "maskencoding":records["native"]["candidates"][0]["region"]["encoding"] = "wrong"
    elif fault == "geometry":records["translated"]["preservation_geometry"][0]["dx"] += 1
    elif fault == "score":records["native"]["candidates"][0]["confidence"] = float("nan")
    elif fault == "query":records["native"]["candidates"][0]["class_idx"] = 3
    elif fault == "pred_dtype":preds["native"] = npy(np.zeros((224,224),dtype=np.float32))
    elif fault == "pred_shape":preds["native"] = npy(np.zeros((223,224),dtype=np.uint16))
    elif fault == "pred_class":preds["native"] = npy(np.full((224,224),3,dtype=np.uint16))
    elif fault == "RGB_GT":gt = png(np.array([[[255,0,0]]],dtype=np.uint8))
    with pytest.raises((ValueError,KeyError)):
        core.render_case(rgb,gt,preds,records,spec,case)


def test_native_PIL_GT_indices_are_not_palette_colors_or_G_cv2_nearest():
    indices = np.array([[0,1,255],[1,0,1]],dtype=np.uint8)
    content = png(indices,palette=True)
    actual = core.annotation(content,3,False)
    expected = np.asarray(Image.fromarray(indices).resize((224,224),Image.Resampling.NEAREST))
    np.testing.assert_array_equal(actual,expected)
    import cv2
    assert not np.array_equal(actual,cv2.resize(indices,(224,224),interpolation=cv2.INTER_NEAREST))


def test_empty_source_not_substituted_with_another_sources_candidate():
    args = list(fixture(index=1))
    for row in args[3].values():
        row["candidates"][0]["batch_idx"] = row["candidates"][0]["image_idx"] = 0
    rebind(args[3])
    data,metadata = core.render_case(*args)
    assert data and metadata["candidate"] is None


@pytest.mark.parametrize("fault",["confidence","empty_native","image_idx","other_order","bool_batch"])
def test_native_content_and_source_identity_cannot_hide_behind_old_digest(fault):
    args = list(fixture())
    row = args[3]["native"]
    if fault == "confidence":row["candidates"][0]["confidence"] = .123
    elif fault == "empty_native":row["candidates"] = []
    elif fault == "image_idx":row["candidates"][0]["image_idx"] = 1
    elif fault == "other_order":args[3]["confidence"]["ordered_ids"][1] = "other"
    else:args[3]["translated"]["preservation_geometry"][0]["batch_idx"] = False
    with pytest.raises(ValueError):core.render_case(*args)


def test_modes_can_have_distinct_legitimate_C_without_reselection():
    args = list(fixture())
    args[3]["translated"]["candidates"][0]["reliability"] = .123
    data,metadata = core.render_case(*args)
    assert data and metadata["candidate"]["C"] == .4


def test_pixel_measured_long_ID_and_query_keep_all_raw_scores_visible(monkeypatch):
    args = list(fixture())
    args[5]["official_id"] = "W"*256
    for record in args[3].values():record["ordered_ids"][0] = "W"*256
    args[4]["vocabulary"]["queries"][1] = "W"*110
    from PIL import ImageDraw
    original = ImageDraw.ImageDraw.text
    drawn = []
    def observer(draw,position,text,*a,**kw):
        bbox = draw.textbbox(position,text,font=kw["font"])
        drawn.append((text,bbox))
        return original(draw,position,text,*a,**kw)
    monkeypatch.setattr(ImageDraw.ImageDraw,"text",observer)
    data,metadata = core.render_case(*args)
    width,height = metadata["PNG_size"]
    assert all(0 <= b[0] <= b[2] <= width and 0 <= b[1] <= b[3] <= height for _,b in drawn)
    assert any("confidence=0.8" in text and "C=0.4" in text for text,_ in drawn)
    assert sum(text.count("W") for text,_ in drawn) == 366
    assert data and metadata["case"]["official_id"] == "W"*256
