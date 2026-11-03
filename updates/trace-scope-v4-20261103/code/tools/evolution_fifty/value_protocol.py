"""Prospective TRACE value/OVSS extension; no models, outcomes or tuning."""

from __future__ import annotations

import math

import numpy as np

from .bootstrap import require

EPS = 1e-8


def budget_steps(native_steps):
    require(type(native_steps) is int and native_steps > 0, "Invalid native step count")
    return dict(image_C=sorted({max(1, math.ceil(native_steps/2)), native_steps}),
                image_confidence=[native_steps, 2*native_steps, 4*native_steps])


def score_variants(confidence, C):
    confidence, C = np.asarray(confidence, dtype=np.float64), np.asarray(C, dtype=np.float64)
    require(confidence.shape == C.shape and confidence.ndim == 1 and len(C) > 0,
            "Paired nonempty scores required")
    require(bool(np.isfinite(confidence).all()) and bool(np.isfinite(C).all())
            and bool(((confidence >= 0) & (confidence <= 1)).all())
            and bool(((C >= 0) & (C <= 1)).all()), "Scores must be probabilities")
    return dict(C=C.copy(), relative=C/(confidence+EPS), product=confidence*C, confidence=confidence.copy())


def protocol():
    return dict(scope="TRACE value extension following actual human request2026-11-02; no external baseline reproduction",
        G04=dict(implementation="SCLIP", backbone="ViT-B/16", dataset="voc20", population=1449,
            arms=["no_adapt", "image_confidence", "image_C"], new_cells=3,
            comparison="Reuse G01 NACLIP B/16 VOC20 only if exact inputs/config/layers/prompts/reset/readout agree",
            difference="OVSS spatial implementation only; each model builds its own candidates and response scores",
            source="https://github.com/wangf3014/SCLIP",
            admission="Existing local csa switch is not proof of official SCLIP equivalence; pin official source and check actual tensor/loss/gradient/readout gate first",
            claim="Only the tested backbone/dataset and explicitly validated SCLIP integration; no universal OVSS framework claim"),
        V01=dict(datasets=["coco_object", "pc60"], backbone="ViT-L/14", steps={"coco_object":budget_steps(30), "pc60":budget_steps(7)},
            timing="Synchronized model-side time including reset, original candidate scoring, C diagnostic if any, adaptation and final readout; exclude RGBdecode/H2D/disk/GT for every arm",
            auxiliary_checkpoints="Separately count/time any1/mid/final diagnostic readouts; do not hide them in end-to-end cost or include them asymmetrically in primary model time",
            primary="Accuracy versus measured time and actual forward-input/update counts, not equal step count",
            budget_match="For each official C native point, find Confidence grid point with smallest abs(log(timeConf/timeC)); tie fewer steps; timing only, never mIoU",
            closeness="Call near-time only if time ratio lies in[0.9,1.1], otherwise report no matched point and retain whole curve; no synthetic interpolated mIoU",
            reuse="Existing native C/Confidence anchors only when exact input/model/candidate/order/config and actual timing/counter scope match; otherwise do not claim cost matching",
            frozen_grid_conditions=10, image_method_evaluations_if_all_run=50525,
            no_per_image_early_stop=True, no_test_selection=True, failures="Retain; do not skip failed settings or retry automatically"),
        A01=dict(datasets=["coco_object", "pc60"], preferred="Existing sealed candidate receipts, M05 per-image transitions and canonical class I/U",
            grouping_before_outcomes=["foreground/background", "thing/stuff", "candidate area", "single/multiple candidate sources", "original confidence", "C-confidence ordering disagreement"],
            metric_separation="Class-subset macro IoU, source-group pixel transitions and candidate-local correctness/update effects are distinct estimands",
            role_metadata="Pinned historical benchmark class-role taxonomy used offline only; never returned to deployment loss/scorer",
            image_ordering="Exact comparable candidate pairs; report all ties/no-pairs/zero-EPS support; same rank can still have different numeric weights",
            uncertainty="10000 original-episode paired bootstrap, seed20261102,95pointwise percentile; report undefined denominators/empty groups, never hide them",
            claim="Descriptive conditional associations at existing TRACE-selected historical settings, not isolated candidate causality",
            new_GPU_required=False),
        optional=dict(relative_scores="C, C/(confidence+1e-8), confidence*C, confidence. Reuse views only for exactly identical model/input/candidate/probe identities; new weights require new adaptation; no post-test default replacement",
            admission_policy="Native confidence TopK vs fixed-seed randomK vs areaK; requires the genuine PRE-TopK pool, not resampling the already confidence-pruned cache",
            mask_boundary="Diagnostic preserve-mask only; original readout/loss stay fixed; report changed retained content and transformed area",
            supervised_reference="Offline diagnostic only after prediction sealing; GT never in deployment; isolated-candidate utility is not additive and not an upper bound"),
        trajectories=dict(current_M01="Frozen running1540 records0/1/native only; no mid-run edit/restart to add midpoint",
            future="Where natural new runs permit, prefreeze1/ceil(steps/2)/final checkpoints, fresh pipeline gate and separate auxiliary costs",
            no_new_algorithm="Do not recompute C every step or infer missing midpoint states from final states"),
        unchanged="Existing M01-M04/G01-G03 remain; K/weight/direction/admission/mask/supervised extras are not all prerequisites",
        overall_scope_boundary="No more unspecified expansion; any later request needs a new explicit scope version",
        executable_queue=False, scientific_completion=False)
