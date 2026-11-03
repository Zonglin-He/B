"""CPU estimators for the prospectively fixed G analysis, not an admission."""

from __future__ import annotations

import hashlib
import io
import json

import cv2
import numpy as np
from PIL import Image

from .bootstrap import require
from .generalization import ARMS, CORRUPTIONS, CORRUPTION_GROUPS


def decode_annotation(content, classes, reduce_zero_label, shape=(224, 224)):
    require(type(classes) is int and 1 <= classes <= 81 and shape == (224, 224),
            "G annotation class/geometry bound differs")
    require(isinstance(content, bytes) and 0 < len(content) <= 16*1024**2,
            "G annotation compressed byte bound exceeded")
    # Preserve palette indices. OpenCV decoding a palette PNG may yield colors.
    with Image.open(io.BytesIO(content)) as image:
        require(image.format in ("PNG", "TIFF") and image.mode in ("P", "L", "I", "I;16", "RGB")
                and 0 < image.width*image.height <= 16*1024**2,
                "Unsupported or oversized G annotation")
        labels = np.asarray(image)
        if labels.ndim == 3:
            require(bool((labels == labels[:, :, :1]).all()), "Color-coded RGB GT is not an indexed annotation")
            labels = labels[:, :, 0]
        require(labels.ndim == 2 and np.issubdtype(labels.dtype, np.integer)
                and bool(((labels >= 0) & (labels <= 255)).all()), "Invalid G label indices")
        labels = cv2.resize(labels.astype(np.uint8), shape[::-1], interpolation=cv2.INTER_NEAREST)
    if reduce_zero_label:
        labels = labels.astype(np.int64)
        labels[labels == 0] = 256
        labels[labels != 255] -= 1
    require(bool(((labels == 255) | (labels < classes)).all()), "G annotation outside benchmark vocabulary")
    return labels.astype(np.uint16)


def counts(prediction, labels, classes):
    require(prediction.shape == labels.shape == (224, 224)
            and bool((prediction < classes).all())
            and bool(((labels == 255) | (labels < classes)).all()), "Invalid paired G labels")
    valid = labels != 255
    matrix = np.bincount(labels[valid].astype(np.int64)*classes+prediction[valid],
                         minlength=classes*classes).reshape(classes, classes)
    intersection = matrix.diagonal().copy()
    union = matrix.sum(0)+matrix.sum(1)-intersection
    return np.stack((intersection, union))


def miou(values):
    """Last two axes are I/U, class. Union-zero classes are omitted, not zero."""
    values = np.asarray(values)
    require(values.shape[-2] == 2 and bool(np.isfinite(values).all())
            and bool((values >= 0).all()) and bool((values[..., 0, :] <= values[..., 1, :]).all()),
            "Invalid G I/U counts")
    valid = values[..., 1, :] > 0
    ratios = np.divide(values[..., 0, :], values[..., 1, :],
                       out=np.zeros(values.shape[:-2]+(values.shape[-1],), dtype=np.float64), where=valid)
    return np.divide(100*ratios.sum(-1), valid.sum(-1),
                     out=np.full(values.shape[:-2], np.nan), where=valid.sum(-1) > 0)


def stream_seed(backbone, dataset, stage="primary", root_seed=20261102):
    key = json.dumps([root_seed, backbone, dataset, stage], separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(key.encode("ascii")).digest()
    return int.from_bytes(digest[:16], "little"), key


def bootstrap(episode_counts, backbone, dataset, *, replicates=10000, chunk=32):
    """Shared whole-episode draws over [episode, condition, arm, I/U, class]."""
    values = np.asarray(episode_counts)
    require(values.ndim == 5 and values.shape[0] > 0 and values.shape[2] == len(ARMS)
            and values.shape[3] == 2 and np.issubdtype(values.dtype, np.integer)
            and bool((values >= 0).all()) and bool((values[..., 0, :] <= values[..., 1, :]).all()),
            "Invalid episode bootstrap population")
    require(type(replicates) is int and 0 < replicates <= 10000 and type(chunk) is int and 0 < chunk <= 64,
            "G bootstrap resource bound exceeded")
    episodes = values.shape[0]
    # Nonnegative integer totals stay exactly representable in float64 BLAS.
    require(int(values.max())*episodes < 2**52, "G bootstrap counts exceed exact float64 integer bound")
    seed, key = stream_seed(backbone, dataset)
    rng = np.random.Generator(np.random.PCG64(seed))
    flattened = values.reshape(episodes, -1).astype(np.float64)
    result = np.empty((replicates, values.shape[1], len(ARMS)), dtype=np.float64)
    for start in range(0, replicates, chunk):
        size = min(chunk, replicates-start)
        draws = rng.integers(0, episodes, size=(size, episodes))
        weights = np.stack([np.bincount(row, minlength=episodes) for row in draws]).astype(np.float64)
        totals = (weights @ flattened).reshape((size,)+values.shape[1:])
        result[start:start+size] = miou(totals)
    return result, dict(key=key, seed_uint128=seed, replicates=replicates,
        unit="Original frozen source-image episode; same draws across conditions/arms",
        undefined_policy="Retain all draws; any undefined draw makes its interval null")


def interval(values):
    values = np.asarray(values, dtype=np.float64)
    undefined = int((~np.isfinite(values)).sum())
    return dict(CI95_pp=None if undefined else np.quantile(values, [.025, .975], method="linear").tolist(),
                undefined_draws=undefined, total_draws=int(values.size))


def finite_or_none(value):
    return float(value) if np.isfinite(value) else None


def describe(point, samples):
    require(point.shape == (len(ARMS),) and samples.ndim == 2 and samples.shape[1] == len(ARMS),
            "G summary arm dimensions differ")
    return dict(arms={arm:dict(mIoU=finite_or_none(point[i]), **interval(samples[:, i]))
                          for i, arm in enumerate(ARMS)},
                C_minus_Confidence_pp=finite_or_none(point[2]-point[1]),
                paired_difference=interval(samples[:, 2]-samples[:, 1]))


def corruption_summary(names, points, samples):
    require(len(names) == len(CORRUPTIONS) and set(names) == set(CORRUPTIONS),
            "Full15 corruption mean requires every predeclared condition")
    require(points.shape == (15, len(ARMS)) and samples.ndim == 3
            and samples.shape[1:] == points.shape, "G corruption summary dimensions differ")
    differences = points[:, 2]-points[:, 1]
    return dict(macro15=describe(points.mean(0), samples.mean(1)),
        groups={name:describe(points[[names.index(t) for t in types]].mean(0),
                             samples[:, [names.index(t) for t in types]].mean(1))
                for name, types in CORRUPTION_GROUPS.items()},
        C_versus_Confidence=dict(wins=int((differences > 0).sum()), ties=int((differences == 0).sum()),
            losses=int((differences < 0).sum()), undefined=int((~np.isfinite(differences)).sum())),
        aggregation="Arithmetic mean of condition-specific pooled mIoUs, never globally pooled corruptions")
