# Pre-Score Official Field Repair

The V1 GT-free fit genuinely succeeded and remains unchanged. Before any real
ablation GT scoring, interface inspection found that the uncollated upstream
dataset returns gt_seg_map, while gt exists only after custom_collate. The V1
scorer indexed the wrong field. No V1 scientific GT score was executed.

V2 copies the frozen CPU worker with only the target-key repair plus new source
and test pins. A real official pipeline test uses synthetic RGB/masks, nearest
target geometry and ignore255 intersection/union; no actual scientific GT is
used by that test. The V1 finite controller is stopped without signalling its
separate adopted GPU child; all original bytes, unit identity and journal remain.
The V2 finite controller adopts the same original Object prediction, not a rerun.

No method/model/config/geometry/metric/resource policy or main score changes.
Any execution failure still needs a distinct changed-code repair and fresh root.
