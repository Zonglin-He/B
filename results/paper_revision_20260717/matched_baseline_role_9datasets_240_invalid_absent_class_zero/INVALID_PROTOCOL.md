# Invalid protocol output

These files must not be used in the paper. The diagnostic runner originally
treated classes absent from the fixed 240-image subset as zero-IoU classes when
computing subset mIoU. Standard semantic-segmentation evaluation excludes those
classes from the mean. Region-level rows are retained only for debugging; the
formal suite was restarted after fixing `metric_summary`.
