import pytest

from tools.evolution_fifty.symmetric_calibration import candidates, select
from tools.evolution_tuning.common import canonical_hash


def test_common_grid_and_anchor_identical_for_all_arms():
    anchor = dict(batch_size=13, lr=.00375, steps=30, trace_max_regions=35, fixed='unchanged')
    grid = candidates(anchor)
    assert len(grid) == 25 and anchor in grid
    assert all(r['fixed'] == 'unchanged' for r in grid)
    assert len({canonical_hash(r) for r in grid}) == 25
    assert candidates(anchor) == grid


def test_selection_requires_complete_trials_and_no_full_score():
    configs = [dict(steps=5, trace_max_regions=16, batch_size=2, lr=.001),
               dict(steps=30, trace_max_regions=16, batch_size=2, lr=.001)]
    rows = [dict(config=c, config_sha256=canonical_hash(c), calibration_miou=10., status='completed', full_miou=100-i)
            for i, c in enumerate(configs)]
    keys = [r['config_sha256'] for r in rows]
    assert select(rows, keys)['config']['steps'] == 5
    rows[1]['calibration_miou'] = 10.01
    assert select(rows, keys)['config']['steps'] == 30
    with pytest.raises(ValueError):
        select(rows[:1], keys)
    rows[1]['status'] = 'failed'
    with pytest.raises(ValueError):
        select(rows, keys)
