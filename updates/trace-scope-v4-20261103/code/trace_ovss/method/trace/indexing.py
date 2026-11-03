"""Transfer CPU connected-component indices to the model device in one copy."""

import numpy as np
import torch


def pack_region_indices(regions, device):
    """Keep row-major token order, including empty regions."""
    indices = [np.flatnonzero(region) for region in regions]
    if not indices:
        return ()
    lengths = [len(index) for index in indices]
    packed = torch.from_numpy(np.concatenate(indices)).to(device=device)
    return packed.split(lengths)
