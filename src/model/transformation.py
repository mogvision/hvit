import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Union



class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    Obtained from https://github.com/voxelmorph/voxelmorph
    """
    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=False, mode=self.mode)


def normalize_displacement(displacement: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Spatially normalize the displacement vector field to the [-1, 1] coordinate system utilized by PyTorch's `grid_sample()` function.
    
    This function assumes that the displacement field size is identical to the corresponding image dimensions.

    Args:
        displacement (Union[np.ndarray, torch.Tensor]): The displacement field with shape (N, ndim, *size).

    Returns:
        Union[np.ndarray, torch.Tensor]: The normalized displacement field.

    Raises:
        TypeError: If the input type is neither numpy.ndarray nor torch.Tensor.
    """
    number_of_dimensions = displacement.ndim - 2

    if isinstance(displacement, np.ndarray):
        normalization_factors = 2.0 / np.array(displacement.shape[2:])
        normalization_factors = normalization_factors.reshape(1, number_of_dimensions, *(1,) * number_of_dimensions)

    elif isinstance(displacement, torch.Tensor):
        normalization_factors = torch.tensor(2.0) / torch.tensor(
            displacement.size()[2:], dtype=displacement.dtype, device=displacement.device)
        normalization_factors = normalization_factors.view(1, number_of_dimensions, *(1,) * number_of_dimensions)

    else:
        raise TypeError("Input data type not recognized. Expected numpy.ndarray or torch.Tensor.")

    return displacement * normalization_factors

