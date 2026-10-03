"""SubCell-specific preprocessing utilities (rescale → crop → normalize).

Used by:
  - scripts/08a_extract_subcell_embeddings.py (frozen inference)
  - scripts/08d_extract_subcell_finetune_embeddings.py (fine-tuned inference)
  - src/prot_loc_benchmark/representations/subcell_finetune.py (training dataset)
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from prot_loc_benchmark.config import SUBCELL_SCALE_FACTOR


class SubCellPreprocessor:
    """HPA-scale preprocessing for SubCell ViT embeddings.

    Physical-scale rescaling: upscale MisLocus crops (0.598 µm/px) to match
    HPA confocal pixel size (0.0801 µm/px), then center-crop to the ViT
    training resolution (448×448). This ensures each 16×16 patch covers
    1.28 µm — the same as in HPA training.

    Parameters
    ----------
    scale_factor : float
        Ratio of MisLocus to HPA pixel sizes (default: 7.47).
    input_size : int
        Input crop size in pixels (default: 128).
    output_size : int
        Target output size matching ViT training (default: 448).
    """

    def __init__(
        self,
        scale_factor: float = SUBCELL_SCALE_FACTOR,
        input_size: int = 128,
        output_size: int = 448,
    ):
        self.input_size = input_size
        self.scale_factor = scale_factor
        self.rescaled_size = int(input_size * scale_factor)
        self.output_size = output_size
        self.crop_offset = (self.rescaled_size - output_size) // 2

    @staticmethod
    def min_max_standardize(im: torch.Tensor) -> torch.Tensor:
        """Per-cell global min-max normalization to [0, 1].

        Matches SubCellPortable's inference.py:min_max_standardize exactly:
        global across all channels and spatial dims per cell.
        """
        min_val = torch.amin(im, dim=(1, 2, 3), keepdim=True)
        max_val = torch.amax(im, dim=(1, 2, 3), keepdim=True)
        return (im - min_val) / (max_val - min_val + 1e-6)

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Full preprocessing pipeline: rescale → crop → normalize.

        Parameters
        ----------
        tensor : torch.Tensor
            Input images of shape (N, C, input_size, input_size), float32.

        Returns
        -------
        torch.Tensor
            Preprocessed images of shape (N, C, output_size, output_size),
            float32, normalized to [0, 1].
        """
        if tensor.ndim != 4 or tensor.shape[-2:] != (self.input_size, self.input_size):
            raise ValueError('Invalid native SubCell crop geometry')
        if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
            raise ValueError('SubCell preprocessing requires finite float32 inputs')
        x = F.interpolate(
            tensor, size=self.rescaled_size, mode="bilinear", align_corners=False
        )
        o = self.crop_offset
        x = x[:, :, o : o + self.output_size, o : o + self.output_size]
        return self.min_max_standardize(x)
