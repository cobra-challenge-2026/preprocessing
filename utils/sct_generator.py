"""Standalone inference utilities for nnUNet-style regression models."""

import json
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.ndimage import gaussian_filter


# ============================================================================
# Helper Functions
# ============================================================================

def compute_steps_for_sliding_window(
    image_size: Tuple[int, ...],
    tile_size: Tuple[int, ...],
    tile_step_size: float
) -> List[List[int]]:
    """Return sliding-window start indices for each spatial dimension."""
    assert all(i >= j for i, j in zip(image_size, tile_size)), \
        "Image size must be >= tile size in all dimensions"
    assert 0 < tile_step_size <= 1, "tile_step_size must be in range (0, 1]"

    # Calculate target step size in voxels
    target_step_sizes_in_voxels = [int(i * tile_step_size) for i in tile_size]

    # Calculate number of steps needed in each dimension
    num_steps = [
        int(np.ceil((i - k) / j)) + 1
        for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)
    ]

    # Compute actual step positions
    steps = []
    for dim in range(len(tile_size)):
        max_step_value = image_size[dim] - tile_size[dim]

        if num_steps[dim] > 1:
            # Evenly distribute steps
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # Doesn't matter, only one step at 0

        steps_here = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]
        steps.append(steps_here)

    return steps


@lru_cache(maxsize=4)
def compute_gaussian_weight(
    tile_size: Tuple[int, ...],
    sigma_scale: float = 1.0 / 8,
    value_scaling_factor: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: Union[str, torch.device] = "cuda"
) -> torch.Tensor:
    """Build a cached Gaussian weight map for patch blending."""
    # Create zero array and set center to 1
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1

    # Apply Gaussian filter
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)

    # Convert to PyTorch tensor
    gaussian_importance_map = torch.from_numpy(gaussian_importance_map)

    # Normalize by max value
    gaussian_importance_map /= (torch.max(gaussian_importance_map) / value_scaling_factor)
    gaussian_importance_map = gaussian_importance_map.to(device=device, dtype=dtype)

    # Ensure no zeros to prevent NaN when dividing
    mask = gaussian_importance_map == 0
    if mask.any():
        gaussian_importance_map[mask] = torch.min(gaussian_importance_map[~mask])

    return gaussian_importance_map


# ============================================================================
# Main Inference Class
# ============================================================================

class StandaloneRegressionInference:
    """Run TorchScript regression inference with optional tiled prediction."""

    def __init__(
        self,
        model_path: Union[str, Path],
        device: Union[str, torch.device] = "cuda"
    ):
        """Load model bundle (`model.pt`, `metadata.json`) on the given device."""
        self.model_path = Path(model_path)
        self.device = torch.device(device) if isinstance(device, str) else device

        # Load metadata
        metadata_path = self.model_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        # Load TorchScript model
        model_file = self.model_path / "model.pt"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")

        print(f"Loading TorchScript model from {model_file}...")
        self.model = torch.jit.load(str(model_file), map_location=self.device)
        self.model.eval()

        # Extract inference configuration
        self.patch_size = tuple(self.metadata['inference_config']['patch_size'])
        self.tile_step_size = self.metadata['inference_config']['tile_step_size']
        self.use_gaussian = self.metadata['inference_config']['use_gaussian']

        # Precompute Gaussian weights for efficiency
        if self.use_gaussian:
            self._gaussian_cache = compute_gaussian_weight(
                self.patch_size,
                sigma_scale=1.0 / 8,
                value_scaling_factor=10.0,  # nnUNet uses 10 for better blending
                dtype=torch.float32,
                device=self.device
            )
        else:
            self._gaussian_cache = None

        print(f"Loaded model: {self.metadata['model_info']['trainer_name']}")
        print(f"Patch size: {self.patch_size}, Tile step size: {self.tile_step_size}")
        print(f"Gaussian blending: {self.use_gaussian}")

    def _normalize(self, data: np.ndarray, channel: str = 'input') -> np.ndarray:
        """Apply metadata-based normalization to input/output arrays."""
        norm_config = self.metadata['normalization'][channel]
        scheme = norm_config['scheme']

        # Convert to float32 for normalization
        data = data.astype(np.float32, copy=False)

        if scheme == 'ZScoreNormalization':
            mean = norm_config['mean']
            std = norm_config['std']
            data = (data - mean) / max(std, 1e-8)

        elif scheme == 'CTNormalization':
            mean = norm_config['mean']
            std = norm_config['std']
            lower = norm_config['percentile_00_5']
            upper = norm_config['percentile_99_5']
            # Clip then normalize
            np.clip(data, lower, upper, out=data)
            data = (data - mean) / max(std, 1e-8)

        elif scheme == 'GlobalNormalization':
            mean = norm_config['mean']
            std = norm_config['std']
            data = (data - mean) / max(std, 1e-8)

        elif scheme == 'NoNormalization':
            # No normalization
            pass

        elif scheme == 'RescaleTo01Normalization':
            data = data - data.min()
            data = data / np.clip(data.max(), a_min=1e-8, a_max=None)

        else:
            raise ValueError(f"Unknown normalization scheme: {scheme}")

        return data

    def _denormalize(self, data: np.ndarray, channel: str = 'output') -> np.ndarray:
        """Undo normalization for channels that store reversible stats."""
        norm_config = self.metadata['normalization'][channel]
        scheme = norm_config['scheme']

        if scheme in ['ZScoreNormalization', 'CTNormalization', 'GlobalNormalization']:
            mean = norm_config['mean']
            std = norm_config['std']
            # Reverse: x_orig = (x_norm * std) + mean
            data = data * std + mean

        elif scheme == 'NoNormalization':
            # No denormalization needed
            pass

        elif scheme == 'RescaleTo01Normalization':
            # This would need original min/max, which aren't stored
            # Return as-is
            pass

        else:
            raise ValueError(f"Unknown normalization scheme: {scheme}")

        return data

    def _sliding_window_inference(
        self,
        data: torch.Tensor,
        tile_step_size: Optional[float] = None
    ) -> torch.Tensor:
        """Run tiled inference and blend overlaps with Gaussian weights."""
        # Use provided tile_step_size or default
        effective_step_size = tile_step_size if tile_step_size is not None else self.tile_step_size

        # Get spatial dimensions (excluding batch and channel)
        data_shape = data.shape[2:]
        num_dimensions = len(data_shape)

        # Check if image is smaller than patch size
        if any(i < j for i, j in zip(data_shape, self.patch_size)):
            # For small images, just run inference directly
            with torch.no_grad():
                prediction = self.model(data)
            return prediction

        # Compute sliding window steps
        steps = compute_steps_for_sliding_window(data_shape, self.patch_size, effective_step_size)

        # Initialize output accumulators
        predicted_logits = torch.zeros(
            (1, 1) + data_shape,
            dtype=torch.float32,
            device=self.device
        )
        n_predictions = torch.zeros(
            data_shape,
            dtype=torch.float32,
            device=self.device
        )

        # Get Gaussian weights if enabled
        if self.use_gaussian and self._gaussian_cache is not None:
            gaussian = self._gaussian_cache
        else:
            gaussian = torch.ones(self.patch_size, dtype=torch.float32, device=self.device)

        # Iterate over all patch positions
        if num_dimensions == 3:
            # 3D case
            for x in steps[0]:
                for y in steps[1]:
                    for z in steps[2]:
                        # Extract patch
                        patch = data[
                            :, :,
                            x:x+self.patch_size[0],
                            y:y+self.patch_size[1],
                            z:z+self.patch_size[2]
                        ]

                        # Run inference
                        with torch.no_grad():
                            prediction = self.model(patch)

                        # Apply Gaussian weighting
                        if self.use_gaussian:
                            prediction = prediction * gaussian

                        # Accumulate
                        predicted_logits[
                            :, :,
                            x:x+self.patch_size[0],
                            y:y+self.patch_size[1],
                            z:z+self.patch_size[2]
                        ] += prediction

                        n_predictions[
                            x:x+self.patch_size[0],
                            y:y+self.patch_size[1],
                            z:z+self.patch_size[2]
                        ] += gaussian

        elif num_dimensions == 2:
            # 2D case
            for x in steps[0]:
                for y in steps[1]:
                    # Extract patch
                    patch = data[
                        :, :,
                        x:x+self.patch_size[0],
                        y:y+self.patch_size[1]
                    ]

                    # Run inference
                    with torch.no_grad():
                        prediction = self.model(patch)

                    # Apply Gaussian weighting
                    if self.use_gaussian:
                        prediction = prediction * gaussian

                    # Accumulate
                    predicted_logits[
                        :, :,
                        x:x+self.patch_size[0],
                        y:y+self.patch_size[1]
                    ] += prediction

                    n_predictions[
                        x:x+self.patch_size[0],
                        y:y+self.patch_size[1]
                    ] += gaussian
        else:
            raise ValueError(f"Unsupported number of dimensions: {num_dimensions}")

        # Normalize by cumulative weights
        predicted_logits = predicted_logits / n_predictions

        return predicted_logits

    def predict(
        self,
        input_array: np.ndarray,
        apply_normalization: bool = True,
        apply_denormalization: bool = True,
        tile_step_size: Optional[float] = None
    ) -> np.ndarray:
        """Predict on 2D/3D numpy input with optional (de)normalization."""
        # Ensure input has channel dimension
        if input_array.ndim == 3:
            # (H, W, D) -> (1, H, W, D)
            input_array = input_array[np.newaxis, ...]
        elif input_array.ndim == 2:
            # (H, W) -> (1, H, W)
            input_array = input_array[np.newaxis, ...]

        # Normalize if requested
        if apply_normalization:
            input_array = self._normalize(input_array, channel='input')

        # Convert to PyTorch tensor: (C, H, W, D) -> (1, C, H, W, D)
        input_tensor = torch.from_numpy(input_array).float()
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        # Run sliding window inference
        prediction = self._sliding_window_inference(input_tensor, tile_step_size)

        # Convert back to NumPy
        prediction_np = prediction.squeeze(0).cpu().numpy()  # (1, H, W, D) -> (C, H, W, D)

        # Denormalize if requested
        if apply_denormalization:
            prediction_np = self._denormalize(prediction_np, channel='output')

        # Remove channel dimension to match input: (1, H, W, D) -> (H, W, D)
        if prediction_np.shape[0] == 1:
            prediction_np = prediction_np[0]

        return prediction_np
