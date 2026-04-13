"""RandomWindowCrop — XLA-friendly random crop utility. JAX port."""

from __future__ import annotations

import math
from typing import List, Tuple

import jax
import jax.numpy as jnp


def _linspace_indices(limit: int, count: int) -> List[int]:
    if count <= 1:
        return [0]
    return sorted({int(round(i * (limit / (count - 1)))) for i in range(count)})


def _gen_positions_1d(length: int, crop: int, slots: int) -> List[int]:
    limit = max(length - crop, 0)
    pos = _linspace_indices(limit, max(slots, 1))
    pos = [max(0, min(p, limit)) for p in pos]
    if slots > 1:
        pos[0] = 0
        pos[-1] = limit
    return pos


class RandomWindowCrop:
    """Random crop with a fixed catalog of windows (XLA-friendly)."""

    def __init__(
        self,
        input_size: int | Tuple[int, int],
        crop: int,
        num_windows: int,
        per_sample: bool = False,
    ):
        if isinstance(input_size, int):
            self.H = self.W = int(input_size)
        else:
            self.H, self.W = map(int, input_size)
        self.crop = int(crop)
        self.per_sample = bool(per_sample)

        rows_min = math.ceil(self.H / self.crop)
        cols_min = math.ceil(self.W / self.crop)

        t_rows = _gen_positions_1d(self.H, self.crop, rows_min)
        l_cols = _gen_positions_1d(self.W, self.crop, cols_min)
        base_offsets = [(t, l) for t in t_rows for l in l_cols]

        offsets = list(base_offsets)
        if num_windows > len(offsets):
            rows_t = max(rows_min, int(math.floor(math.sqrt(num_windows * self.H / self.W))))
            cols_t = max(cols_min, int(math.ceil(num_windows / rows_t)))
            while rows_t * cols_t < num_windows:
                cols_t += 1
            t_more = _gen_positions_1d(self.H, self.crop, rows_t)
            l_more = _gen_positions_1d(self.W, self.crop, cols_t)
            dense = [(t, l) for t in t_more for l in l_more]
            seen = set(offsets)
            for off in dense:
                if len(offsets) >= num_windows:
                    break
                if off not in seen:
                    offsets.append(off)
                    seen.add(off)
            idx = 0
            while len(offsets) < num_windows and idx < len(dense):
                offsets.append(dense[idx])
                idx += 1

        self.offsets = offsets[:num_windows]
        self.num_windows = len(self.offsets)

    def __call__(self, tensor: jnp.ndarray, rng: jax.random.PRNGKey) -> jnp.ndarray:
        """Crop tensor [..., H, W] using a randomly selected window."""
        idx = jax.random.randint(rng, (), 0, self.num_windows)
        # Use lax.switch for XLA compatibility
        def _crop_at(i, t):
            top, left = self.offsets[i]
            return t[..., top:top + self.crop, left:left + self.crop]

        # For JIT compatibility, pre-compute all crops and select
        crops = jnp.stack([tensor[..., t:t + self.crop, l:l + self.crop]
                           for t, l in self.offsets])
        return crops[idx]
