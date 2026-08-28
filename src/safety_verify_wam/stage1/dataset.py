from __future__ import annotations

import os
from typing import Any

from ahawam.datasets.lerobot.robot_video_dataset import RobotVideoDataset


class FailFastRobotVideoDataset(RobotVideoDataset):
    """Keep the released dataset behavior but never replace a failed sample."""

    def __getitem__(self, idx: int) -> dict[str, Any]:
        data = self._get(idx)
        sample_idx = int(data.pop("_sample_idx"))
        if self.video_latent_cache_dir is not None:
            data["video_latent_cache_path"] = os.path.join(
                self.video_latent_cache_dir,
                f"{sample_idx:09d}.pt",
            )
        return data
