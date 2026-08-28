from __future__ import annotations

import unittest
from types import MethodType

from safety_verify_wam.stage1.dataset import FailFastRobotVideoDataset


class FailFastRobotVideoDatasetTest(unittest.TestCase):
    def _dataset(self) -> FailFastRobotVideoDataset:
        dataset = object.__new__(FailFastRobotVideoDataset)
        dataset.video_latent_cache_dir = None
        return dataset

    def test_returns_the_requested_sample_without_random_replacement(self) -> None:
        dataset = self._dataset()

        def get_sample(_self: object, index: int) -> dict[str, object]:
            return {"_sample_idx": index, "value": index}

        dataset._get = MethodType(get_sample, dataset)
        self.assertEqual(dataset[17], {"value": 17})

    def test_propagates_missing_cache_errors(self) -> None:
        dataset = self._dataset()

        def fail(_self: object, _index: int) -> dict[str, object]:
            raise FileNotFoundError("missing prompt cache")

        dataset._get = MethodType(fail, dataset)
        with self.assertRaisesRegex(FileNotFoundError, "missing prompt cache"):
            _ = dataset[17]


if __name__ == "__main__":
    unittest.main()
