# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors.
# Modifications Copyright 2025 MarineSTD-GS authors.
#
# This file is derived from Nerfstudio dataset utilities and adapted for
# MarineSTD-GS underwater reconstruction training.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Underwater dataset for MarineSTD-GS.
"""

from __future__ import annotations

from typing import Dict
import cv2
import numpy as np

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.data.utils.data_utils import get_depth_image_from_path


class UnderwaterDataset(InputDataset):
    """Input dataset for underwater scene reconstruction.

    This dataset extends Nerfstudio's InputDataset by loading per-image
    relative depth priors and exposing required image identifiers from metadata.

    Args:
        dataparser_outputs: Description of where and how to read input images.
        scale_factor: Scaling factor applied to dataparser outputs.
        cache_compressed_images: Unused placeholder for compatibility.
    """

    exclude_batch_keys_from_device = InputDataset.exclude_batch_keys_from_device + ["depth_image"]

    def __init__(
        self,
        dataparser_outputs: DataparserOutputs,
        scale_factor: float = 1.0,
        cache_compressed_images: bool = False,
    ):
        super().__init__(dataparser_outputs, scale_factor)

        metadata = self.metadata or {}

        if "depth_filenames" not in metadata or metadata["depth_filenames"] is None:
            raise ValueError(
                "MarineSTD-GS requires 'depth_filenames' in dataparser metadata, "
                "but it was not found."
            )
        if "split_image_ids" not in metadata:
            raise ValueError(
                "MarineSTD-GS requires 'split_image_ids' in dataparser metadata, "
                "but it was not found."
            )

        self.depth_filenames = metadata["depth_filenames"]
        self.depth_unit_scale_factor = metadata.get("depth_unit_scale_factor", 1.0)
        self.split_image_ids = metadata["split_image_ids"]

    def get_metadata(self, data: Dict) -> Dict:
        """Returns extra per-image metadata for the current sample."""
        out: Dict = {}

        image_idx = data["image_idx"]

        filepath = self.depth_filenames[image_idx]
        height = int(self._dataparser_outputs.cameras.height[image_idx])
        width = int(self._dataparser_outputs.cameras.width[image_idx])

        # Scale depth images to meter units and also by scaling applied to cameras.
        scale_factor = self.depth_unit_scale_factor * self._dataparser_outputs.dataparser_scale

        # Original Nerfstudio-style depth loading branch:
        # depth_image = get_depth_image_from_path(
        #     filepath=filepath,
        #     height=height,
        #     width=width,
        #     scale_factor=scale_factor,
        # )
        #
        # We intentionally keep this branch commented out because MarineSTD-GS
        # does not use absolute metric depth. Instead, it only uses relative
        # depth priors normalized to [0, 1].

        # MarineSTD-GS setting:
        # Auto-detect depth map bit depth and normalize to [0, 1].
        # - uint8  (0~255)   → divide by 255
        # - uint16 (0~65535) → divide by 65535
        _raw_depth = cv2.imread(str(filepath.absolute()), cv2.IMREAD_ANYDEPTH)
        if _raw_depth.dtype == np.uint8:
            _max_val = 255.0
        elif _raw_depth.dtype == np.uint16:
            _max_val = 65535.0
        else:
            raise ValueError(f"Unsupported depth image dtype: {_raw_depth.dtype}, expected uint8 or uint16.")

        depth_image = get_depth_image_from_path(
            filepath=filepath,
            height=height,
            width=width,
            scale_factor=1.0 / _max_val,
        )


        out["depth_image"] = depth_image

        hard_image_id = self.split_image_ids[image_idx]
        out["hard_image_id"] = hard_image_id

        return out