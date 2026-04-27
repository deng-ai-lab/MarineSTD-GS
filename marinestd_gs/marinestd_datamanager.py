# Copyright 2022 The Nerfstudio Team. All rights reserved.
# Modifications Copyright 2025 MarineSTD-GS authors.
#
# This file is derived from Nerfstudio's full image datamanager and adapted for
# MarineSTD-GS.
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
MarineSTD-GS full-image datamanager.
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type, Literal

import torch

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
)
from nerfstudio.data.utils.dataloaders import RandIndicesEvalDataloader

from marinestd_gs.underwater_dataset import UnderwaterDataset
from marinestd_gs.marinestd_dataparser import MarineSTDGsDataParserConfig



    
@dataclass
class MarineSTDGsFullImageDatamanagerConfig(FullImageDatamanagerConfig):
    """MarineSTD-GS full-image datamanager config."""

    _target: Type = field(default_factory=lambda: MarineSTDGsFullImageDatamanager)
    dataparser: MarineSTDGsDataParserConfig = field(default_factory=MarineSTDGsDataParserConfig)

class MarineSTDGsFullImageDatamanager(FullImageDatamanager[UnderwaterDataset]):
    """Full-image datamanager for MarineSTD-GS.

    This datamanager extends Nerfstudio's FullImageDatamanager and adds:
    1. Support for moving custom depth_image tensors during image caching.
    2. MarineSTD-GS-specific camera metadata:
       - cam_idx
       - hard_image_id
       - input_img
       - depth_img
    3. An eval_dataloader used by rendering/evaluation paths.
    """

    config: MarineSTDGsFullImageDatamanagerConfig
    train_dataset: UnderwaterDataset
    eval_dataset: UnderwaterDataset

    @staticmethod
    def _to_float01(image: torch.Tensor) -> torch.Tensor:
        """Convert image to float in [0, 1] when needed."""
        if image.dtype == torch.uint8:
            return image.float() / 255.0
        return image.float()

    def _attach_camera_metadata(self, camera: Cameras, data: Dict, image_idx: int) -> Cameras:
        """Attach MarineSTD-GS-specific metadata to a camera."""
        if camera.metadata is None:
            camera.metadata = {}

        camera.metadata["cam_idx"] = image_idx
        camera.metadata["hard_image_id"] = data["hard_image_id"]
        camera.metadata["input_img"] = self._to_float01(data["image"])

        # MarineSTD-GS assumes depth_image is always present and already normalized
        # to [0, 1] in underwater_dataset.py.
        camera.metadata["depth_img"] = data["depth_image"].float()

        return camera

    def _load_images(self, split: Literal["train", "eval"], cache_images_device: Literal["cpu", "gpu"]):
        
        """Load images using the parent implementation, then additionally handle depth_image."""
        undistorted_images = super()._load_images(split=split, cache_images_device=cache_images_device)

        if cache_images_device == "gpu":
            for cache in undistorted_images:
                if "depth_image" in cache:
                    cache["depth_image"] = cache["depth_image"].to(self.device)

                # Keep this behavior consistent with the user's current working version.
                self.train_cameras = self.train_dataset.cameras.to(self.device)

        elif cache_images_device == "cpu":
            for cache in undistorted_images:
                if "depth_image" in cache and hasattr(cache["depth_image"], "pin_memory"):
                    cache["depth_image"] = cache["depth_image"].pin_memory()

                # Keep this behavior consistent with the user's current working version.
                self.train_cameras = self.train_dataset.cameras

        else:
            raise ValueError(f"Unsupported cache_images_device: {cache_images_device}")

        return undistorted_images

    def setup_eval(self):
        """Sets up the data loader for evaluation."""
        super().setup_eval()

        # Keep this additional eval dataloader for rendering/evaluation usage and used by ns-render.
        self.eval_dataloader = RandIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )

    @property
    def fixed_indices_eval_dataloader(self) -> List[Tuple[Cameras, Dict]]:
        """Returns a list of (camera, data) tuples for all eval images."""
        image_indices = [i for i in range(len(self.eval_dataset))]
        data = [d.copy() for d in self.cached_eval]
        cameras_copy = deepcopy(self.eval_dataset.cameras).to(self.device)
        cameras = []

        for i in image_indices:
            data[i]["image"] = data[i]["image"].to(self.device)
            camera = cameras_copy[i : i + 1]
            camera = self._attach_camera_metadata(camera, data[i], i)
            cameras.append(camera)

        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        return list(zip(cameras, data))

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next training batch as (camera, data)."""
        self.train_count += 1

        image_idx = self.train_unseen_cameras.pop(0)
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.train_unseen_cameras) == 0:
            self.train_unseen_cameras = self.sample_train_cameras()

        # We're going to copy to make sure we don't mutate the cached dictionary.
        # This can cause a memory leak: https://github.com/nerfstudio-project/nerfstudio/issues/3335
        data = self.cached_train[image_idx].copy()
        data["image"] = data["image"].to(self.device)

        assert len(self.train_cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.train_cameras[image_idx : image_idx + 1].to(self.device)
        camera = self._attach_camera_metadata(camera, data, image_idx)

        return camera, data

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch."""
        self.eval_count += 1
        return self.next_eval_image(step=step)

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation image batch as (camera, data)."""
        image_idx = self.eval_unseen_cameras.pop(random.randint(0, len(self.eval_unseen_cameras) - 1))
        # Make sure to re-populate the unseen cameras list if we have exhausted it        
        if len(self.eval_unseen_cameras) == 0:
            self.eval_unseen_cameras = [i for i in range(len(self.eval_dataset))]

        data = self.cached_eval[image_idx].copy()
        data["image"] = data["image"].to(self.device)

        assert len(self.eval_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.eval_dataset.cameras[image_idx : image_idx + 1].to(self.device)
        camera = self._attach_camera_metadata(camera, data, image_idx)

        return camera, data
    
