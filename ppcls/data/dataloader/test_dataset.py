#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import print_function

import random

import numpy as np
import os

import cv2

from .common_dataset import CommonDataset
from ..preprocess import transform
from ...utils import logger


class TestDataset(CommonDataset):
    def __init__(
            self,
            image_root,
            cls_label_path,
            transform_ops=None):
        super().__init__(image_root, cls_label_path, transform_ops)

    def _load_anno(self, seed=None):
        assert os.path.exists(self._cls_path)
        assert os.path.exists(self._img_root)

        self.images = []

        with open(self._cls_path) as fd:
            lines = fd.readlines()
            if seed is not None:
                np.random.RandomState(seed).shuffle(lines)
            for l in lines:
                l = l.strip()
                self.images.append(os.path.join(self._img_root, l))
                assert os.path.exists(self.images[-1])

    def __getitem__(self, idx):
        try:
            with open(self.images[idx], 'rb') as f:
                img = f.read()
            if self._transform_ops:
                data = transform({"img": img, "label": None}, self._transform_ops)
            img = data["img"].transpose((2, 0, 1))
            return (img, data["label"])

        except Exception as ex:
            logger.error("Exception occured when parse line: {} with msg: {}".
                         format(self.images[idx], ex))
            rnd_idx = np.random.randint(self.__len__())
            return self.__getitem__(rnd_idx)

    # def __getitem__(self, idx):
    #     with open(self.images[idx], 'rb') as f:
    #         img = f.read()
    #     if self._transform_ops:
    #         data = transform({"img": img, "label": None}, self._transform_ops)
    #     img = data["img"].transpose((2, 0, 1))
    #     return (img, data["label"])
