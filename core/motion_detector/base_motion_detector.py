from abc import ABC, abstractmethod
from typing import List

import numpy.typing as npt

from common.data.source_model import SourceModel
from common.utilities import logger
from core.data_changed.prev_image_cache import PrevImageCache
from core.filters.detections import DetectionBox


class HasMotionResult:
    def __init__(self):
        self.detection_boxes: List[DetectionBox] = []
        self.has_motion: bool = False

    @staticmethod
    def create(has_motion: bool):
        ret = HasMotionResult()
        ret.has_motion = has_motion
        return ret


_no_motion_result = HasMotionResult()


class BaseMotionDetector(ABC):
    def __init__(self, source_model: SourceModel, prev_img_cache: PrevImageCache):
        self.source_model = source_model
        self.prev_img_cache = prev_img_cache
        self.type_name = type(self).__name__

    @abstractmethod
    def _process_img(self, whole_img: npt.NDArray) -> any:
        raise NotImplementedError('BaseMotionDetector._process_img()')

    @abstractmethod
    def _has_motion(self, source_model: SourceModel, processed_img: any, prev_processed_img: any) -> HasMotionResult:
        raise NotImplementedError('BaseMotionDetector._get_loss()')

    def has_motion(self, whole_img: npt.NDArray) -> HasMotionResult:
        source_id = self.source_model.id
        processed_img = self._process_img(whole_img)
        prev_img = self.prev_img_cache.get(source_id)
        if prev_img is not None:
            result = self._has_motion(self.source_model, processed_img, prev_img)
            if result.has_motion:
                # replace the previous one to the new image
                self.prev_img_cache.set(source_id, processed_img)
                logger.info(f'{self.type_name} (camera {source_id}) switched prev image')
                return result
            else:
                logger.info(f'{self.type_name} (camera {source_id}) did not detect any motion')
                return _no_motion_result
        else:
            logger.info(f'{self.type_name} (camera {source_id}) detected first time')
            self.prev_img_cache.set(source_id, processed_img)
            return _no_motion_result
