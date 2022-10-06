from abc import ABC, abstractmethod
from typing import List

from common.data.source_model import MotionDetectionType, SourceModel
from common.utilities import logger
from core.data_changed.od.od_cache import OdCache
from core.data_changed.prev_image_cache import PrevImageCache
from core.filters.messages import InMessage, OutMessage
from core.filters.detections import DetectionBox, DetectionResult
from core.motion_detector.base_motion_detector import BaseMotionDetector
from core.motion_detector.imagehash_detector import ImageHashDetector
from core.motion_detector.opencv_detector import OpenCVDetector
from core.motion_detector.psnr_detector import PsnrDetector


class Filter(ABC):
    def __init__(self, od_cache: OdCache):
        self.od_cache = od_cache
        self.source_cache = od_cache.source_cache

    @abstractmethod
    def ok(self, message: InMessage) -> bool:
        raise NotImplementedError('Filter.ok')


class TimeFilter(Filter):
    def __init__(self, od_cache: OdCache):
        super().__init__(od_cache)

    def ok(self, message: InMessage) -> bool:
        od = self.od_cache.get(message.source_id)
        if od is None:
            return True
        if not od.is_in_time():
            logger.warning(f'source({message.source_id}) was not in time between {od.start_time} and {od.end_time}')
            return False
        return True


class MotionDetectionFilter(Filter):
    def __init__(self, od_cache: OdCache, source_model: SourceModel, prev_img_cache: PrevImageCache):
        super().__init__(od_cache)
        self.__source_model = source_model
        self.__prev_img_cache = prev_img_cache
        self.__detection_boxes: List[DetectionBox] = []

    def _create_motion_detector(self, source_model: SourceModel) -> BaseMotionDetector | None:
        if source_model.md_type == MotionDetectionType.OpenCV:
            return OpenCVDetector(source_model, self.__prev_img_cache)
        elif source_model.md_type == MotionDetectionType.ImageHash:
            return ImageHashDetector(source_model, self.__prev_img_cache)
        elif source_model.md_type == MotionDetectionType.Psnr:
            return PsnrDetector(source_model, self.__prev_img_cache)
        else:
            logger.warning(f'Motion Detection Type was not found for source({source_model.id})')
            return None

    def ok(self, message: InMessage) -> bool:
        if self.__source_model.md_type == MotionDetectionType.NoMotionDetection:
            return True

        md: BaseMotionDetector | None = self._create_motion_detector(self.__source_model)
        if md is None:
            return False

        ret = md.has_motion(message.np_img)
        self.__detection_boxes = ret.detection_boxes
        return ret.has_motion

    def get_detection_boxes(self) -> List[DetectionBox]:
        return self.__detection_boxes


class ZoneFilter(Filter):
    def __init__(self, od_cache: OdCache, boxes: List[DetectionBox]):
        super().__init__(od_cache)
        self.boxes: List[DetectionBox] = boxes

    def ok(self, message: InMessage) -> bool:
        od = self.od_cache.get(message.source_id)
        if od is None:
            return True
        for box in self.boxes:
            if not od.is_in_zones(box):
                logger.warning(f'a object which was detected by source({message.source_id}) was in the specified zone')
                return False
        return True


class MaskFilter(Filter):
    def __init__(self, od_cache: OdCache, boxes: List[DetectionBox]):
        super().__init__(od_cache)
        self.boxes: List[DetectionBox] = boxes

    def ok(self, message: InMessage) -> bool:
        od = self.od_cache.get(message.source_id)
        if od is None:
            return True
        for box in self.boxes:
            if od.is_in_masks(box):
                logger.warning(f'a object which was detected by source({message.source_id}) was in the specified mask')
                return False
        return True


class OdFilter(Filter):
    def __init__(self, od_cache: OdCache):
        super().__init__(od_cache)

    def ok(self, out_message: OutMessage) -> bool:
        source_id = out_message.source_id
        od = self.od_cache.get(source_id)
        if od is None:
            logger.warning(f'no Od record has been found for source({source_id})')
            return False

        if len(out_message.detections) == 0:
            return False

        filtered_list: List[DetectionResult] = []
        for d in out_message.detections:
            if not od.is_selected(d.pred_cls_idx):
                logger.warning(f'class index was not selected for source({source_id}) {d.pred_cls_idx} {d.pred_cls_name} ({d.pred_score})')
                continue
            if not od.check_threshold(d.pred_cls_idx, d.pred_score):
                logger.warning(f'threshold is lower then expected for source({source_id}) {d.pred_cls_name} ({d.pred_score})')
                continue
            filtered_list.append(d)

        out_message.detections = filtered_list

        return len(filtered_list) > 0
