from abc import abstractmethod
from typing import List
import numpy as np
from redis.client import Redis

from common.utilities import logger
from core.data.od_cache import OdCache
from core.object_detectors.object_detector import ObjectDetector
from core.models.object_detector_model import BaseObjectDetectorModel, BaseDetectedObject, DetectionBox


# Also make sure it performant good on jetson nano.
class BaseOnceDetector(ObjectDetector):
    def __init__(self, connection: Redis, detector_model: BaseObjectDetectorModel):
        super(BaseOnceDetector, self).__init__(detector_model)
        self.od_cache = OdCache(connection)
        self.collection = OnceObjectDetectorList()  # works with numpy array image
        self.type_name = type(self).__name__

    @abstractmethod
    def _process_img(self, whole_img: np.array):
        raise NotImplementedError('BaseOnceDetector._process_img()')

    @abstractmethod
    def _get_loss(self, processed_img: np.array, prev_processed_img: np.array):
        raise NotImplementedError('BaseOnceDetector._get_loss()')

    @abstractmethod
    def _get_algorithm_threshold(self):
        raise NotImplementedError('BaseOnceDetector._get_algorithm_threshold()')

    def detected_before(self, whole_img: np.array, source_id: str, cls_idx: int) -> bool:
        processed_img = self._process_img(whole_img)
        prev_processed_img = self.collection.get(source_id, cls_idx)
        class_name = self.get_detected_object_class_name(cls_idx)
        if prev_processed_img is not None:
            loss = self._get_loss(processed_img, prev_processed_img)
            specific_threshold = self._get_algorithm_threshold()
            if loss > specific_threshold:
                # replace the previous one to the new image
                self.collection.set(source_id, cls_idx, processed_img)
                logger.info(
                    f'{self.type_name} (camera {source_id}) - (class {class_name}) switched prev image, the loss is {loss}, the threshold {specific_threshold}')
                return False
            else:
                logger.info(
                    f'{self.type_name} (camera {source_id}) - (class {class_name}) same image detected, the loss is {loss}, the threshold {specific_threshold}')
                return True
        else:
            logger.info(
                f'{self.type_name} (camera {source_id}) - (class {class_name}) detected first time')
            self.collection.set(source_id, cls_idx, processed_img)
            return False

    def get_detect_boxes(self, img: np.array, source_id: str) -> List[DetectionBox]:
        od = self.od_cache.get_od_model(source_id)
        if od is None or od.selected_list_length == 0:
            return []

        boxes: List[DetectionBox] = self.concrete.get_detect_boxes(img, source_id)
        ret: List[DetectionBox] = []
        for box in boxes:
            conf, cls_idx = box.confidence, box.cls_idx
            if not od.is_selected(cls_idx):
                continue
            if not od.check_threshold(cls_idx, conf):
                logger.warning(f'threshold is lower then expected for {self.get_detected_object_class_name(cls_idx)} ({conf})')
                continue

            if not od.is_in_time():
                logger.warning(
                    f'a detected object({self.get_detected_object_class_name(cls_idx)}, conf:{conf}) was not in time between {od.start_time} and {od.end_time}')
                continue

            if not od.is_in_zones(box):
                logger.warning(f'a {self.get_detected_object_class_name(cls_idx)} object was in the specified zone, conf:{conf}')
                continue

            if od.is_in_masks(box):
                logger.warning(f'a {self.get_detected_object_class_name(cls_idx)} object was detected is in the mask area, conf:{conf}')
                continue

            if not self.detected_before(img, source_id, cls_idx):
                ret.append(box)
        return ret

    def create_detected_object(self, img: np.array, source_id: str, box: DetectionBox) -> BaseDetectedObject:
        return self.concrete.create_detected_object(img, source_id, box)


class OnceObjectDetectorList:
    def __init__(self):
        self.cameras = {}

    def __get_inner_dic(self, source_id: str, cls_idx: int):
        camera_items = self.cameras.get(source_id)
        if camera_items is None:
            camera_items = {cls_idx: None}
            self.cameras[source_id] = camera_items
        return camera_items

    def get(self, source_id: str, cls_idx: int) -> object:
        camera_items = self.__get_inner_dic(source_id, cls_idx)
        return camera_items.get(cls_idx)

    def set(self, source_id: str, cls_idx: int, item):
        camera_items = self.__get_inner_dic(source_id, cls_idx)
        camera_items[cls_idx] = item

    def remove(self, source_id: str, cls_idx: int, ):
        camera_items = self.__get_inner_dic(source_id, cls_idx)
        del camera_items[cls_idx]
