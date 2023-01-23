from __future__ import annotations

from common.utilities import logger
from core.data_changed.od.od_cache import OdCache
from core.data_changed.prev_image_cache import PrevImageCache
from core.filters.filters import Filter, TimeFilter, MotionDetectionFilter, ZoneFilter, MaskFilter
from core.filters.messages import InMessage


class InFilters(Filter):
    def __init__(self, od_cache: OdCache):
        super().__init__(od_cache)
        self.prev_image_cache: PrevImageCache | None = None

    def set_prev_image_cache(self, prev_image_cache: PrevImageCache):
        self.prev_image_cache = prev_image_cache

    # noinspection DuplicatedCode
    def ok(self, dic: dict) -> InMessage | None:
        message = InMessage()
        message.form_dic(dic)
        if message.np_img is None:
            logger.error(f'a snapshot image is not valid for source({message.source_id})')
            return None

        source_model = self.source_cache.get(message.source_id)
        if source_model is None:
            logger.error(f'source({message.source_id}) was not found in filters operation')
            return None

        time_filter: TimeFilter = TimeFilter(self.od_cache)
        if not time_filter.ok(message):
            logger.warning(f'time filter is not ok for source({message.source_id})')
            return None

        motion_detection_filter = MotionDetectionFilter(self.od_cache, source_model, self.prev_image_cache)
        if not motion_detection_filter.ok(message):
            logger.warning(f'motion detection filter is not ok for source({message.source_id})')
            return None

        detection_boxes = motion_detection_filter.get_detection_boxes()
        if len(detection_boxes) == 0:  # it may be NoMotionDetection, ImageHash or PSNR
            return message

        zone_filter = ZoneFilter(self.od_cache, detection_boxes)
        if not zone_filter.ok(message):
            logger.warning(f'zone filter is not ok for source({message.source_id})')
            return None

        mask_filter = MaskFilter(self.od_cache, detection_boxes)
        if not mask_filter.ok(message):
            logger.warning(f'mask filter is not ok for source({message.source_id})')
            return None

        return message
