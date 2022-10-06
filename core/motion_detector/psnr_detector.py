import cv2
import numpy.typing as npt

from common.data.source_model import SourceModel
from core.data_changed.prev_image_cache import PrevImageCache

from core.motion_detector.base_motion_detector import BaseMotionDetector, HasMotionResult


class PsnrDetector(BaseMotionDetector):
    def __init__(self, source_model: SourceModel, prev_img_cache: PrevImageCache):
        super(PsnrDetector, self).__init__(source_model, prev_img_cache)

    def _process_img(self, whole_img: npt.NDArray):
        return whole_img

    def _has_motion(self, source_model: SourceModel, processed_img: npt.NDArray, prev_processed_img: npt.NDArray) -> HasMotionResult:
        psnr = cv2.PSNR(processed_img, prev_processed_img)
        loss = 20.0 - psnr
        return HasMotionResult.create(loss > source_model.md_psnr_threshold)
