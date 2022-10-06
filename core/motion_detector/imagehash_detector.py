import imagehash
from PIL import Image
import numpy.typing as npt

from common.data.source_model import SourceModel
from core.data_changed.prev_image_cache import PrevImageCache
from core.motion_detector.base_motion_detector import BaseMotionDetector, HasMotionResult


class ImageHashDetector(BaseMotionDetector):
    def __init__(self, source_model: SourceModel, prev_img_cache: PrevImageCache):
        super(ImageHashDetector, self).__init__(source_model, prev_img_cache)

    def _process_img(self, whole_img: npt.NDArray) -> Image:
        return Image.fromarray(whole_img)

    def _has_motion(self, source_model: SourceModel, processed_img: Image, prev_processed_img: Image) -> HasMotionResult:
        hash1 = imagehash.average_hash(processed_img)
        hash2 = imagehash.average_hash(prev_processed_img)
        loss = hash1 - hash2
        return HasMotionResult.create(loss > source_model.md_imagehash_threshold)
