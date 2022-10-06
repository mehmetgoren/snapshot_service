from typing import List
import numpy as np
from numpy import typing as npt
import cv2

from common.data.source_model import SourceModel
from core.data_changed.prev_image_cache import PrevImageCache
from core.filters.detections import DetectionBox
from core.motion_detector.base_motion_detector import BaseMotionDetector, HasMotionResult


class OpenCVDetector(BaseMotionDetector):
    def __init__(self, source_model: SourceModel, prev_img_cache: PrevImageCache):
        super(OpenCVDetector, self).__init__(source_model, prev_img_cache)
        self.ksize = (5, 5)
        self.kernel = np.ones((5, 5))

    def _process_img(self, whole_img: npt.NDArray) -> npt.NDArray:
        # 1. Load image; convert to RGB
        # img_rgb = cv2.cvtColor(src=img_rgb, code=cv2.COLOR_BGR2RGB) ***
        # 2. Prepare image; grayscale and blur
        prepared_frame = cv2.cvtColor(whole_img, cv2.COLOR_BGR2GRAY)
        prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=self.ksize, sigmaX=0)
        return prepared_frame

    def _has_motion(self, source_model: SourceModel, img_rgb: npt.NDArray, prev_img_rgb: npt.NDArray) -> HasMotionResult:
        threshold = source_model.md_opencv_threshold
        contour_area_limit = source_model.md_contour_area_limit

        # calculate difference and update previous frame
        diff_frame = cv2.absdiff(src1=prev_img_rgb, src2=img_rgb)

        # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
        diff_frame = cv2.dilate(diff_frame, self.kernel, 1)

        # 5. Only take different areas that are different enough (>20 / 255)
        thresh_frame = cv2.threshold(src=diff_frame, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY)[1]

        contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(image=img_rgb, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA) ***

        contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[DetectionBox] = []
        for contour in contours:
            if cv2.contourArea(contour) < contour_area_limit:
                # cv2.imshow('diff', diff_frame) ***
                # cv2.imshow('thresh_frame', thresh_frame) ***
                # cv2.imshow('Motion detector', img_rgb) ***
                # cv2.waitKey() ***
                # too small: skip!
                continue

            (x, y, w, h) = cv2.boundingRect(contour)
            box = DetectionBox()
            box.x1, box.y1, box.x2, box.y2 = x, y, x + w, y + h
            boxes.append(box)
            # print(f'x: {x}, y: {y}, w: {w}, h: {h}') ***
            # cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 0, 255), 3) ***

            # cv2.imshow('diff', diff_frame) ***
            # cv2.imshow('thresh_frame', thresh_frame) ***
            # cv2.imshow('Motion detector', img_rgb) ***
            # cv2.waitKey()

        ret = HasMotionResult.create(len(boxes) > 0)
        ret.detection_boxes = boxes
        return ret
