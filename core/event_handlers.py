import base64
import io
import json
import os
from datetime import datetime
import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from multiprocessing import Pool, Manager

from common.event_bus.event_bus import EventBus
from common.event_bus.event_handler import EventHandler

publisher = EventBus('od_service')
manager = Manager()
prev_img_dic = manager.dict()
prev_img_dic['kejraamt2md'] = None


class MotionDetectorConfig:
    def __init__(self):
        self.encoding = 'utf-8'
        self.overlay = True  # config.ai.overlay

        self.frame_count = 0
        self.ksize = (5, 5)
        self.kernel = np.ones((5, 5))
        self.threshold = 120
        self.contour_area_limit = 10000


md_config = MotionDetectorConfig()


class ReadServiceMpEventHandler(EventHandler):
    def __init__(self):
        self.pool: Pool = None  # Pool(4)  # None

    def __enter__(self):
        self.pool = Pool()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.pool.close()
        self.pool.join()
        return self

    def handle(self, dic: dict):
        if dic is None or dic['type'] != 'message':
            return

        self.pool.apply_async(_handle, args=(dic,))

        # th = Thread(target=_handle, args=[dic])
        # th.daemon = True
        # th.start()


# you can handle it and add dict to multiprocessing.Queue then execute on a parameterless function by Pool.run_async to provide a real multi-core support
def _handle(dic: dict):
    data: bytes = dic['data']
    dic = json.loads(data.decode(md_config.encoding))
    name = dic['name']
    source_id = dic['source']
    img_str = dic['img']
    ai_clip_enabled = dic['ai_clip_enabled']

    base64_decoded = base64.b64decode(img_str)
    try:
        image = Image.open(io.BytesIO(base64_decoded))
    except UnidentifiedImageError as err:
        # logger.error(f'an error occurred while creating a PIL image from base64 string, err: {err}')
        return

    motion_detection_32(source_id, image)


def motion_detection_32(source_id: str, image) -> bool:
    md_config.frame_count += 1
    # 1. Load image; convert to RGB
    img_rgb = np.array(image)
    # img_rgb = cv2.cvtColor(src=img_rgb, code=cv2.COLOR_BGR2RGB)

    # 2. Prepare image; grayscale and blur
    prepared_frame = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    prepared_frame = cv2.GaussianBlur(src=prepared_frame, ksize=md_config.ksize, sigmaX=0)
    # 3. Set previous frame and continue if there is None
    if (prev_img_dic[source_id] is None):
        # First frame; there is no previous one yet/mnt/sdc1/test_projects/capture_service/main.py
        prev_img_dic[source_id] = prepared_frame
        return False

    # calculate difference and update previous frame
    diff_frame = cv2.absdiff(src1=prev_img_dic[source_id], src2=prepared_frame)
    prev_img_dic[source_id] = prepared_frame

    # 4. Dilute the image a bit to make differences more seeable; more suitable for contour detection
    diff_frame = cv2.dilate(diff_frame, md_config.kernel, 1)

    # 5. Only take different areas that are different enough (>20 / 255)
    thresh_frame = cv2.threshold(src=diff_frame, thresh=md_config.threshold, maxval=255, type=cv2.THRESH_BINARY)[1]

    contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(image=img_rgb, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)

    contours, _ = cv2.findContours(image=thresh_frame, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    print(f'pid: {os.getpid()}, contours:{len(contours)}')
    for contour in contours:
        if cv2.contourArea(contour) < md_config.contour_area_limit:
            # too small: skip!
            # cv2.imshow('Motion detector', img_rgb)
            # cv2.waitKey(1)
            continue
        print(f'pid: {os.getpid()}, geçtim amına koyduğum at {datetime.now()}')
        return True
        # (x, y, w, h) = cv2.boundingRect(contour)
        # cv2.rectangle(img=img_rgb, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=2)
        #
        # cv2.imshow('Motion detector', img_rgb)
        # cv2.waitKey(1)
    return False
