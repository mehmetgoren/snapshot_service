import base64
import json
from typing import List
import numpy as np
import numpy.typing as npt
from PIL import Image, UnidentifiedImageError
import io

from common.utilities import logger, datetime_now
from core.filters.detections import DetectionResult
from core.utilities import generate_id


class InMessage:
    def __init__(self):
        self.name: str = ''
        self.source_id: str = ''
        self.base64_image: str = ''
        self.np_img: npt.NDArray | None = None
        self.pil_image: Image = None
        self.ai_clip_enabled: bool = False
        self.encoding = 'utf-8'

    def form_dic(self, dic: dict) -> dict:
        data: bytes = dic['data']
        dic = json.loads(data.decode(self.encoding))
        self.name = dic['name']
        self.source_id = dic['source']
        self.base64_image = dic['img']
        self.ai_clip_enabled = dic['ai_clip_enabled']

        base64_decoded = base64.b64decode(self.base64_image)
        try:
            self.pil_image = Image.open(io.BytesIO(base64_decoded))
            self.np_img = np.asarray(self.pil_image)
        except UnidentifiedImageError as err:
            logger.error(f'an error occurred while creating a PIL image from base64 string, err: {err}')

        return dic

    def create_publish_dic(self) -> str:
        dic = {'name': self.name, 'source_id': self.source_id, 'base64_image': self.base64_image, 'ai_clip_enabled': self.ai_clip_enabled}
        js = json.dumps(dic)
        return js


class OutMessage(InMessage):
    def __init__(self):
        super().__init__()
        self.channel: str = ''
        self.list_name: str = ''
        self.detections: List[DetectionResult] = []

    def form_dic(self, dic: dict):
        dic = super(OutMessage, self).form_dic(dic)
        self.channel = dic['channel']
        self.list_name = dic['list_name']
        ds = dic['detections']
        for d in ds:
            r = DetectionResult()
            r.pred_cls_name = d['pred_cls_name']
            r.pred_cls_idx = d['pred_cls_idx']
            r.pred_score = d['pred_score']
            b = d['box']
            box = r.box
            box.x1 = b['x1']
            box.y1 = b['y1']
            box.x2 = b['x2']
            box.y2 = b['y2']
            self.detections.append(r)

    def create_publish_dic(self) -> str:
        ds = []
        for d in self.detections:
            ds.append({'pred_cls_name': d.pred_cls_name, 'pred_cls_idx': d.pred_cls_idx, 'pred_score': d.pred_score})
        dic = {'id': generate_id(), 'source_id': self.source_id, 'created_at': datetime_now(),
               self.list_name: ds, 'base64_image': self.base64_image, 'ai_clip_enabled': self.ai_clip_enabled}
        # self.detections was already came form self.dic
        js = json.dumps(dic)
        return js
