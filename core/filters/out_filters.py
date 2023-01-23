from __future__ import annotations

from PIL import ImageDraw
from typing import List
import io
import base64

from common.utilities import logger, config
from core.data_changed.od.od_cache import OdCache
from core.event_handlers.channel_names import EventChannels
from core.filters.filters import Filter, ZoneFilter, MaskFilter, OdFilter
from core.filters.messages import OutMessage


class OutFilters(Filter):
    def __init__(self, od_cache: OdCache):
        super().__init__(od_cache)
        self.colors = self.__create_colors()
        self.colors_length = len(self.colors)
        self.overlay = config.snapshot.overlay

    @staticmethod
    def __create_colors() -> List[str]:
        return ['#FFFF00', '#FAEBD7', '#F0F8FF', '#7FFFD4', '#F0FFFF', '#F5F5DC', '#FFE4C4', '#000000', '#FFEBCD', '#0000FF', '#8A2BE2', '#A52A2A', '#DEB887',
                '#5F9EA0', '#7FFF00', '#D2691E', '#FF7F50', '#6495ED', '#FFF8DC', '#DC143C', '#00FFFF', '#00008B', '#008B8B', '#B8860B', '#A9A9A9', '#006400',
                '#A9A9A9', '#BDB76B', '#8B008B', '#556B2F', '#FF8C00', '#9932CC', '#8B0000', '#E9967A', '#8FBC8F', '#483D8B', '#2F4F4F', '#2F4F4F', '#00CED1',
                '#9400D3', '#FF1493', '#00BFFF', '#696969', '#696969', '#1E90FF', '#B22222', '#FFFAF0', '#228B22', '#FF00FF', '#DCDCDC', '#F8F8FF', '#FFD700',
                '#DAA520', '#808080', '#008000', '#ADFF2F', '#808080', '#F0FFF0', '#FF69B4', '#CD5C5C', '#4B0082', '#FFFFF0', '#F0E68C', '#E6E6FA', '#FFF0F5',
                '#7CFC00', '#FFFACD', '#ADD8E6', '#F08080', '#E0FFFF', '#FAFAD2', '#D3D3D3', '#90EE90', '#D3D3D3', '#FFB6C1', '#FFA07A', '#20B2AA', '#87CEFA',
                '#778899', '#778899', '#B0C4DE', '#FFFFE0', '#00FF00', '#32CD32', '#FAF0E6', '#FF00FF', '#800000', '#66CDAA', '#0000CD', '#BA55D3', '#9370DB',
                '#3CB371', '#7B68EE', '#00FA9A', '#48D1CC', '#C71585', '#191970', '#F5FFFA', '#FFE4E1', '#FFE4B5', '#FFDEAD', '#000080', '#FDF5E6', '#808000',
                '#6B8E23', '#FFA500', '#FF4500', '#DA70D6', '#EEE8AA', '#98FB98', '#AFEEEE', '#DB7093', '#FFEFD5', '#FFDAB9', '#CD853F', '#FFC0CB', '#DDA0DD',
                '#B0E0E6', '#800080', '#663399', '#FF0000', '#BC8F8F', '#4169E1', '#8B4513', '#FA8072', '#F4A460', '#2E8B57', '#FFF5EE', '#A0522D', '#C0C0C0',
                '#87CEEB', '#6A5ACD', '#708090', '#708090', '#FFFAFA', '#00FF7F', '#4682B4', '#D2B48C', '#008080', '#D8BFD8', '#FF6347', '#40E0D0', '#EE82EE',
                '#F5DEB3', '#FFFFFF', '#F5F5F5', '#00FFFF', '#9ACD32']

    def __draw(self, message: OutMessage):
        np_image = message.np_img
        if np_image is None:
            logger.error(f'could not convert base64 image to numpy array, source id:{message.source_id}, channel: {message.channel}')
            return
        # pil_image = Image.fromarray(np_image)
        for idx, d in enumerate(message.detections):
            color = self.colors[idx % self.colors_length]
            xy1 = (d.box.x1, d.box.y1)
            xy2 = (d.box.x2, d.box.y2)
            text = d.format()
            draw = ImageDraw.Draw(message.pil_image)
            draw.rectangle((xy1, xy2), outline=color, width=1)
            draw.text(xy1, text)
            # cv2.rectangle(np_image, xy1, xy2, color)
            # cv2.putText(np_image, text, xy1, cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=1)
        # set base64 image again
        buffered = io.BytesIO()
        message.pil_image.save(buffered, format="JPEG")
        img_to_bytes = buffered.getvalue()
        message.base64_image = base64.b64encode(img_to_bytes).decode()

    # noinspection DuplicatedCode
    def ok(self, dic: dict) -> OutMessage | None:
        message = OutMessage()
        message.form_dic(dic)
        if message.np_img is None:
            logger.error(f'a snapshot image is not valid for source({message.source_id})')
            return None

        if message.channel == EventChannels.od_service:
            source_model = self.source_cache.get(message.source_id)
            if source_model is None:
                logger.error(f'source({message.source_id}) was not found in filters operation')
                return None

            od_filter = OdFilter(self.od_cache)
            if not od_filter.ok(message):
                logger.warning(f'od filter is not ok for source({message.source_id})')
                return None

            boxes = [d.box for d in message.detections]
            if source_model.md_type > 1 and len(boxes) > 0:
                zone_filter = ZoneFilter(self.od_cache, boxes)
                if not zone_filter.ok(message):
                    logger.warning(f'zone filter is not ok for source({message.source_id})')
                    return None
                mask_filter = MaskFilter(self.od_cache, boxes)
                if not mask_filter.ok(message):
                    logger.warning(f'mask filter is not ok for source({message.source_id})')
                    return None

        if self.overlay:
            self.__draw(message)

        return message
