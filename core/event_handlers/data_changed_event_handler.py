import json
from enum import IntEnum
from redis.client import Redis

from common.event_bus.event_handler import EventHandler
from common.utilities import logger
from core.data_changed.od.od_cache import OdCache
from core.data_changed.prev_image_cache import PrevImageCache
from core.data_changed.source_cache import SourceCache
from core.event_handlers.channel_names import EventChannels


class ModelChanged:
    def __init__(self):
        self.source_id: str = ''


class ModelChangedOp(IntEnum):
    SAVE = 0
    DELETE = 1


class DataChangedEvent:
    def __init__(self):
        self.model_name: str = ''
        self.params_json: str = ''
        self.op: ModelChangedOp = ModelChangedOp.SAVE


class DataChangedEventHandler(EventHandler):
    def __init__(self, connection: Redis, prev_image_cache: PrevImageCache):
        self.channel = EventChannels.data_changed
        self.encoding = 'utf-8'
        self.prev_image_cache = prev_image_cache
        self.source_cache = SourceCache(connection)
        self.od_cache = OdCache(connection, self.source_cache)

    def handle(self, dic: dict):
        if dic is None or dic['type'] != 'message':
            return

        data: bytes = dic['data']
        dic = json.loads(data.decode(self.encoding))

        event = DataChangedEvent()
        event.__dict__.update(dic)

        mc = ModelChanged()
        dic = json.loads(event.params_json)
        mc.__dict__.update(dic)

        if event.model_name == 'source':
            if event.op == ModelChangedOp.SAVE:
                self.source_cache.refresh(mc.source_id)
                logger.warning('Source Cache has been refreshed')
            elif event.op == ModelChangedOp.DELETE:
                self.source_cache.remove(mc.source_id)
                logger.warning('Source Cache has been removed')
            else:
                raise NotImplementedError(event.op)

        elif event.model_name == 'od':
            if event.op == ModelChangedOp.SAVE:
                self.od_cache.refresh(mc.source_id)
                logger.warning('Od Cache has been refreshed')
            elif event.op == ModelChangedOp.DELETE:
                self.od_cache.remove(mc.source_id)
                logger.warning('Od Cache has been removed')
            else:
                raise NotImplementedError(event.op)

        self.prev_image_cache.set(mc.source_id, None)
