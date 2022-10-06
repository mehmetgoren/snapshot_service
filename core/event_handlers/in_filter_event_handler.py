import os
from multiprocessing import Pool

from core.data_changed.od.od_cache import OdCache
from core.data_changed.prev_image_cache import PrevImageCache
from core.data_changed.source_cache import SourceCache
from common.event_bus.event_bus import EventBus
from common.event_bus.event_handler import EventHandler
from common.utilities import config, crate_redis_connection, RedisDb
from core.event_handlers.channel_names import EventChannels
from core.filters.in_filters import InFilters

_publisher = EventBus(EventChannels.snapshot_in)
_main_connection = crate_redis_connection(RedisDb.MAIN)
_source_cache = SourceCache(_main_connection)
_od_cache = OdCache(_main_connection, _source_cache)
_in_filters = InFilters(_od_cache)


# noinspection DuplicatedCode
class InFilterEventHandler(EventHandler):
    def __init__(self, prev_image_cache: PrevImageCache, source_cache_dic: dict, od_cache_dic: dict):
        self.pool: Pool = None  # Pool(4)  # None
        _in_filters.set_prev_image_cache(prev_image_cache)
        _source_cache.set_dict(source_cache_dic)
        _od_cache.set_dict(od_cache_dic)

    def __enter__(self):
        self.pool = Pool(config.snapshot.process_count if config.snapshot.process_count > 0 else os.cpu_count())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()
        return self

    def handle(self, dic: dict):
        if dic is None or dic['type'] != 'message':
            return

        self.pool.apply_async(_handle, args=(dic,))


def _handle(dic: dict):
    in_message = _in_filters.ok(dic)
    if in_message is not None:
        _publisher.publish(in_message.create_publish_dic())
