import time
import uuid
from threading import Thread
from typing import Callable
from redis.client import Redis

from common.data.heartbeat_repository import HeartbeatRepository
from common.data.service_repository import ServiceRepository
from common.event_bus.event_bus import EventBus
from common.utilities import crate_redis_connection, RedisDb, logger
from core.data_changed.prev_image_cache import PrevImageCache
from core.event_handlers.channel_names import EventChannels
from core.event_handlers.data_changed_event_handler import DataChangedEventHandler


def register_detect_service(service_name: str, instance_name: str, description: str):
    connection_main = crate_redis_connection(RedisDb.MAIN)
    heartbeat = HeartbeatRepository(connection_main, service_name)
    heartbeat.start()
    service_repository = ServiceRepository(connection_main)
    service_repository.add(service_name, instance_name, description)
    return connection_main


def listen_data_changed_event_async(connection: Redis, prev_image_cache: PrevImageCache, source_cache: dict, od_cache: dict):
    def fn():
        while 1:
            event_bus = None
            try:
                handler = DataChangedEventHandler(connection, prev_image_cache)
                handler.source_cache.set_dict(source_cache)
                handler.od_cache.set_dict(od_cache)
                event_bus = EventBus(EventChannels.data_changed)
                event_bus.subscribe_async(handler)
            except BaseException as ex:
                logger.error(f'an error occurred on listen data changed event, ex: {ex}')
            finally:
                if event_bus is not None:
                    try:
                        event_bus.unsubscribe()
                    except BaseException as ex:
                        logger.error(f'an error occurred during the unsubscribing data changed event, err: {ex}')
            time.sleep(1.)
            fn()

    start_thread(fn, False)


def generate_id() -> str:
    return str(uuid.uuid4().hex)


def start_thread(fn: Callable, daemon: bool, args=()):
    th = Thread(target=fn, args=args)
    th.daemon = daemon
    th.start()
