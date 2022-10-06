import time
from multiprocessing import Manager

from common.utilities import logger
from core.data_changed.prev_image_cache import PrevImageCache
from core.event_handlers.channel_names import EventChannels
from core.event_handlers.in_filter_event_handler import InFilterEventHandler
from common.event_bus.event_bus import EventBus
from core.event_handlers.out_filter_event_handler import OutFilterEventHandler
from core.utilities import register_detect_service, listen_data_changed_event_async, start_thread


def main():
    conn = register_detect_service('snapshot_service', 'snapshot_service-instance', 'The Snapshot Service®')
    with Manager() as manager:
        prev_image_cache = PrevImageCache(manager.dict())
        source_cache_dic = manager.dict()
        od_cache_dic = manager.dict()
        listen_data_changed_event_async(conn, prev_image_cache, source_cache_dic, od_cache_dic)

        def fn_in():
            try:
                with InFilterEventHandler(prev_image_cache, source_cache_dic, od_cache_dic) as handler:
                    event_bus = EventBus(EventChannels.read_service)
                    event_bus.subscribe_async(handler)
            except BaseException as ex:
                logger.error(f'an error occurred while listening InFilterEventHandler, ex: {ex}')
                time.sleep(1.)
                fn_in()

        start_thread(fn_in, True)

        def fn_out():
            try:
                with OutFilterEventHandler(source_cache_dic, od_cache_dic) as handler:
                    event_bus = EventBus(EventChannels.snapshot_out)
                    event_bus.subscribe_async(handler)
            except BaseException as ex:
                logger.error(f'an error occurred while listening OutFilterEventHandler, ex: {ex}')
                time.sleep(1.)
                fn_out()

        fn_out()


if __name__ == '__main__':
    main()
