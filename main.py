import os
import time

from common.event_bus.event_bus import EventBus
from core.event_handler_single_thread import ReadServiceEventHandler
from core.event_handlers import ReadServiceMpEventHandler
import multiprocessing as mp


def motion_detect(index: int):
    time.sleep(1.)
    print(f'index: {index} - pid: {os.getpid()}')
    time.sleep(1.)


def pool_test():
    with mp.Pool() as pool:
        for j in range(os.cpu_count()):
            pool.apply_async(motion_detect, args=(j,))
        pool.close()
        pool.join()


def main():
    # pool_test()
    # return
    # with ReadServiceEventHandler() as handler:
    #     event_bus = EventBus('read_service')
    #     event_bus.subscribe_async(handler)
    # print('done')
    with ReadServiceMpEventHandler() as handler:
        event_bus = EventBus('read_service')
        event_bus.subscribe_async(handler)


if __name__ == '__main__':
    main()
