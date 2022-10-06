import os
import time
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


pool_test()
