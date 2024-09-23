import torch
import multiprocessing
# from utils import device_util
import os
import tqdm
from concurrent.futures import ThreadPoolExecutor
from utils import io_cy as io
def _get_pool(num_workers=None):
    if num_workers is None: num_workers = multiprocessing.cpu_count()  # Use the number of available CPU cores
    assert isinstance(num_workers, int) and num_workers > 0, 'invalid num_workers: {}'.format(num_workers)

    print('Multiprocess Pool is initialized with {} workers'.format(num_workers))
    return multiprocessing.Pool(processes=num_workers)


def launch_multi_processes(worker, jobs, desc, num_processes=None, return_required=False):
    if len(jobs) == 0:
        print('[mp_util] no pending jobs.')
        return
    if num_processes is not None: num_processes = min(len(jobs), num_processes)

    
    with _get_pool(num_processes) as pool: 
        with tqdm.tqdm(total=len(jobs), desc=desc) as pbar:
            results = {} if return_required else None
            for result in pool.imap_unordered(worker, jobs):
                pbar.update()  
                results.update(result) if return_required else 0
    return results

def parallel_copytree(src, dst, num_threads=4): 
    os.makedirs(dst, exist_ok=True)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for filename in os.listdir(src):
            src_path = os.path.join(src, filename)
            dst_path = os.path.join(dst, filename)
            if os.path.isfile(src_path):
                executor.submit(io.copy_file, src_path, dst_path)
            elif os.path.isdir(src_path):
                parallel_copytree(src_path, dst_path, num_threads)