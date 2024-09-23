import os
import urllib.request
import tensorflow.compat.v1 as tf
import numpy as np
import tqdm
from utils.utils_io import read_image, save_json, load_json
from .BaseDataParser import BaseDataParser
DEBUG_RUN_SUBSET = False
from starter.env_getter import get_env

PREP_DIR = get_env('PREP')
PREP_DIR = PREP_DIR.replace('sdb', 'sdc')
# If need to move images to a new disk, might have to modify the symlink with os.path.islink + os.symlink(src_file_name, tmpLink)

is_online_url = lambda x: x.startswith('http')

def download_worker(job_arg):
    (img_url, img_save_path) = job_arg

    if is_online_url(img_url):
        try:
            opener = urllib.request.build_opener()
            opener.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3")]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(img_url, img_save_path)
            #print(f'[image_downloader_worker] fetch: image saved to: {img_save_path}')
        except Exception as e:
            print(f'[image_downloader_worker] error: {e} - {img_url}')
    else:
        os.system('ln -sf {} {}'.format(img_url, img_save_path))
        print(f'[image_downloader_worker] dedup: linked saved to: {img_save_path}')

def convert_worker(job_arg):
    (img_load_path, txt_save_path) = job_arg

    img = read_image(img_load_path)
    if img.ndim == 2: img = np.stack([img, img, img], axis=-1)
    img_string = tf.io.encode_png(img)

    tf.io.write_file(txt_save_path, img_string)
    print(f'[image_converter_worker] converted: {img_load_path} -> {txt_save_path}')


# Use this to parse the image path in the tfrecord and download them locally
class ImageDataParser(BaseDataParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
                
        self._sel_img_urls_suffix = 'img_urls_for_cap.json' # subset selected for image captioning task
        self._all_img_urls_suffix = 'img_urls_for_all.json'
    
    def get_img_string_id_from_url(self, split:str, page_idx:int, img_url:str):
        # for each page, we create a string-id for each image (according to their urls)
        img_fname = img_url.split('/')[-1]
        img_string_id = f'{split}-{page_idx}-{img_fname}'
        return img_string_id

    def get_img_local_path_from_string_id(self, save_root, img_string_id):
        tokens = img_string_id.split('-')
        assert len(tokens) >=3 , f'invalid img_string_id: {img_string_id}'

        split, sample_idx, img_fname = tokens[0], tokens[1], '-'.join(tokens[2:])
        assert img_string_id == '-'.join([split, sample_idx, img_fname])

        # Linux system have a maxium file name length of 256, make the name 210 considering the folder path with around length 40
        if len(img_fname) > 210: img_fname = img_fname[:210] 

        sample_save_dir = os.path.join(save_root, str(sample_idx))
        img_local_path = os.path.join(sample_save_dir, img_fname)
        return img_local_path
    
    def _build_image_download_tasks(self, split: str, prep_dir : str = PREP_DIR):
        if split not in self.split_datasets: self._load_tfrecord_dataset(split)


        save_dir = os.path.join(prep_dir, 'download_tasks')
        os.makedirs(save_dir, exist_ok=True)

        sel_json_save_path = os.path.join(save_dir, f'{split}_{self._sel_img_urls_suffix}') # [official] selected img_urls for image captioning task
        all_json_save_path = os.path.join(save_dir, f'{split}_{self._all_img_urls_suffix}') # [byproduct] all img_urls
        print(f'building {split}-download_tasks to: {sel_json_save_path}')
        if os.path.exists(sel_json_save_path):
            print(f'{sel_json_save_path} already exists. Type "y" to overwrite.')
            if input() != 'y':
                print('exiting...')
                return
            print('overwriting...')
            
        all_tasks, sel_tasks = {}, {}
        for sample_idx, sample_page in tqdm.tqdm(enumerate(self.split_datasets[split]), desc=f'building {split}-download_tasks'):
            ctx, seq = sample_page

            if not ctx['page_contains_images']: continue

            _img_urls = []
            for i in range(len(seq['section_image_url'].values)):
                img_url = seq['section_image_url'].values[i].numpy().decode()
                img_used_for_cap = seq['is_image_caption_sample'].values[i].numpy()

                if img_url not in _img_urls:
                    _img_urls.append(img_url)
                else:
                    # at some rare cases, the same image url appears twice in the same page-TensorData['section_image_url'] (probably typo in data collection)
                    # However, the image might only exist once in the page.
                    # For example, image_url (https://en.wikipedia.org/wiki/File:AlfredPalmerM3tank1942b.jpg) appears twice in page (https://en.wikipedia.org/wiki/Jacob_L._Devers)
                    # Note: not in actual page; but in the tensor data of the page
                    continue

                img_string_id:str = self.get_img_string_id_from_url(split, sample_idx, img_url) # e.g., 'test-175-Puckett%27s_Farm_Equipment_in_Derita.jpg"

                ### byproduct - all images
                all_tasks[img_string_id] = img_url

                ### official instruction - select subset of images for image captioning
                if img_used_for_cap: sel_tasks[img_string_id] = img_url
        
        print(f'{split}-images2download: {len(sel_tasks)}, saved to {sel_json_save_path}')
        save_json(sel_json_save_path, sel_tasks)
        save_json(all_json_save_path, all_tasks)


    def preprocess_images(self, split: str, task: str, prep_dir : str = PREP_DIR, fetch_all: bool = False):
        # print(prep_dir)
        # exit()
        ### load download tasks (from json file)
        task_json_path = os.path.join(prep_dir, 'download_tasks', f'{split}_{self._all_img_urls_suffix}') if fetch_all else \
                         os.path.join(prep_dir, 'download_tasks', f'{split}_{self._sel_img_urls_suffix}') # [Official] selected img_urls for image captioning task
        print(f'loading {split}-download_tasks from: {task_json_path}')
        if not os.path.exists(task_json_path): 
            print(f'{task_json_path} does not exist. Building download tasks...')
            self._build_image_download_tasks(split)
        tasks = load_json(task_json_path)

        # for entry_key, entry_value in list(tasks.items())[:10]:
        #     print(f"{entry_key}: {entry_value}")
        # exit()
        
        ### check number of tasks (https://github.com/google-research-datasets/wit/blob/main/wikiweb2m.md)
        if fetch_all: assert len(tasks) in [5340694, 299057, 300666], f'{split}: {len(tasks)} -> check official dataset specification (train: 5340708, val: 299057, test: 300666)'
        else:         assert len(tasks) in [2222810, 124703, 124188], f'{split}: {len(tasks)} -> train: 2222810, val: 124703, test: 124188'
        print(f'{split}-images2download (download tasks loaded): {len(tasks)}')

        ### official instruction - only keep specific images (i.e., jpeg, jpg and png)
        img_suffix_kept = ['jpeg', 'jpg', 'png']
        tasks = {k:v for k, v in tasks.items() if v.split('.')[-1].lower() in img_suffix_kept}
        print(f'{split}-images2download (after filtering img_types): {len(tasks)}')

        ### image deduplication: convert raw_urls -> dedup_urls + img_string_ids (use saved img_string_ids to replace redundant img_urls)
        _unique_urls = set(tasks.values())
        dedup_memo = {} # store the first img_fname for each unique img_url 
        for img_string_id in tqdm.tqdm(tasks.keys(), desc=f'{split}-deduplicating images'):

            img_url = tasks[img_string_id]
            if img_url not in dedup_memo: # first time seeing this img_url -> saving its img_string_id 
                assert is_online_url('http'), f'invalid online img_url (not starts with "http"): {img_url}'
                dedup_memo[img_url] = img_string_id
            else:                         # redundant img_url, rewrite the duplicate img_url with the stored img_string_id
                tasks[img_string_id] = dedup_memo[img_url]
        assert len(dedup_memo) == len(_unique_urls), 'deduplication failed'
        print(f'{split}-deduplicated images: {len(tasks)} (raw) -> {len(dedup_memo)} (deduplicated)')

        ### [definition] build one job 
        def build_one_download_job(save_dir, img_string_id, img_url):
            # destination path (local)
            img_save_path = self.get_img_local_path_from_string_id(save_dir, img_string_id)
            os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
            if os.path.exists(img_save_path): return None # skip existing images

            # source path (local or online)
            if not is_online_url(img_url): # deduplicated tasks (img_url -> task_name)
                img_url = self.get_img_local_path_from_string_id(save_dir, img_url)


            return (img_url, img_save_path)
        def build_one_convert_job(save_dir, img_string_id, _):
            tmp_path = self.get_img_local_path_from_string_id(save_dir, img_string_id)
            
            # destination path (local '*/strings/*')
            txt_save_path = tmp_path.replace(tmp_path.split('.')[-1], 'txt')
            os.makedirs(os.path.dirname(txt_save_path), exist_ok=True)
            if os.path.exists(txt_save_path): return None

            # source path (local '*/images/*')
            img_load_path = tmp_path.replace('/strings/', '/images/')
            if not os.path.exists(img_load_path): return None
            
            return (img_load_path, txt_save_path)

        ### build jobs
        if task == 'download_img':
            save_dir = os.path.join(prep_dir, 'images', split)
            print(f'downloading {split}-images to: {save_dir}')
            build_1_job = build_one_download_job
            mp_worker = download_worker
            num_workers = None
        elif task == 'convert_img2txt':
            save_dir = os.path.join(prep_dir, 'strings', split)
            print(f'converting {split}-strings to: {save_dir}')
            build_1_job = build_one_convert_job
            mp_worker = convert_worker
            num_workers = 2
        else:
            raise NotImplementedError

        jobs = []
        for sample_idx, (img_string_id, img_url) in tqdm.tqdm(enumerate(tasks.items()), desc=f'building {split}-jobs'):
            if DEBUG_RUN_SUBSET and sample_idx > 500: break


            ret = build_1_job(save_dir, img_string_id, img_url)
            if ret is not None:
                (src, dst) = ret
                jobs.append((src, dst))
            
        print('jobs: ', len(jobs))

        from utils.utils_mp import get_pool
        pool = get_pool(num_workers)
        with pool as p:
            r = list(tqdm.tqdm(p.map(mp_worker, jobs), total=len(jobs)))



if __name__ == '__main__':
    print(ImageDataParser(filepath=1))