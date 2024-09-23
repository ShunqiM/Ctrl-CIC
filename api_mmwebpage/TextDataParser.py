import os
import urllib.request
import tensorflow.compat.v1 as tf
import numpy as np
import tqdm
from utils.utils_io import read_image, save_json, load_json, save_pkl
from .BaseDataParser import BaseDataParser
import pandas as pd
DEBUG_RUN_SUBSET = False
from starter.env_getter import get_env

PREP_DIR = get_env('PREP')

is_online_url = lambda x: x.startswith('http')

# Extract the local texts for Ctrl-CIC experiments
class TextDataParser(BaseDataParser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
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

        if len(img_fname) > 210: img_fname = img_fname[:210] 

        sample_save_dir = os.path.join(save_root, str(sample_idx))
        img_local_path = os.path.join(sample_save_dir, img_fname)
        return img_local_path
    
    def get_json_local_path(self, save_root, sample_idx):
        return os.path.join(save_root, (str(sample_idx) + '.json'))
    
    def get_pkl_local_path(self, save_root, sample_idx):
        return os.path.join(save_root, (str(sample_idx) + '.pkl'))
    
    def extract_txt(self, split: str, prep_dir : str = PREP_DIR):
        if split not in self.split_datasets: self._load_tfrecord_dataset(split)

        total_pages = {'train':1803225, 'val':100475, 'test':100833}
        img_save_dir = os.path.join(prep_dir, 'images', split)
        json_save_dir = os.path.join(prep_dir, 'extracted_texts', split)
        os.makedirs(json_save_dir, exist_ok=True)
            
        sample_ids = []
        img_paths = []
        json_paths = []
        img_ids = []
        sec_ids = []
        exist = []
        for sample_idx, sample_page in tqdm.tqdm(enumerate(self.split_datasets[split]), desc=f'building {split}-extraction_tasks', total=total_pages[split]):
            ctx, seq = sample_page

            if not ctx['page_contains_images']: continue

            # Pickle path have been used here, but save them as pickles will lead to large GPU usage at loading so they were saved as jsons instead
            
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


                ### official instruction - select subset of images for image captioning
                data_saved = False
                if img_used_for_cap:
                    img_path = self.get_img_local_path_from_string_id(img_save_dir, img_string_id)
                    img_paths.append(img_path)
                    sample_ids.append(sample_idx)
                    img_ids.append(i)
                    if not os.path.exists(img_path):
                        exist.append(0)
                    else:
                        exist.append(1)

                    # The index here in the web2m dataset refers to the relative sections they have, not the actual section
                    sec_idx = seq['section_image_url'].indices[i][0].numpy()
                    sec_ids.append(sec_idx)
                    assert sec_idx in seq['section_text'].indices.numpy()[:, 0], f"Section text not exist in {str(sample_idx)}"
       
                    if not data_saved:
                        sec_num = seq['section_index'].indices.numpy()[-1, 0] + 1
                        assert sec_num == seq['section_text'].indices.numpy().shape[0], f"Sections text missing in {str(sample_idx)}"
                        assert sec_num == seq['section_index'].indices.numpy().shape[0], f"Sections index missing in {str(sample_idx)}"
                        assert sec_num == seq['section_title'].indices.numpy().shape[0], f"Sections title missing in {str(sample_idx)}"
                        sec_dict = {} # will be a dict of string tuples sec_idx: (title, ctx)
                        for x in range(sec_num):
                            # Sections index might be like: 0,1,2,3,4,5,8 -- check data index 36 for example
                            
                            section_idx = x
                            section_title = seq['section_title'].values[x].numpy().decode('utf-8')
                            section_ctx = seq['section_text'].values[x].numpy().decode('utf-8')

                            sec_dict[section_idx] = (section_title, section_ctx)

                        page_url = ctx['page_url'].numpy().decode('utf-8')
                        page_title = ctx['page_title'].numpy().decode('utf-8')
                        image_captions = seq['section_image_clean_ref_desc'].values.numpy()
                        attribution_captions = seq['section_image_clean_attr_desc'].values.numpy()
                        alt_texts = seq['section_image_alt_text'].values.numpy()
                        full_captions = seq['section_image_captions'].values.numpy()

                        image_captions_sections = seq['section_image_clean_ref_desc'].indices.numpy()
                        attribute_sections = seq['section_image_clean_attr_desc'].indices.numpy()
                        assert np.array_equal(attribute_sections, image_captions_sections)
                        assert image_captions.shape[0] == len(seq['section_image_url'].values), f"Image caption missing in {str(sample_idx)}"
                        caption_dict = {}
                        attribute_dict = {}
                        alt_dict = {}
                        full_captions_dict = {}
                        for i, v in enumerate(image_captions_sections):
                            if v[0] not in attribute_dict:
                                attribute_dict[v[0]] = {i: attribution_captions[i].decode('utf-8')}
                            else: 
                                attribute_dict[v[0]][i] = attribution_captions[i].decode('utf-8')

                        for i, v in enumerate(image_captions_sections):
                            if v[0] not in alt_dict:
                                alt_dict[v[0]] = {i: alt_texts[i].decode('utf-8')}
                            else: 
                                alt_dict[v[0]][i] = alt_texts[i].decode('utf-8')

                        for i, v in enumerate(image_captions_sections):
                            if v[0] not in full_captions_dict:
                                full_captions_dict[v[0]] = {i: full_captions[i].decode('utf-8')}
                            else: 
                                full_captions_dict[v[0]][i] = full_captions[i].decode('utf-8')
                        
                        for i, v in enumerate(image_captions_sections):
                            # print(v[0]) # sec index of the caption
                            if v[0] not in caption_dict:
                                # caption_dict[v[0]] = [(i, image_captions[i].decode('utf-8'))] # a tuple of image index and caption
                                caption_dict[v[0]] = {i: image_captions[i].decode('utf-8')}
                            else: 
                                # caption_dict[v[0]].append((i, image_captions[i].decode('utf-8')))
                                caption_dict[v[0]][i] = image_captions[i].decode('utf-8')

                        data2save = {'page_url': page_url,
                                     'page_title': page_title,
                                     'image_captions': caption_dict,
                                     'section_dict': sec_dict}
                        
                        """
                        # Extract additional image captions for Ctrl-CIC evaluation etc
                        data2save = {'attribution': attribute_dict,
                                     'alt_texts': alt_dict,
                                     'full_caption': full_captions_dict}
                        pkl_path = os.path.join(json_save_dir, (str(sample_idx) + '_caps.pkl'))
                        save_pkl(pkl_path, data2save)
                        """
                        
                        json_path = self.get_json_local_path(json_save_dir, sample_idx)
                        save_json(json_path, {'ctx': ctx, 'seq': seq})
                        save_json(json_path, data2save)
                        data_saved = True
                    # if len(img_ids) >= 100: break
                # """
                # exit()
            else: continue
            break

        data_dict = {'sample_idx':sample_ids, 'img_path':img_paths, 'img_idx':img_ids, 'sec_idx': sec_ids, 'exist': exist}

        df = pd.DataFrame.from_dict(data_dict, orient='index').transpose()
        csv_save_path = os.path.join(prep_dir, f'{split}_image_dict_v7.csv')
        df.to_csv(csv_save_path, index=False)





if __name__ == '__main__':
    print(TextDataParser(filepath=1))