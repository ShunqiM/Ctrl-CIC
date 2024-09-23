import numpy as np
import glob
import tensorflow.compat.v1 as tf
from collections import defaultdict
import tqdm
import os
from pydprint import dprint as dp
from starter.env_getter import get_env

DATA_DIR = get_env('DATA')
SPLITS = get_env('SPLITS')

DEBUG_RUN_SUBSET = False

class BaseDataParser():
    def __init__(self,
                filepath: str = 'wikiweb2m-*',
                path: str = DATA_DIR):
        self.filepath = filepath
        self.path = path
        self.data = defaultdict(list)
        self.splits = SPLITS
        self.split_datasets = {}


    
    def _load_tfrecord_dataset(self, split:str):
        assert split in self.splits, f'split must be one of {self.splits}'

        # detailed explanation: https://github.com/google-research-datasets/wit/blob/main/wikiweb2m.md#tfrecord-features
        context_feature_description = {
            'split': tf.io.FixedLenFeature([], dtype=tf.string),
            'page_title': tf.io.FixedLenFeature([], dtype=tf.string),
            'page_url': tf.io.FixedLenFeature([], dtype=tf.string),
            'clean_page_description': tf.io.FixedLenFeature([], dtype=tf.string),
            'raw_page_description': tf.io.FixedLenFeature([], dtype=tf.string),
            'is_page_description_sample': tf.io.FixedLenFeature([], dtype=tf.int64),
            'page_contains_images': tf.io.FixedLenFeature([], dtype=tf.int64),
            'page_content_sections_without_table_list': tf.io.FixedLenFeature([] , dtype=tf.int64)
        }

        sequence_feature_description = {
            'is_section_summarization_sample': tf.io.VarLenFeature(dtype=tf.int64),
            'section_title': tf.io.VarLenFeature(dtype=tf.string),
            'section_index': tf.io.VarLenFeature(dtype=tf.int64),
            'section_depth': tf.io.VarLenFeature(dtype=tf.int64),
            'section_heading_level': tf.io.VarLenFeature(dtype=tf.int64),
            'section_subsection_index': tf.io.VarLenFeature(dtype=tf.int64),
            'section_parent_index': tf.io.VarLenFeature(dtype=tf.int64),
            'section_text': tf.io.VarLenFeature(dtype=tf.string),
            'section_clean_1st_sentence': tf.io.VarLenFeature(dtype=tf.string),
            'section_raw_1st_sentence': tf.io.VarLenFeature(dtype=tf.string),
            'section_rest_sentence': tf.io.VarLenFeature(dtype=tf.string),
            'is_image_caption_sample': tf.io.VarLenFeature(dtype=tf.int64),
            'section_image_url': tf.io.VarLenFeature(dtype=tf.string),
            'section_image_mime_type': tf.io.VarLenFeature(dtype=tf.string),
            'section_image_width': tf.io.VarLenFeature(dtype=tf.int64),
            'section_image_height': tf.io.VarLenFeature(dtype=tf.int64),
            'section_image_in_wit': tf.io.VarLenFeature(dtype=tf.int64),
            'section_contains_table_or_list': tf.io.VarLenFeature(dtype=tf.int64),
            'section_image_captions': tf.io.VarLenFeature(dtype=tf.string),
            'section_image_alt_text': tf.io.VarLenFeature(dtype=tf.string),
            'section_image_raw_attr_desc': tf.io.VarLenFeature(dtype=tf.string),
            'section_image_clean_attr_desc': tf.io.VarLenFeature(dtype=tf.string),
            'section_image_raw_ref_desc': tf.io.VarLenFeature(dtype=tf.string),
            'section_image_clean_ref_desc': tf.io.VarLenFeature(dtype=tf.string),
            'section_contains_images': tf.io.VarLenFeature(dtype=tf.int64)
        }

        def _parse_function(example_proto):
            return tf.io.parse_single_sequence_example(example_proto,
                                                        context_feature_description,
                                                        sequence_feature_description)

        suffix = '.tfrecord*'

        glob_path = self.path + '/' + self.filepath + suffix
        split_paths = [p for p in glob.glob(glob_path) if split in p]
        print(f'TFRecordDataset-{split}: {split_paths}')

        raw_dataset = tf.data.TFRecordDataset(split_paths, compression_type=None)

        self.split_datasets[split] =  raw_dataset.map(_parse_function)

    def parse_data(self, split:str):
        if split not in self.split_datasets: self._load_tfrecord_dataset(split)

        for d in tqdm.tqdm(self.split_datasets[split]):
            assert len(d) == 2
            (ctx, seq) = d
            assert len(ctx) == 8 and len(seq) == 25

            dp(seq['section_title'])
            dp(seq['section_index'])
            if not ctx['page_contains_images']: continue
            print(ctx['page_url'].numpy().decode())

            assert np.allclose(seq['section_image_url'].indices.numpy(), seq['section_image_mime_type'].indices.numpy()) and \
                    np.allclose(seq['section_image_url'].indices.numpy(), seq['section_image_width'].indices.numpy()) and \
                        np.allclose(seq['section_image_url'].indices.numpy(), seq['section_image_height'].indices.numpy()), 'their indices are all the same'
            assert seq['section_image_url'].indices.ndim == 2 and seq['section_image_url'].indices.shape[1] == 2, 'indices.shape: (section_id, image_id within section)'
            
            _split = ctx['split'].numpy().decode()
            assert _split == split, f'{_split} != {split}'

            self.data[split].append(d)


if __name__ == '__main__':
    parser = BaseDataParser()
    parser.parse_data(split='train')
    #parser.parse_data(split='val')
    #parser.parse_data(split='test')
    print((len(parser.data['train']), len(parser.data['val']), len(parser.data['test'])))