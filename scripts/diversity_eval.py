# Code for evaluate Ctrl-CIC Diversity


import os
from starter.env_getter import get_env
from math import ceil
from utils.utils_io import load_json, save_json
from utils.utils import distinct_n_corpus_level

if __name__ == '__main__':
    PREP_DIR = get_env('PREP')
    CKPT_DIR = get_env('CKPT')

    exp_name = '231009-162123-pid311299-full_word_prefix-foraneous'
    ckpt_idx = 2760000


    dir = os.path.join(CKPT_DIR, 'CIC', 'experiments', exp_name)
    json_path = os.path.join(dir, f'checkpoint-{ckpt_idx}', 'ccic_outputs_5002.json')

    predictions = load_json(json_path)
    if 'clip_score' in predictions:
        clip_score = predictions.pop('clip_score')
    if 'highlight_score' in predictions:
        highlight_score = predictions.pop('highlight_score')

    grouped_dict = {}
    # Iterate through each item in the original dictionary
    for key, value in predictions.items():
        key_split = key.split('_')
        main_key = key_split[0]

        # Append the value to the corresponding list in the grouped dictionary
        if main_key in grouped_dict:
            grouped_dict[main_key].append(value)
        else:
            grouped_dict[main_key] = [value]

    print(len(grouped_dict))
    div_1_list = []
    div_2_list = []
    for k, v in grouped_dict.items():

        div_1 = distinct_n_corpus_level(v, 1)
        div_1_list.append(div_1)

        div_2 = distinct_n_corpus_level(v, 2)
        div_2_list.append(div_2)

    print(exp_name)
    final_div_1 = sum(div_1_list) / len(div_1_list)
    print('div_1', round(final_div_1 * 100, 2))

    final_div_2 = sum(div_2_list) / len(div_2_list)
    print('div_2', round(final_div_2 * 100, 2))
