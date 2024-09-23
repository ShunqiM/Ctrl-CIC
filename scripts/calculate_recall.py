# Code for evaluate Ctrl-CIC Recall

import os
from utils.utils_io import load_json
from starter.env_getter import get_env
from tqdm import tqdm
PREP_DIR = get_env('PREP')
CKPT_DIR = get_env('CKPT')

exp_name = '231009-162123-pid311299-full_word_prefix-foraneous'
ckpt_idx = 2760000
highlight_path = 'sample_5_highlight_1_v2.json'

highlight_path = os.path.join(PREP_DIR,'selected_highlights', highlight_path)
highlight_samples = load_json(highlight_path)

dir = os.path.join(CKPT_DIR, 'CIC', 'experiments', exp_name)

json_path = os.path.join(dir, f'checkpoint-{ckpt_idx}', 'ccic_outputs_5002.json')



ccic_results = load_json(json_path)

recalls = []

if 'clip_score' in ccic_results:
    ccic_results.pop('clip_score')
if 'highlight_score' in ccic_results:
    ccic_results.pop('highlight_score')

for k, v in tqdm((ccic_results.items())):
    original_index = (k[:-2])
    ith_highlight = int(k[-1])
    highlight_segments = highlight_samples[original_index][ith_highlight]
    highlight_cnt = len(highlight_segments)
    assert highlight_cnt > 0
    present_cnt = 0

    for highlight_phrase in highlight_segments:
        if highlight_phrase.strip().lower() in v.strip().lower():
            present_cnt += 1

    recall = present_cnt / highlight_cnt
    recalls.append(recall)


avg_recall = sum(recalls) / len(recalls)
print('highlight recall:', avg_recall)