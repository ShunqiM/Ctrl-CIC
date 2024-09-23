# Generate Highlight Files based on the highlight selection response from GPT-4

import os
import json
import argparse
from tqdm import tqdm
import time
import pandas as pd
from utils.utils_io import load_pkl, load_json, save_json
from starter.env_getter import get_env
from transformers import AutoTokenizer
import shutil
import itertools
import random

PREP_DIR = get_env('PREP')
CKPT_DIR = get_env('CKPT')

def get_context(sample_idx, sec_idx):
    pkl_path = os.path.join(PREP_DIR, 'extracted_texts', split, (str(sample_idx) + '.pkl'))
    text_data = load_pkl(pkl_path)
    sec_title, sec_text = text_data['section_dict'][sec_idx]
    # while titles are added to highlight selection for more comprehensive understanding of the context, we only use highlights from the section texts instead of titiles, as titles itself are treated a bit specially with unique proceeding tokens  
    # Ideally, during highlight selection should have explain to GPT that first two sentence are titles and should not be selected, but explaining that in prompt can be tedious and might not distract GPT, while i could simply remove titles from highlight grounding filtering to achieve the same effect.  
    return sec_text

def sample_from_combination(phrases, n, n_highlight_segments = 1, seed = 42):

    random.seed(seed) 
    
    all_combinations = []
    if n_highlight_segments == -1:
        for r in range(1, len(phrases) + 1):
            all_combinations.extend(itertools.combinations(phrases, r))

    else:
        all_combinations.extend(itertools.combinations(phrases, n_highlight_segments))
    sampled_combinations = random.sample(all_combinations, n)
    return sampled_combinations

def filter_shorter_subset_phrases(phrases):
    # Sort phrases by length in descending order
    phrases_sorted = sorted(set(phrases), key=len, reverse=True)

    filtered_phrases = []
    for i, phrase in enumerate(phrases_sorted):
        if all(phrase not in other for other in phrases_sorted[:i]):
            filtered_phrases.append(phrase)

    return filtered_phrases


split = 'test'
csv_path = os.path.join(PREP_DIR, f"{split}_image_dict_ccic_fullset.csv")
df = pd.read_csv(csv_path)
df['original_index'] = df.index

# print(df[:100])
df = df[df['use_in_ccic'] == 'True']
df = df.reset_index(drop=True)

# df = df[10000:20000]

json_files = [os.path.join(PREP_DIR, 'prompts', 'fullset', f"test_highlight_fullset_prompt_fullset_{i}_response.json") for i in range(6)]
raw_highlights = {}
combined_data = {}
for file in json_files:
    with open(file, 'r') as f:
        data = json.load(f)
        combined_data.update(data)

for key, value in combined_data.items():
    # Convert key to integer if it's a digit, otherwise keep as is
    new_key = int(key) if key.isdigit() else key
    raw_highlights[new_key] = value

 
print(len(raw_highlights), len(df))



max_samples = 50000
n_sample = 5
n_highlight_segments = 2

highlight_dict = {}


print(df)
cnt = {}
new_rows = []

invalid_format_cnt = 0
comma_seperator_cnt = 0
insufficient_grounded_cnt = 0

global_seed = 0

total_found = 0
valid_found = 0

for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    sample_idx, img_path, img_idx, sec_idx, inclusion, original_index = row
    if sample_idx not in raw_highlights:
        print('missing sample', sample_idx, original_index)
        exit()
        continue

    if len(new_rows) >= max_samples:
        break

    raw_h = raw_highlights[sample_idx]
    # valid = True
    if '|' in raw_h:
        highlights = raw_h.split('|')
    elif ', ' in raw_h:
        highlights = raw_h.split(',')
        comma_seperator_cnt += 1

    else:
        # valid = False
        invalid_format_cnt += 1
        continue

    hs = [h.strip() for h in highlights]
    total_found += len(hs)

    sec_text = get_context(sample_idx, sec_idx)
    lower_sec_text = sec_text.lower()
    
    # remove ungrounded words
    found_phrases = [(phrase, sec_text.index(phrase)) for phrase in hs if phrase in sec_text]
    # should have done this, but the lack of it have lead to inevitable manual fix
    found_phrases = [p for p in found_phrases if p != ""]

    found_phrases = filter_shorter_subset_phrases(found_phrases)

    valid_found += len(found_phrases)

    if len(found_phrases) < n_highlight_segments * n_sample:
        insufficient_grounded_cnt += 1
        continue


    sampled_phrases = sample_from_combination(found_phrases, n_sample, n_highlight_segments, global_seed)
    # changing the seed for the sample function, so that different samples with same context will not have the same highlight
    global_seed += 1


    sorted_sampled_phrases = [tuple(sorted(tup, key=lambda x: x[1])) for tup in sampled_phrases]

    highlight_samples = [[phrase for phrase, index in tup] for tup in sorted_sampled_phrases]



    assert len(highlight_samples) == n_sample
    assert all(len(highlighted_segments) == n_highlight_segments for highlighted_segments in highlight_samples)
    highlight_dict[original_index] = highlight_samples


    for i in range(n_sample):
        key = f'{original_index}_{i}'
        # print(original_index)
        # print(cur_prompt)
        new_rows.append({
            'sample_idx': sample_idx,
            'img_path': img_path,
            'img_idx': img_idx,
            'sec_idx': sec_idx,
            'original_index': original_index,
            'key': key
        })


highlight_save_path = os.path.join(PREP_DIR, 'selected_highlights', f"sample_{n_sample}_highlight_{n_highlight_segments}_fullset.json")
save_json(highlight_save_path, highlight_dict)

print(invalid_format_cnt, comma_seperator_cnt, insufficient_grounded_cnt)
print(len(highlight_dict), len(df))
print(f'{valid_found} valid highlight out of {total_found} highlights in responses found')
new_df = pd.DataFrame(new_rows)
# new_df.to_csv(os.path.join(PREP_DIR, f"test_image_dict_ccic_fullset_v1.csv"), index=False)
# new_df.to_csv(os.path.join(PREP_DIR, f"test_image_dict_ccic_fullset_multiple_v1.csv"), index=False)
# new_df.to_csv(os.path.join(PREP_DIR, f"test_image_dict_ccic_fullset_div_v1.csv"), index=False)
new_df.to_csv(os.path.join(PREP_DIR, f"test_image_dict_ccic_fullset_multiple_div_v1.csv"), index=False)


