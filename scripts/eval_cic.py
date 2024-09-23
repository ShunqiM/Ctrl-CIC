# Code used to evaluate traditional CIC performance 

import os
from utils.utils import compute_metrics
from utils.utils_io import load_json
from tqdm import tqdm
from starter.env_getter import get_env
from src.pipeline.EvaluatorWrapper import remove_prefix_from_list
from transformers import AutoTokenizer
import pandas as pd
PREP_DIR = get_env('PREP')
CKPT_DIR = get_env('CKPT')
pred_strs = []
target_strs = []
# For P-Ctrl, the generated prefix needs to be removed for CIC evaluation
remove_prefix = True
csv_name = "test_image_dict_v7.csv"
df = pd.read_csv(os.path.join(PREP_DIR, csv_name))
df = df.drop(df[df['exist'] == 0].index)
df = df.drop('exist', axis=1)

tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base")

dir = "/mnt/sdb/CKPT/MMWebpage/CIC/experiments/231009-162123-pid311299-full_word_prefix-foraneous/checkpoint-2760000/test_results"
for index in tqdm(range(123435)):
    original_index = index

    name = os.path.join(dir, (str(original_index) + '.json'))
    dict = load_json(name)

    pred_strs.append(dict['predicted'])
    target_strs.append(dict['target'])


if remove_prefix:
    pred_str = remove_prefix_from_list(pred_strs)
    label_str = remove_prefix_from_list(target_strs)
    pred_ids = tokenizer(pred_str)['input_ids']
    labels_ids = tokenizer(label_str)['input_ids']
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)


print(len(pred_str), len(label_str))
results = compute_metrics(pred_str, label_str)
print(results)