import json
import re
from scipy.stats import spearmanr, pearsonr, kendalltau
from starter.env_getter import get_env
from utils.utils_io import load_json
import os
import math
PREP_DIR = get_env('PREP')
CKPT_DIR = get_env('CKPT')


def calculate_correlation(pred_score, human_score, result):
    assert len(pred_score) == len(human_score)

    if (len(result) == 0):
        result = {'pearson': 0, 'spearman': 0, 'kendalltau': 0}
    result['pearson'] += pearsonr(pred_score, human_score)[0]
    result['spearman'] += spearmanr(pred_score, human_score)[0]
    result['kendalltau'] += kendalltau(pred_score, human_score)[0]

    return result

def extract_scores(data_str):
    # Define the regex pattern to extract the metrics and their scores
    pattern = r'- ([\w\s]+): (\d+)'
    # Find all matches in the string
    matches = re.findall(pattern, data_str)
    
    # Convert the matches to a dictionary
    scores = {metric.strip(): int(score) for metric, score in matches}
    return scores

def extract_scores(block):
    pattern_scores = r"- ([\w\s]+): (\d+)"
    matches = re.findall(pattern_scores, block)
    return {metric.strip(): int(score) for metric, score in matches}


a1_string_list = []
a2_string_list = []

text = "\n\n[ASSISTANT1-Score]:\n5\n\n[ASSISTANT2-Score]:\n7"
pattern = r'\[ASSISTANT1-Score\]:\s*(\d+)\s*\[ASSISTANT2-Score\]:\s*(\d+)'
pattern = r'\[ASSISTANT1-Score\]:\s*(\d+).*?\[ASSISTANT2-Score\]:\s*(\d+)'

path = os.path.join(PREP_DIR, 'prompts', 'new', 'merged_5000_response.json')
query_path = os.path.join(PREP_DIR, 'prompts', 'new', 'merged_5000.json')


model_names = [
    'rc', 
    'gpt',
    'pc',
    'ext',
    'tune',
    'llava'
]
model_names = [
    'llama'
]


prompts_list = ['combined']
test = 'combined'
test_model = 'llama'

all_metrics = ['Relevance with Context', 'Relevance with Highlight', 'Consistency with Image', 'Overall Quality']

# Read JSON file and convert to dictionary
with open(path, 'r') as f:
    data = json.load(f)


with open(query_path, 'r') as f:
    query = json.load(f)
    processed_query = {}
    for k, v in query.items():
        if isinstance(v, list):
            assert 'text' in v[0].keys()
            processed_query[k] = v[0]['text']
    query = processed_query

if test_model == 'llama':
    data = {key + 'combined_llama': value for key, value in data.items()}
    query = {key + 'combined_llama': value for key, value in query.items()}

cnt = 0
ass_1_cnt = 0
ass_1_sum = 0
ass_2_cnt = 0
ass_2_sum = 0
max_val = 0
max_index = []

model_scores = {'Relevance with Context': {}, 'Relevance with Highlight': {}, 'Consistency with Image': {}, 'Overall Quality': {}}
ref_scores =  {'Relevance with Context': {}, 'Relevance with Highlight': {}, 'Consistency with Image': {}, 'Overall Quality': {}}


for key, value in data.items():
    first_part, second_part = key.split("_")[:2]
    model = key.split("_")[-1]
    original_index = int(first_part)
    human_key = f'{original_index}_{model}'


    accumulated_scores = {}
    pattern_blocks = r"\[ASSISTANT\d+-Score\]:.*?(?=\[ASSISTANT\d+-Score\]|$)"
    assistant_blocks = re.findall(pattern_blocks, value, re.DOTALL)
    pattern_scores = r"- ([\w\s]+): (\d+)"

    
    all_scores = {}

    for block in assistant_blocks:
        # Extract the assistant's name
        assistant_name = re.search(r"\[ASSISTANT\d+-Score\]", block).group()
        # Extract the individual scores from the block
        scores_dict = extract_scores(block)
        if 'ASSISTANT1' in assistant_name:
            if original_index % 2 == 0:
                for m, raw_s in scores_dict.items():
                    ref_scores[m].update({human_key: raw_s})
            else:
                for m, raw_s in scores_dict.items():
                    model_scores[m].update({human_key: raw_s})
        elif 'ASSISTANT2' in assistant_name:
            if original_index % 2 == 1:
                for m, raw_s in scores_dict.items():
                    ref_scores[m].update({human_key: raw_s})
            else:
                for m, raw_s in scores_dict.items():
                    model_scores[m].update({human_key: raw_s})
        else:
            print(value)
            print('Unknow Case')



def get_score(ref_scores, model_scores, m, macro = False):
    ref_score_dict = ref_scores[m]
    model_score_dict = model_scores[m]
    if macro:
        ref_score_sum = sum(ref_score_dict.values())
        model_score_sum = sum(model_score_dict.values())
        final_score = model_score_sum / ref_score_sum
        final_score = math.log(final_score, 5)
        return final_score
    else:
        final_score_list = []
        for k, v in ref_score_dict.items():
            final_score = model_score_dict[k] / v
            final_score = math.log(final_score, 5)
            
            final_score_list.append(final_score)
        return sum(final_score_list) / len(final_score_list)
    
all_scores = {}
ref_all_scores = {}
model_all_scores = {}


for model in model_names:
    all_scores[model] = {}
    ref_all_scores[model] = {}
    model_all_scores[model] = {}
for metric, score_dict in model_scores.items():
    for model in model_names:
        all_scores[model][metric] = {} 
        ref_all_scores[model][metric] = {}
        model_all_scores[model][metric] = {}
        for id, score in score_dict.items():
            numeric_id = int(id.split('_')[0])
            if model_scores[metric][id] == 0:
                model_scores[metric][id] += 1
                ref_scores[metric][id] += 1
            div = model_scores[metric][id] / ref_scores[metric][id]
            
            if model in id:
                all_scores[model][metric][id] = model_scores[metric][id] / ref_scores[metric][id] 
                ref_all_scores[model][metric][id] = ref_scores[metric][id] 
                model_all_scores[model][metric][id] = model_scores[metric][id] 

                all_scores[model][metric][id] = math.log(all_scores[model][metric][id], 5)
                

for m in model_names:
    print(m)
    for met in all_metrics:
        final_score = get_score(ref_all_scores[m], model_all_scores[m], met)
        final_score = pow(5, final_score)
        print(met, final_score)


