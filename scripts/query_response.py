import os
from utils.tools import AzureOpenAIQueryClient, query_worker
from utils import utils_mp
from starter.env_getter import get_env
from utils.utils_io import save_json, load_json
from scripts.split_json import split_dict
import json
from tqdm import tqdm

PREP_DIR = get_env('PREP')


def merge_output_files(num_chunks, base_output_path, final_output_path):
    combined_results = {}
    for i in range(num_chunks):
        chunk_output_path = os.path.join(base_output_path, f"chunk_{i}.json")
        with open(chunk_output_path, 'r') as file:
            chunk_results = json.load(file)
            combined_results.update(chunk_results)

    with open(final_output_path, 'w') as file:
        json.dump(combined_results, file, indent=4)

def GPT_run(src_json_path, num_processes=10, prefix_desc='', chunk_size = 1000):

    queries = load_json(src_json_path)
    
    print(src_json_path, len(queries))


    chunks = list(split_dict(queries, chunk_size))

    base_output_path = src_json_path.replace('.json', '')
    if not os.path.exists(base_output_path):
        os.makedirs(base_output_path)

    # Process each chunk
    for i, chunk in tqdm(enumerate(chunks), position = 0, total = len(chunks)):

        jobs = [(index, query, 'GPT4-V') for index, query in chunk.items()]


        responses = utils_mp.launch_multi_processes(query_worker, jobs, desc=f'[GPTEval{prefix_desc}] {src_json_path}',num_processes=num_processes, return_required=True)
        # responses = chunk.copy()

        chunk_output_path = os.path.join(base_output_path, f"chunk_{i}.json")
        save_json(chunk_output_path, responses)


    final_output_path = src_json_path.replace('.json', '_response.json')
    merge_output_files(len(chunks), base_output_path, final_output_path)


# def GPT
src_json_path = os.path.join(PREP_DIR, 'prompts', 'new', 'test_eval_combined_v1.json')

# Process in different chunks as GPT-4V have a limited rate.
GPT_run(src_json_path, chunk_size=100, num_processes = 1)