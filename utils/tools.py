import os
# import openai
from openai import OpenAI
import numpy as np
import time
import json
import multiprocessing
from multiprocessing import Pool
from starter.env_getter import get_env
from tqdm import tqdm

PREP_DIR = get_env('PREP')

client = OpenAI(api_key='your_open_ai_key')


query_num_thread = 10
balance_ratio = 3.3 



error_prefix = '[ERROR] '

def is_acceptable_response(content, response_as='completion'):
    assert response_as in ['completion', 'embedding', 'GPT4V'], f'unexpected response_as: {response_as}'

    if response_as == 'embedding':
        assert isinstance(content, (list, str)), type(content)
        if isinstance(content, list): return True
    else:
        assert isinstance(content, str), type(content)
        if not content.startswith(error_prefix): return True 

    if 'content_filter' in content: return True
    if 'content filtering policies' in content: return True

    return False

def response2content(response, response_as='completion'):
    assert response_as in ['completion', 'embedding', 'GPT4V'], f'unexpected response_as: {response_as}'
    if response_as == 'embedding': content = response.data[0].embedding
    else:
        choice_0 = response.choices[0]

        finish_reason = choice_0.finish_reason

        if finish_reason is None or finish_reason == 'stop': content = choice_0.message.content
        else: content = error_prefix + finish_reason
    return content
















def get_query_sleep_time(nthreads=None):
    if nthreads is None: nthreads = query_num_thread
    return nthreads / balance_ratio


def query_worker(job, do_sleep=True):
    assert len(job) == 3, 'job should now be (key, query, model_name), where model_name could be GPT4/GPT3.5 (recent upgrade)'
    key, query, model_name = job
    cli = AzureOpenAIQueryClient(model_name=model_name)
    res = cli.query(query)
    if do_sleep: time.sleep(get_query_sleep_time())
    return {key: res}

def embed_worker(job, do_sleep=False):
    key, query = job
    cli = AzureOpenAIQueryClient(model_name='embedding')
    res = cli.query(query)
    if do_sleep: time.sleep(get_query_sleep_time() * 1.6)
    return {key: res}

def split_text(text, N):
    return [text[i:i+N] for i in range(0, len(text), N)]
class AzureOpenAIQueryClient:
    char_limit = 40960 

    supported_model_ids = {
        'GPT4': 'gpt-4-0125-preview',  #This will correspond to the custom name you chose for your deployment when you deployed a model.
                        #    'GPT3.5': 'gpt-35-turbo-version-0301',
                            'GPT4-V': 'gpt-4-vision-preview',
                           'GPT3.5': 'gpt-3.5-turbo-0125',
                           'embedding': 'text-embedding-ada-002'}

    def get_model_deployment_id(self): return self.supported_model_ids[self.model_name]

    def __init__(self, temperature=0, model_name='GPT4'):
        self.model_name = model_name
        assert self.model_name in self.supported_model_ids, f'we only support {self.model_ids.keys()} for now'
        self.temperature = temperature
        self.response_as = 'embedding' if self.model_name == 'embedding' else 'completion'
        
    def query(self, query, max_trials=10):
        # core query function

        i = 1
        while True:
            try:
                content = self._query(query)
            except Exception as e_msg:
                content = error_prefix + str(e_msg)

            if is_acceptable_response(content, response_as=self.response_as): break
            
            if 'Rate limit reached' in content:
                print('usage limit reached, sleep for a minute')
                time.sleep(60)
                # Wait for human confirmation to proceed
                # user_input = input("Rate limit reached. Type 'continue' to retry or 'stop' to exit: ").strip().lower()
                # if user_input == 'stop':
                #     print("User requested to stop the process.")
                #     break


            # safty exit (for debug/develop usages)
            if max_trials > 0 and i == max_trials: 
                print(f'max_trials ({max_trials}) reached. ')
                break
        
            
            # retry for unacceptable response
            #dprint(query)
            print('\n(retry later...) ->', content)
            time.sleep(get_query_sleep_time())
            i += 1
        return content
        
    def _query(self, query):
        model_deployment_id = self.get_model_deployment_id()

        if self.response_as == 'embedding':
            response = client.embeddings.create(model=model_deployment_id, input=query)
        else:
            _messages = [
                {"role": "user", "content": query},
            ]

            response = client.chat.completions.create(model=model_deployment_id,
            temperature=self.temperature,
            messages=_messages,
            max_tokens=4096)
            # print(response)
        return response2content(response, response_as=self.response_as)
    
def process_queries(jobs_dict, output_json_path, worker_function, num_workers=2):

    # Create jobs for the worker function
    jobs = [(index, query, 'GPT3.5') for index, query in jobs_dict.items()]

    # Initialize multiprocessing pool
    with Pool(num_workers) as pool:
        results = pool.map(worker_function, jobs)

    # Combine results into a dictionary
    combined_results = {}
    for result in results:
        combined_results.update(result)

    # Write the results to the output JSON file
    with open(output_json_path, 'w') as file:
        json.dump(combined_results, file, indent=4)

def split_data(input_dict, chunk_size):
    """Split a dictionary into chunks of specified size."""
    return [dict(list(input_dict.items())[i:i + chunk_size]) for i in range(0, len(input_dict), chunk_size)]

def process_chunk(chunk, chunk_index, output_json_path, worker_function, num_workers=10):
    chunk_output_path = f"{output_json_path}_chunk_{chunk_index}.json"
    process_queries(chunk, chunk_output_path, worker_function, num_workers)

def merge_output_files(num_chunks, base_output_path, final_output_path):
    combined_results = {}
    for i in range(num_chunks):
        chunk_output_path = f"{base_output_path}_chunk_{i}.json"
        with open(chunk_output_path, 'r') as file:
            chunk_results = json.load(file)
            combined_results.update(chunk_results)
    
    with open(final_output_path, 'w') as file:
        json.dump(combined_results, file, indent=4)

if __name__ == '__main__':
    # Example usage
    input_json_path = os.path.join(PREP_DIR, 'prompts', 'test.json')  # Replace with the path to your input JSON file
    output_json_path = input_json_path.replace('.json', '_response.json')  # Replace with the path to your output JSON file
    print(output_json_path)
    # Read the input JSON file
    with open(input_json_path, 'r') as file:
        jobs_dict = json.load(file)

    chunk_size = 1000 

    # Split data into chunks
    chunks = split_data(jobs_dict, chunk_size)

    # Process each chunk
    for i, chunk in tqdm(enumerate(chunks)):
        process_chunk(chunk, i, output_json_path[:-4], query_worker)

    # Merge output files
    merge_output_files(len(chunks), output_json_path[:-4], output_json_path)
    
