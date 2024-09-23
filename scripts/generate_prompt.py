# Code for generate Prompt for various purpose -- Zero-shot CIC/Ctrl-CIC, Ctrl-CIC evaluation...

import os
import json
import argparse
from tqdm import tqdm
import time
import pandas as pd
from utils.utils_io import load_pkl, load_json, save_json
from starter.env_getter import get_env
from transformers import AutoTokenizer
import base64

PREP_DIR = get_env('PREP')
CKPT_DIR = get_env('CKPT')
# Refer to https://github.com/haotian-liu/LLaVA/blob/main/llava/eval/table/prompt.jsonl
# and https://github.com/nlpyang/geval/tree/main/prompts/summeval


def get_context_and_gt_caption(index, df, split, add_special_token = True, add_image_caption = True):
    
    sample_idx, img_path, img_idx, sec_idx, *original_index = df.iloc[index]

    pkl_path = os.path.join(PREP_DIR, 'extracted_texts', split, (str(sample_idx) + '.pkl'))
    text_data = load_pkl(pkl_path)
    pkl_path = os.path.join(PREP_DIR, 'extracted_texts', split, (str(sample_idx) + '_caps.pkl'))

    # ['ImageToken', 'SectionIndex', 'SectionTitle', 'SectionText', 'PageURL', 'PageTitle', 'ImageCaption']
    sec_title, sec_text = text_data['section_dict'][sec_idx]
    page_title = text_data['page_title']
    image_captions = text_data['image_captions'] # dict of dict
    target_text = image_captions[sec_idx][img_idx]
    if add_special_token:
        if len(sec_text) > 0:
            input_text = " ".join(['[PageTitle]', page_title, 
                            '[SectionTitle]', sec_title, '[SectionText]', sec_text,
                            ])
        else:
            input_text = " ".join(['[PageTitle]', page_title, 
                            '[SectionTitle]', sec_title
                            ])
    else:
        input_text = sec_text


    if add_image_caption:

        for k, v in image_captions[sec_idx].items():
            if k != img_idx and len(v) > 0:
                input_text = " ".join([input_text, '[ImageCaption]', v])

    return input_text, target_text


def get_image_description(index, df, split):
    sample_idx, img_path, img_idx, sec_idx, *original_index = df.iloc[index]
    pkl_path = os.path.join(PREP_DIR, 'extracted_texts', split, (str(sample_idx) + '_caps.pkl'))
    caption_data = load_pkl(pkl_path)
    image_alts = caption_data['alt_texts']
    image_attribute = caption_data['attribution']
    
    target_alt = image_alts[sec_idx][img_idx]
    target_attri = image_attribute[sec_idx][img_idx]
    if target_alt == "":
        return target_attri

    image_description = ", ".join([target_alt, target_attri])
    return image_description

def get_predicted_caption(index, prediction_path):
    name = os.path.join(prediction_path, (str(index) + '.json'))
    dict = load_json(name)
    return dict['predicted']


def constrain_string_length(text, tokenizer):
    
    MAX_TOKENS = 511
    tokenized = tokenizer.tokenize(text, return_tensors="pt")
    
    while len(tokenized) > MAX_TOKENS:
        last_space = text.rfind(' ')
        
        # If there's no space left (a single word remaining), just truncate a fixed amount
        if last_space == -1:
            return text
        else:
            text = text[:last_space]
        
        tokenized = tokenizer.tokenize(text, return_tensors="pt")
    
    return text

# Constraint the input string length for GPT-4-based CIC/Ctrl-CIC, for fair comparison.
def constrain_string_length_binary(text, tokenizer):
    
    MAX_TOKENS = 511
    
    start = 0
    end = len(text)
    
    while start < end:
        mid = (start + end) // 2
        substring = text[:mid]
        
        # Find the position of the last space in the substring
        last_space = substring.rfind(' ')
        
        # If a space was found, adjust the substring
        if last_space != -1:
            substring = substring[:last_space]
        
        tokenized = tokenizer.tokenize(substring, return_tensors="pt")
        
        if len(tokenized) > MAX_TOKENS:
            end = mid
        else:
            start = mid + 1
    
    return text[:start]

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
def encode_and_resize_image(image_path):
    from PIL import Image
    import io
    # Open the image
    _, file_extension = os.path.splitext(image_path)
    if file_extension.lower() in ['.jpg', '.jpeg']:
        image_format = 'JPEG'
    elif file_extension.lower() == '.png':
        image_format = 'PNG'
    else:
        raise ValueError(f"Unsupported image format {file_extension}")
    with Image.open(image_path) as img:
        # Get the size of the image
        width, height = img.size

        # Find the shorter edge
        shorter_edge = min(width, height)

        # Resize if the shorter edge is longer than 224
        if shorter_edge > 224:
            # Calculate the new size, preserving the aspect ratio
            aspect_ratio = width / height
            if width < height:
                new_width = 224
                new_height = round(224 / aspect_ratio)
            else:
                new_height = 224
                new_width = round(224 * aspect_ratio)

            # Resize the image
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Convert image to base64
        buffered = io.BytesIO()
        img.save(buffered, format=image_format)
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return img_str

def get_eval_prompt(prompt, context, highlight, description, caption):
    cur_prompt = prompt.replace('{{Document}}', context).replace('{{Highlight}}', highlight).replace('{{Attribution}}', description).replace('{{Caption}}', caption)
    return cur_prompt

def get_cic_prompt(prompt, context, description):
    cur_prompt = prompt.replace('{{Document}}', context).replace('{{Attribution}}', description)
    return cur_prompt

def get_highlight_prompt(prompt, context):
    cur_prompt = prompt.replace('{{Document}}', context)
    return cur_prompt

def get_ccic_prompt(prompt, context, description, highlight):
    cur_prompt = prompt.replace('{{Document}}', context).replace('{{Attribution}}', description).replace('{{Highlight}}', highlight)
    return cur_prompt

def get_eval_prompt(prompt, context, description, highlight, cap_1, cap_2):
    cur_prompt = prompt.replace('{{Document}}', context).replace('{{Attribution}}', description).replace('{{Highlight}}', highlight).replace('{{Caption_1}}', cap_1).replace('{{Caption_2}}', cap_2)
    return cur_prompt

def sufficient_len(row):
    # original_index = row['original_index']
    sample_idx, img_path, img_idx, sec_idx, original_index = row
    pkl_path = os.path.join(PREP_DIR, 'extracted_texts', split, (str(sample_idx) + '.pkl'))
    text_data = load_pkl(pkl_path)
    sec_title, sec_text = text_data['section_dict'][sec_idx]
    page_title = text_data['page_title']
    input_text = ". ".join([page_title, sec_title, sec_text])
    return len(input_text) >= 100

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--task', type=str, default='eval', choices=['cic', 'eval', 'highlight', 'ccic', 'eval_ah', 'highlight_fullset'])
    argparser.add_argument('--prompt_fp', type=str, default='prompts/eval_context_con_v3.txt')
    argparser.add_argument('--image_description', type=str, default='attribution', choices=['grit', 'attribution']) # attribution texts are the context-free caption given by the dataset, grit texts are the image caption generated by GRIT 
    argparser.add_argument('--subset_size', type=int, default=1000)
    argparser.add_argument('--compare_with_ref', type=bool, default=True)
    argparser.add_argument('--grit_as_ref', type=bool, default=False)
    argparser.add_argument('--add_image', type=bool, default=False)
    argparser.add_argument('--single_sample', type=bool, default=False)



    args = argparser.parse_args()
    prompt = open(args.prompt_fp).read()
    print(prompt)

    if args.task == 'cic':
        split = 'test'
        csv_path = os.path.join(PREP_DIR, f"{split}_image_dict_{args.subset_size}_filtered.csv")
        df = pd.read_csv(csv_path)

    elif args.task == 'eval':
        split = 'test'
        csv_path = os.path.join(PREP_DIR, f"{split}_image_dict_v7.csv")
        df = pd.read_csv(csv_path)
        df = df.drop(df[df['exist'] == 0].index)
        df = df.drop('exist', axis=1)

    if args.task == 'cic':
        tokenizer = AutoTokenizer.from_pretrained('t5-small')
        special_tokens = ['[ImageToken]', '[SectionIndex]', '[SectionTitle]', '[SectionText]', '[PageURL]', '[PageTitle]', '[ImageCaption]']
        token_dict = {'additional_special_tokens': special_tokens}
        num_added_tokens = tokenizer.add_special_tokens(token_dict)
        json_dict = {}
        tmp_df = df.drop('original_index', axis=1)
        image_json_dict = {}
        image_json_dict_subset = {}
        image_description_path = os.path.join(PREP_DIR, 'images_1000_response.json')
        image_descriptions = load_json(image_description_path)
        for index, row in tqdm(df.iterrows(), total = len(df)):
            original_index = row['original_index']
            sample_idx, img_path, img_idx, sec_idx = tmp_df.iloc[index]
            context, caption = get_context_and_gt_caption(index, tmp_df, split)
            context = constrain_string_length_binary(context, tokenizer)
            if args.image_description == 'attribution':
                image_description = get_image_description(index, tmp_df, split)
            else:
                image_description = image_descriptions[str(original_index)]
            cur_prompt = get_cic_prompt(prompt, context, image_description)
            json_dict[original_index] = cur_prompt
            
        save_path = os.path.join(PREP_DIR, 'prompts', f"{args.task}_prompt_grit_{args.subset_size}.json")
        save_json(save_path, json_dict)

        exit()

    if args.task == "ccic":

        tokenizer = AutoTokenizer.from_pretrained('t5-small')
        special_tokens = ['[ImageToken]', '[SectionIndex]', '[SectionTitle]', '[SectionText]', '[PageURL]', '[PageTitle]', '[ImageCaption]']
        token_dict = {'additional_special_tokens': special_tokens}
        num_added_tokens = tokenizer.add_special_tokens(token_dict)

        split = 'test'
        csv_path = os.path.join(PREP_DIR, f"test_image_dict_ccic_1000_multiple.csv")
        highlight_save_path = os.path.join(PREP_DIR, 'selected_highlights', f"sample_{5}_highlight_{2}.json")
        df = pd.read_csv(csv_path)
        highlight_samples = load_json(highlight_save_path)
        highlight_samples = {int(key): value for key, value in highlight_samples.items()}
        prompt_dict = {}
        # maintain a new df for our model to run inference
        image_description_path = os.path.join(PREP_DIR, 'images_1000_response.json')
        image_descriptions = load_json(image_description_path)
        new_rows = []
        for index, row in tqdm(df.iterrows(), total = 1000):
            sample_idx, img_path, img_idx, sec_idx , original_index, _ = df.iloc[index]
            highlight_segments_list = highlight_samples[original_index]
            context, caption = get_context_and_gt_caption(index, df, split)
            # context = constrain_string_length(context, tokenizer)
            context = constrain_string_length_binary(context, tokenizer)

            if args.image_description == 'attribution':
                image_description = get_image_description(index, df, split)
            else:
                image_description = image_descriptions[str(original_index)]
            for i, highlight_segments in enumerate(highlight_segments_list):
                highlight_string = ""
                for highlight in highlight_segments:
                    highlight_string = "".join([highlight_string, highlight, '\n'])
                cur_prompt = get_ccic_prompt(prompt, context, image_description, highlight_string)
                key = f'{original_index}_{i}'
                prompt_dict[key] = cur_prompt

                new_rows.append({
                    'sample_idx': sample_idx,
                    'img_path': img_path,
                    'img_idx': img_idx,
                    'sec_idx': sec_idx,
                    'original_index': original_index,
                    'key': key
                })
            # exit()
        # Save the results
        # save_path = os.path.join(PREP_DIR, 'prompts', f"{args.task}_prompt_{args.image_description}_{args.subset_size * 5}_v2.json")
        save_path = os.path.join(PREP_DIR, 'prompts', f"{args.task}_prompt_{args.image_description}_multiple.json")
                
        save_json(save_path, prompt_dict)
        new_df = pd.DataFrame(new_rows)

        new_df.to_csv(os.path.join(PREP_DIR, f"{split}_image_dict_ccic_{args.subset_size * 5}_v1.csv"), index=False)

        exit()

    if args.task == 'eval': # Prompt for GPT-4(V) Evaluation
        tokenizer = AutoTokenizer.from_pretrained('t5-small')
        special_tokens = ['[ImageToken]', '[SectionIndex]', '[SectionTitle]', '[SectionText]', '[PageURL]', '[PageTitle]', '[ImageCaption]']
        token_dict = {'additional_special_tokens': special_tokens}
        num_added_tokens = tokenizer.add_special_tokens(token_dict)

        split = 'test'
        csv_path = os.path.join(PREP_DIR, f"{split}_image_dict_{args.subset_size}_filtered.csv")
        csv_path = os.path.join(PREP_DIR, f"{split}_image_dict_ccic_{args.subset_size}.csv")
        df = pd.read_csv(csv_path)
        highlight_save_path = os.path.join(PREP_DIR, 'selected_highlights', f"sample_{5}_highlight_{1}_v2.json")
        highlight_samples = load_json(highlight_save_path)
        highlight_samples = {int(key): value for key, value in highlight_samples.items()}
        prompt_dict = {}
        if args.compare_with_ref: gpt_prompt_dict = {}

        image_description_path = os.path.join(PREP_DIR, 'images_1000_response.json')
        image_descriptions = load_json(image_description_path)

        model_type = 'llama'

        if model_type == 'pc':  

            model_response_path = os.path.join(CKPT_DIR, 'CIC', 'experiments', 
                                            '231009-162123-pid311299-full_word_prefix-foraneous',
                                           'checkpoint-2760000', 'ccic_outputs_5002.json')   
        if model_type == 'rc':

            model_response_path = os.path.join(CKPT_DIR, 'CIC', 'experiments', 
                                           '231113-162802-pid1897475-weight_predictor-nonmystic', 
                                           'checkpoint-660000', 'ccic_outputs_5002.json')  
        if model_type == 'ext':
            model_response_path = os.path.join(CKPT_DIR, 'CIC', 'experiments', 
                                           '231217-185937-pid359535-3m_steps-unsubjectedness', 
                                           'checkpoint-2760000', 'ccic_outputs_5002.json')
        if model_type == 'tune':
            model_response_path = os.path.join(CKPT_DIR, 'CIC', 'experiments', 
                                           '240204-150827-pid612127-test-suitcases', 
                                           'checkpoint-1820000', 'ccic_outputs_5002.json') 
        if model_type == 'llava':
            model_response_path = os.path.join(CKPT_DIR, 'CIC', 'experiments', 
                                           'zeroshot-llava', 'llava-hf', 'llava-1.5-7b-hf',
                                           'ccic_outputs.json') 
            
        if model_type == 'llama':
            model_response_path = os.path.join(CKPT_DIR, 'CIC', 'experiments', 'zeroshot-llama', 'meta-llama','Llama-2-7b-chat-hf', 'ccic_outputs_50_15:22.json')

        if model_type == 'GPT':
            model_response_path = os.path.join(os.path.join(PREP_DIR, 'prompts', 'ccic_prompt_grit_5000_v2_response.json') )
        
        model_caps = load_json(model_response_path)



        new_rows = []
        for index, row in tqdm(df.iterrows(), total = args.subset_size):
            sample_idx, img_path, img_idx, sec_idx , original_index, _ = df.iloc[index]
            
            if f'{original_index}_0' not in model_caps: continue


            highlight_segments_list = highlight_samples[original_index]
            context, caption = get_context_and_gt_caption(index, df, split)
            # context = constrain_string_length(context, tokenizer)
            context = constrain_string_length_binary(context, tokenizer)

            if args.grit_as_ref:
                caption = image_descriptions[str(original_index)]
            # exit()
            if args.image_description == 'attribution':
                image_description = get_image_description(index, df, split)
            else:
                image_description = image_descriptions[str(original_index)]

            for i, highlight_segments in enumerate(highlight_segments_list):
                if args.single_sample and i != 0:
                    continue 
                highlight_string = ""
                for highlight in highlight_segments:
                    highlight_string = "".join([highlight_string, highlight, '\n'])
                key = f'{original_index}_{i}'
                if args.compare_with_ref:
                    if original_index % 2 == 0:
                        model_cap_1 = caption
                        model_cap_2 = model_caps[key]

                    else:
                        model_cap_2 = caption
                        model_cap_1 = model_caps[key]

                    model_prompt = get_eval_prompt(prompt, context, image_description, highlight_string, model_cap_1, model_cap_2)

                    if args.add_image:
                        # base64_image = encode_image(img_path)
                        base64_image = encode_and_resize_image(img_path)

                        prompt_dict[key] = [
                            {
                                "type": "text",
                                "text": model_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]

                    else:
                        prompt_dict[key] = model_prompt

                else:
                    raise NotImplementedError

        import re
        match = re.search(r'eval_(\w+).txt', args.prompt_fp)

        if match:
            eval_type = match.group(1)
        else:
            eval_type = args.prompt_fp[-7:-4]
            print("No match found")
        if args.compare_with_ref:
            model_cap_save_path = os.path.join(PREP_DIR,'prompts','new', f"{split}_{args.task}_{eval_type}_{model_type}_prompt_{args.subset_size}.json")
            print(model_cap_save_path)
            save_json(model_cap_save_path, prompt_dict)

        else:
            raise NotImplementedError


        exit()

    # Prompt for GPT-4 to select highlight candidates from the context for a subset. This also select a subset where context string length is larger than 50
    if args.task == "highlight":
        split = 'test'
        csv_path = os.path.join(PREP_DIR, f"{split}_image_dict_v7_2000.csv")
        df = pd.read_csv(csv_path)
        duplicate_count = df['original_index'].duplicated(keep=False).sum()
        print(duplicate_count)
        filtered_df = df[df.apply(sufficient_len, axis=1)]
        print(len(filtered_df))
        duplicate_count = df['original_index'].duplicated(keep=False).sum()
        print(duplicate_count)
        # exit()
        # Sample from filtered dataframe
        seed = 1140
        sampled_df = filtered_df.sample(n=args.subset_size, replace=False, random_state=seed) 

        sampled_df = sampled_df.drop_duplicates()


        print(sampled_df)
        duplicate_count = sampled_df['original_index'].duplicated(keep=False).sum()
        # print(duplicate_count)
        csv_save_path = os.path.join(PREP_DIR, f"{split}_image_dict_v7_{len(sampled_df)}.csv")
        sampled_df.to_csv(csv_save_path, index=False)
        # exit()
        tokenizer = AutoTokenizer.from_pretrained('t5-small')
        special_tokens = ['[ImageToken]', '[SectionIndex]', '[SectionTitle]', '[SectionText]', '[PageURL]', '[PageTitle]', '[ImageCaption]']
        token_dict = {'additional_special_tokens': special_tokens}
        num_added_tokens = tokenizer.add_special_tokens(token_dict)
        json_dict = {}
        tmp_df = sampled_df.drop('original_index', axis=1)
        for index, row in tqdm(sampled_df.iterrows(), total=sampled_df.shape[0]):
            # original_index = row['original_index']
            sample_idx, img_path, img_idx, sec_idx, original_index = row
            context, caption = get_context_and_gt_caption(index, tmp_df, split, add_special_token= False, add_image_caption = False)
            context = constrain_string_length(context, tokenizer)
            cur_prompt = get_highlight_prompt(prompt, context)
            json_dict[original_index] = cur_prompt

        save_path = os.path.join(PREP_DIR, f"{split}_{args.task}_prompt_v7_{len(sampled_df)}.json")
        
        # dict_v1 = load_json(save_path)
        print(len(json_dict))

        save_json(save_path, json_dict)


        exit()

    # Prompt for GPT-4 to select highlight candidates from the context for the full testset, here we use sample index to avoid querying the same article section
    if args.task == "highlight_fullset":
        split = 'test'
        csv_path = os.path.join(PREP_DIR, f"{split}_image_dict_ccic_fullset.csv")
        df = pd.read_csv(csv_path)
        df['original_index'] = df.index

        df = df[df['use_in_ccic'] == 'True']
        df = df.reset_index(drop=True)

        tokenizer = AutoTokenizer.from_pretrained('t5-small')
        special_tokens = ['[ImageToken]', '[SectionIndex]', '[SectionTitle]', '[SectionText]', '[PageURL]', '[PageTitle]', '[ImageCaption]']
        token_dict = {'additional_special_tokens': special_tokens}
        num_added_tokens = tokenizer.add_special_tokens(token_dict)
        json_dict = {}
        tmp_df = df.drop('original_index', axis=1)
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            sample_idx, _, _, sec_idx, inclusion, original_index = row
            if sample_idx in json_dict:
                continue
            context, _ = get_context_and_gt_caption(index, tmp_df, split, add_special_token= False, add_image_caption = False)
            context = constrain_string_length(context, tokenizer)
            cur_prompt = get_highlight_prompt(prompt, context)
            json_dict[sample_idx] = cur_prompt

        save_path = os.path.join(PREP_DIR, 'prompts', f"{split}_{args.task}_prompt_fullset.json")
        
        # dict_v1 = load_json(save_path)
        print(len(json_dict))

        save_json(save_path, json_dict)


        exit()

