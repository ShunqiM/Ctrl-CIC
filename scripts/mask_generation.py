# Generate Training Token Relevance Scores

import os, sys
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2Model, AutoModel, T5EncoderModel, GPT2Tokenizer, LongT5EncoderModel, RobertaModel, BertModel
from src.dataset import Web2MTextDataset
from torch.utils.data import DataLoader
from nltk.corpus import stopwords
import pathlib

from utils.utils import colorize, normalize_scores
from utils.utils_io import get_shard_index
from starter.env_getter import get_env
PREP_DIR = get_env('PREP')

def cosine_similarity(a, b, mask = None):
    norm_a = torch.norm(a, dim=-1, keepdim=True)
    norm_b = torch.norm(b, dim=-1, keepdim=True)

    dot_product = torch.sum(a * b.unsqueeze(1), dim=-1)

    cosine_sim = dot_product / (norm_a.squeeze(-1) * norm_b)

    return cosine_sim

def masked_softmax(scores, mask):
    masked_scores = scores + (1 - mask) * (-1e9)  # Set masked elements to -1e9 (or any large negative value)

    softmax_scores = torch.softmax(masked_scores, dim=-1)

    return softmax_scores

def get_aggregated_embedding(token_embeddings, attention_mask, mode = 'mean'):
    if mode == 'mean':
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = attention_mask.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        sum_mask.unsqueeze_(-1)
        pooled = torch.div(sum_embeddings, sum_mask)
        return pooled
    if mode == 'max':
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        masked_embeddings = token_embeddings * input_mask_expanded
        max_embeddings = torch.max(masked_embeddings, dim=1)
        return max_embeddings.values
    if mode == 'last':

        index = attention_mask.min(axis = 1).indices
        index = [x-1 for x in index]
        last_tokens = token_embeddings[:, index, :]
        last_tokens = token_embeddings[torch.arange(attention_mask.shape[0]), index]
        return last_tokens
    if mode == 'first': # Aiming for models like Bert where the first token correspond to CLS:
        return token_embeddings[:, 0, :]
    
# Something strange with gpt2-tokenizer-fast class. A similar problem here: https://github.com/huggingface/transformers/issues/10258
def convert_ids_to_tokens_gpt2(token_ids, tokenizer):

    from transformers import GPT2Tokenizer
    gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    out = []
    for id in token_ids:
        text = tokenizer._convert_id_to_token(id)
        text = bytearray([gpt_tokenizer.byte_decoder[c] for c in text]).decode("utf-8", errors=gpt_tokenizer.errors)
        out.append(text)
    return out

def get_filtered_weight(tokens, scores, stops):
    special_space = (b'\xe2\x96\x81').decode('utf-8')

    filtered_mask = np.zeros((len(tokens)))

    for i, t in enumerate(tokens):
        if t.replace(special_space, '').lower() not in stops:

            filtered_mask[i] = 1
            
    return filtered_mask

if __name__ == '__main__':


    model_name = 't5-large'

    aggregation_mode = 'mean' # mean, max, last (last for decoder only models)

    add_cap = False
    save = False # whether save the masks locally or just for visualization
    split = 'test'
    csv_name = split + '_image_dict_v7.csv'
    max_src_len = 512
    max_tgt_len = 128
    stops = set(stopwords.words('english'))
    stops.add('')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'
    # model = AutoModel.from_pretrained(model_name)
    # print(model.encoder)
    if 'gpt' in model_name:
        model = GPT2Model.from_pretrained(model_name)
    elif 'long-t5' in model_name:
        model = LongT5EncoderModel.from_pretrained(model_name)
    elif 't5' in model_name:
        model = T5EncoderModel.from_pretrained(model_name)
    elif 'roberta' in model_name:
        model = RobertaModel.from_pretrained(model_name)
    elif 'bert' in model_name:
        model = BertModel.from_pretrained(model_name)


    dataset = Web2MTextDataset(tokenizer, csv_name, split, max_src_len, max_tgt_len, add_cap=add_cap, all_inputs = False)
    train_dataloader = DataLoader(dataset, shuffle=False, batch_size=128, num_workers=16)
    # train_dataloader = DataLoader(dataset, shuffle=False, batch_size=4)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(os.path.join(PREP_DIR, 'vis', f'weight_visualization_{model_name}')):
        os.mkdir(os.path.join(PREP_DIR, 'vis', f'weight_visualization_{model_name}'))
    # model2 = CLIPModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    # print(model)
    # exit()
    if save:
        mask_path = os.path.join(PREP_DIR, 'input_masks', f'{model_name}_{aggregation_mode}_{str(max_src_len)}', split)
        if not os.path.exists(mask_path):
            os.makedirs(mask_path)

    # When do it with full training set, continue on the first batches would do
    for idx, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        input_ids = batch.pop("input_ids").to(device)
        attention_mask = batch.pop("attention_mask").to(device)
        target_ids = batch.pop("target_ids").to(device)
        target_mask = batch.pop("target_mask").to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask = attention_mask)
            context_hidden_states = outputs.last_hidden_state
            tgt_outputs = model(target_ids, attention_mask = target_mask)
            target_hidden_states = tgt_outputs.last_hidden_state

            target_feature = get_aggregated_embedding(target_hidden_states, target_mask, aggregation_mode)

            word_scores = cosine_similarity(context_hidden_states, target_feature)

            for i in range(len(input_ids)):
                tokens_ids = torch.masked_select(input_ids[i], attention_mask[i].bool()).cpu()
                

                if 'gpt' in model_name or 'roberta' in model_name:
                    tokens = convert_ids_to_tokens_gpt2(tokens_ids, tokenizer)
                else:
                    tokens = tokenizer.convert_ids_to_tokens(tokens_ids)
                score = torch.masked_select(word_scores[i], attention_mask[i].bool()).cpu()

                if save:
                    # It might be better to save to un normalised and unfiltered scores here, and choose whether to filter in dataset
                    data_index = idx*train_dataloader.batch_size + i


                    shard_index = get_shard_index(data_index, 1000)
                    save_dir = os.path.join(PREP_DIR, 'input_masks', f'{model_name}_{aggregation_mode}_{str(max_src_len)}', split, str(shard_index))
                    save_path= pathlib.Path(save_dir)
                    save_path.mkdir(exist_ok=True, parents=True)
                    save_path = os.path.join(save_dir, str(data_index) + '.pt')
                    # remove the score of EOS token
                    score = score[:-1]
                    torch.save(score, save_path)
                    
                else:
                    # some codes for visualiztion during development... Feel free to ignore them
                    print('unnormalized score', score)
                    score = normalize_scores(score, 0, 1) # list
                    score = np.array(score)
                    print('normalized score', score)

                    filtered_mask = get_filtered_weight(tokens, score, stops)
                    score = filtered_mask * score
                    print('filtered score', score)
                    s = colorize(tokens, score)
                    save_path = os.path.join(PREP_DIR, 'vis', f'weight_visualization_{model_name}', str(idx*train_dataloader.batch_size + i) + f'_{str(add_cap)}_{aggregation_mode}.html')
                    
                    print(save_path)
                    with open(save_path, 'w') as f:
                        f.write(s)
                    # exit()
            if save:
                continue
            exit()
            word_scores_masked = word_scores * attention_mask
            word_scores_softmax = masked_softmax(word_scores, attention_mask)

            print(word_scores_softmax, word_scores_softmax.shape)
            cut_off = torch.masked_select(input_ids[0], attention_mask[0].bool()).cpu()
            print(cut_off, cut_off.shape)
            score_cut_off = torch.masked_select(word_scores_softmax[0], attention_mask[0].bool()).cpu()
            score_cut_off = score_cut_off * 10
            print(torch.min(word_scores_softmax, dim=1), torch.max(word_scores_softmax, dim=1))
            tokens = tokenizer.convert_ids_to_tokens(cut_off)
            s = colorize(tokens, score_cut_off)
            with open('./tmp/colorize.html', 'w') as f:
                f.write(s)
            exit()

        exit()


