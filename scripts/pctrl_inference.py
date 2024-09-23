import os
import torch
import pandas as pd
from PIL import Image
from transformers import AutoTokenizer, LongT5ForConditionalGeneration
from starter.env_getter import get_env
from src.dataset import Web2MFTDataset, Web2MMaskPrefixDataset, token2word
from termcolor import colored, cprint
import random
import numpy as np
import tkinter as tk
from utils.utils import compute_metrics

PREP_DIR = get_env('PREP')
CKPT_DIR = get_env('CKPT')

# NOTE select all occurences is redudant code with unexpected behaviours here
def highlight_selection(paragraph_text, highlighted_tokens, selected_all_occurrences=True):
    selected_text = paragraph_text.selection_get()
    selected_tokens = split_into_words(selected_text)
    
    if selected_all_occurrences:
        highlighted_tokens.update(selected_tokens)
    else:
        selected_start_index = get_selected_start_index(paragraph_text)
        highlighted_tokens.update({(selected_token, selected_start_index + i) for i, selected_token in enumerate(selected_tokens)})
            
    return update_highlighted_words(highlighted_tokens, selected_all_occurrences)


def split_into_words(text):
    words = []
    word = ""
    for char in text:
        if char.isalnum():
            word += char
        else:
            if word:
                words.append(word)
                word = ""
    if word:
        words.append(word)
    return words

def get_selected_start_index(paragraph_text):
    start_index, _ = paragraph_text.tag_ranges(tk.SEL)
    start_line, start_col = map(int, str(start_index).split('.'))
    
    accumulated_index = sum(len(paragraph_text.get(f"{i}.0", f"{i}.end")) + 1 for i in range(1, start_line))
    accumulated_index += start_col
    
    return accumulated_index

def update_highlighted_words(highlighted_tokens, selected_all_occurrences):
    if selected_all_occurrences:
        return "Highlighted words: {}".format(", ".join(highlighted_tokens))
    else:
        return "Highlighted words: {}".format(", ".join([token for token, _ in highlighted_tokens]))
    
def show_app(model_input, select_all_occur = False):

    # Create the main window
    root = tk.Tk()
    root.title("Highlight Text")

    # Create and place UI elements
    paragraph_label = tk.Label(root, text="Enter a paragraph:")
    paragraph_label.pack()

    paragraph_text = tk.Text(root, height=10, width=100)
    paragraph_text.insert("1.0", model_input)
    paragraph_text.pack()

    highlighted_tokens = set()

    def on_highlight():
        updated_text = highlight_selection(paragraph_text, highlighted_tokens, select_all_occur)
        highlighted_words_label.config(text=updated_text)

    highlight_button = tk.Button(root, text="Highlight Selection", command=on_highlight)
    highlight_button.pack()

    def on_exit():
        root.quit()  # Stops the main loop

    exit_button = tk.Button(root, text="Exit and Return Words", command=on_exit)
    exit_button.pack()

    highlighted_words_label = tk.Label(root, text="Highlighted words: ")
    highlighted_words_label.pack()

    root.mainloop()
    
    if select_all_occur:
        return [token for token, _ in highlighted_tokens]
    else:
        return [token for token in highlighted_tokens]



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default=None)
    parser.add_argument('--from_dataset', type=bool, default=True)
    parser.add_argument('--csv_name', type=str, default = 'test_image_dict_v7.csv') 
    parser.add_argument('--index', type=int, default=0) 
    parser.add_argument('--max_len', type=int, default=128) 
    parser.add_argument('--type_input', type=bool, default=False) # Will I Implement in-program text input later?
    # parser.add_argument('--selected_all_occurrences', type=bool, default=False)
    parser.add_argument('--run_id', type=str, default = "231001-232704-pid113602-word_prefix-corymbed") 
    parser.add_argument('--ckpt_index', type=int, default = 660000) 

    args = parser.parse_args()

    run_id = args.run_id
    ckpt_index = 'checkpoint-' + str(args.ckpt_index)
    MODEL_PATH = os.path.join(CKPT_DIR, 'CIC', 'experiments', run_id, f'checkpoint-{str(args.ckpt_index)}')

    tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base")
    special_tokens = ['[ImageToken]', '[SectionIndex]', '[SectionTitle]', '[SectionText]', '[PageURL]', '[PageTitle]', '[ImageCaption]']
    extra_special_tokens = ['<MSK>', '<SEP>', '<CPT>']
    special_tokens.extend(extra_special_tokens)
    token_dict = {'additional_special_tokens': special_tokens}
    num_added_tokens = tokenizer.add_special_tokens(token_dict)
    print(MODEL_PATH)
    model = LongT5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    # exit()
    model = model.to('cuda')

    if args.from_dataset:
        split = args.csv_name.split('_')[0]
        if split not in ['train', 'test', 'val']:
            print('Cannot inference the split of dataset, set to train, might cause data loading trouble')
        dataset = Web2MMaskPrefixDataset(csv_name=args.csv_name, max_src_len=511, max_tgt_len=args.max_len, split=split,
                            feature_matrix_name = f"{split}_image_feature_matrix.pt", mask_dir = 't5-large_mean_512',
                            normalise_score= 'fix_scale', tokenizer=tokenizer, return_prefix = True, mask_threshold=0.65, token2word = True)
        inputs = dataset.__getitem__(args.index, return_raw_text= True)
    else:
        raise NotImplementedError
    # """
    model_input = inputs['raw_texts']
    # Create the main window
    highlight_words = show_app(model_input)
    print(highlight_words)
    highlight_words = sorted(highlight_words, key=lambda x: x[1])
    highlight_words = [word for word, _ in highlight_words]
    print(highlight_words)
    if len(highlight_words) == 0:
            mask_prefix = "<MSK> <CPT>"
    else:
        mask_prefix = "<MSK> " + highlight_words[0]
        for wd in highlight_words[1:]:
            mask_prefix = " ".join([mask_prefix, "<SEP>", wd])
        mask_prefix = " ".join([mask_prefix, "<CPT>"])
    
    print(mask_prefix)
    mask_prefix = "<pad> " + mask_prefix
    decoder_input_ids = tokenizer(mask_prefix, return_tensors="pt").input_ids
    # remove eos
    decoder_input_ids = decoder_input_ids[:, :-1].to("cuda")
    print(decoder_input_ids)
    

    inputs = dataset.__getitem__(args.index)

        
    if args.image_path is not None: 
        print('Raw Image Input is not tested yet')
        # raise NotImplementedError

        if os.path.exists(args.image_path): image = Image.open(args.image_path)
        else: raise "Image does not exist"
        image.show()
        model_name = "openai/clip-vit-large-patch14" # [257, 1024] 1029kb /image, *100k = 103G,  * 2M = 2T
        from transformers.image_transforms import convert_to_rgb
        from transformers import CLIPModel, CLIPProcessor
        from utils.utils import pad_to_square_np
        from transformers.image_utils import to_numpy_array
        img_model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
        image = convert_to_rgb(image)
        image = to_numpy_array(convert_to_rgb(image))
        image = pad_to_square_np(image)
        image = processor(images=image, return_tensors="pt")['pixel_values'][0]
        pixel_values = processor(images=image, return_tensors="pt")['pixel_values']
        with torch.no_grad():
            outputs = img_model.vision_model(pixel_values=pixel_values, return_dict = True)
            pooler_output = outputs['pooler_output']
            image_features = img_model.visual_projectarget_texttion(pooler_output)

        image_features = image_features.detach().to('cpu')[0]
        inputs['image_feature'] = image_features
    
    target_ids = inputs.pop('labels') 
    target_text = tokenizer.decode(target_ids, skip_special_tokens=False)
    assert inputs['input_ids'] != None and inputs['image_feature'] != None, 'Missing Input modalities' 


    model_input = tokenizer.decode(inputs['input_ids'], skip_special_tokens=False)
    end_token = '</s>'
    end_token_index = model_input.find(end_token)

    # Remove content after the end token
    if end_token_index != -1:
        model_input = model_input[:end_token_index + len(end_token)]
    else:
        model_input = model_input
    model_input_p = colored('Model Input: ' + model_input, 'light_green')
    cprint(model_input_p)
    for k,v in inputs.items():
        inputs[k] = v.unsqueeze(0).to("cuda")
    seed = 1140
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model.eval()


    with torch.no_grad():
        print('generating')
        image_embs = inputs['image_feature']
        image_embs = image_embs.unsqueeze(1)
        input_embedding_layer = model.get_input_embeddings()
        input_embeddings = input_embedding_layer(inputs['input_ids'])


        combined_embs = torch.cat((input_embeddings[:, :1, :], image_embs, input_embeddings[:, 1:, :]), dim = 1)
        print(combined_embs.shape)
        new_token_mask = torch.ones((combined_embs.shape[0],1), dtype=torch.int64).to(inputs['attention_mask'].device)
        new_attention_mask = torch.cat((new_token_mask, inputs['attention_mask']), dim = 1).to('cuda')

        output = model.generate(inputs_embeds=combined_embs, decoder_input_ids=decoder_input_ids, max_new_tokens=args.max_len, attention_mask = new_attention_mask).detach()
        decode = tokenizer.decode(output[0], skip_special_tokens=False)
       
        predicted_p = colored('Generated: ' + decode, 'light_yellow')        
        cprint(predicted_p)
        if target_text is not None:
            target_p = colored("CIC Caption: " + target_text, 'light_red')
            cprint(target_p)
