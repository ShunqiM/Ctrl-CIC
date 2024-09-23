import os
import torch
import pandas as pd
from PIL import Image, ImageTk
from transformers import AutoTokenizer
from starter.env_getter import get_env
from src.dataset import Web2MFTDataset, CCICInferenceDataset
from termcolor import colored, cprint
import random
import numpy as np
import tkinter as tk
from tkinter import filedialog
from utils.utils import compute_metrics, pad_to_square_np
from src.model.CICModel import ReWeightLongT5, LongT5forTokenRegression, TwoStageLongT5
from transformers.image_transforms import convert_to_rgb
from transformers import CLIPModel, CLIPProcessor
from transformers.image_utils import to_numpy_array

PREP_DIR = get_env('PREP')
CKPT_DIR = get_env('CKPT')

# Global variables to be used across functions
highlighted_tokens = set()
mask = []
inputs = None  # Will be initialized later
model_input = ""
output_text_widget = None  # Will be initialized in the GUI setup
image_features = None
image_label = None
tokenizer = None  # Will be initialized in main
model = None  # Will be initialized in main
predictor = None  # Will be initialized in main
dataset = None  # Will be initialized in main
args = None  # Will be initialized in main
img_model = None  # Will be initialized in main
processor = None  # Will be initialized in main
selected_all_occurrences = False  # Will be initialized in main
default_image_path = None

def highlight_selection():
    selected_text = paragraph_text.selection_get()
    selected_tokens = tokenizer.tokenize(selected_text)
    if selected_all_occurrences:
        highlighted_tokens.update(selected_tokens)
    else:
        selected_positions = []
        selected_start_index = get_selected_start_index()
        selected_token_index = convert_char_to_token_index(selected_start_index, paragraph_text.get("1.0", tk.SEL_FIRST))
        if selected_token_index < 512:
            selected_positions.extend([selected_token_index + i for i in range(len(selected_tokens))])
            highlighted_tokens.update({(selected_token, position) for selected_token, position in zip(selected_tokens, selected_positions)})
    update_highlighted_words(selected_all_occurrences)

def selected_tokens_with_positions(tokens):
    selected_start_index = get_selected_start_index()
    return [(token, selected_start_index + i) for i, token in enumerate(tokens)]

def convert_char_to_token_index(char_index, text):
    tokens = tokenizer.tokenize(text)
    return len(tokens)

def get_selected_start_index():
    start_index = paragraph_text.index(tk.SEL_FIRST)
    row_num, start_col = map(int, str(start_index).split('.')) 
    return start_col

def update_highlighted_words(selected_all_occurrences):
    if selected_all_occurrences:
        highlighted_words_label.config(text="Highlighted words: {}".format(", ".join(highlighted_tokens)))
    else:
        highlighted_words_label.config(text="Highlighted words: {}".format(", ".join([token for token, _ in highlighted_tokens])))

def generate_highlight_mask(paragraph, highlighted_tokens, tokenizer):
    tokens = tokenizer.tokenize(paragraph)
    if selected_all_occurrences:
        highlight_mask = [1 if token in highlighted_tokens else 0 for token in tokens]
    else:
        positions = [position for token, position in highlighted_tokens]
        highlight_mask = [1 if position in positions else 0 for position in range(len(tokens))]
    return highlight_mask, tokens

def show_highlight_mask():
    global mask
    mask, tokens = generate_highlight_mask(paragraph_text.get("1.0", "end-1c"), highlighted_tokens, tokenizer)
    print("Highlight mask:", mask)
    print("Tokens", tokens)
    generate_output()  # Call the function to generate output

def generate_output():
    global mask, model_input, inputs, image_features
    # Get the model input from the GUI
    model_input = paragraph_text.get("1.0", "end-1c")
    print("Mask:", mask)
    
    # Tokenize the model input
    token_ids = tokenizer.encode(model_input)
    if len(mask) != len(token_ids) - 1:
        print('Unmatched length, double check', len(mask), len(token_ids) - 1)
        print(tokenizer.convert_ids_to_tokens(token_ids))
        return

    if len(mask) > 511:
        mask = mask[:511]
    new_mask = torch.tensor(mask)
    new_mask = torch.where(new_mask == 0, torch.tensor(0), torch.tensor(0.1))
    
    # Prepare initial inputs
    inputs = dataset.__getitem__(
        args.index,
        override_mask=new_mask,
        return_raw_text=False
    )
    
    target_ids = inputs.pop('labels')
    target_text = tokenizer.decode(target_ids, skip_special_tokens=True)
    
    # Ensure inputs have necessary keys
    assert inputs['input_ids'] is not None and inputs['image_feature'] is not None, 'Missing input modalities'
    
    # Display the model input
    model_input_p = colored('Model Input: ' + model_input, 'light_green')
    cprint(model_input_p)
    
    for k, v in inputs.items():
        inputs[k] = v.unsqueeze(0).to("cuda")
    
    with torch.no_grad():
        print('Predicting weights with predictor')
        image_embs = inputs['image_feature']
        image_embs = image_embs.unsqueeze(1)
        input_embedding_layer = model.get_input_embeddings()
        input_embeddings = input_embedding_layer(inputs['input_ids'])
        combined_embs = torch.cat((input_embeddings[:, :1, :], image_embs, input_embeddings[:, 1:, :]), dim=1)
        new_token_mask = torch.ones((combined_embs.shape[0],1), dtype=torch.int64).to(inputs['attention_mask'].device)
        new_attention_mask = torch.cat((new_token_mask, inputs['attention_mask']), dim=1).to('cuda')

        zero_mask = torch.zeros((combined_embs.shape[0],1), dtype=torch.int64).to(inputs['attention_mask'].device)
        token_weight_mask = torch.cat((zero_mask.clone(), inputs['token_weight_mask']), dim=1)

        weights = predictor(inputs_embeds=combined_embs, attention_mask=new_attention_mask, token_weight_mask=token_weight_mask)
        tensor = weights['scores'][0].cpu()
        print(tensor)
        non_zero_indices = torch.nonzero(tensor, as_tuple=True)[0]

        # Get the range of non-zero values
        if non_zero_indices.nelement() != 0:
            start = non_zero_indices[0]
            end = non_zero_indices[-1] + 1  # Include the last non-zero index
            middle_floats = tensor[start:end]
        else:
            middle_floats = torch.tensor([])
        if len(new_mask) > len(middle_floats):
            # new_mask = new_mask[:len(middle_floats)]
            padding_needed = max(0, len(new_mask) - len(middle_floats))
            middle_floats = torch.nn.functional.pad(middle_floats, (0, padding_needed))
        new_mask = new_mask + middle_floats
    
    # Update inputs with new mask
    inputs = dataset.__getitem__(
        args.index,
        override_mask=new_mask,
        return_raw_text=False
    )

    # If a custom image has been loaded, overwrite the image feature
    if image_features is not None:
        inputs['image_feature'] = image_features.to('cuda')

    for k, v in inputs.items():
        inputs[k] = v.unsqueeze(0).to("cuda")
    
    seed = 1140
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model.eval()
    with torch.no_grad():
        print('Generating output')
        image_embs = inputs['image_feature']
        token_weights = inputs['token_weights']
        image_embs = image_embs.unsqueeze(1)
        input_embeddings = model.get_input_embeddings()(inputs['input_ids'])
        combined_embs = torch.cat((input_embeddings[:, :1, :], image_embs, input_embeddings[:, 1:, :]), dim=1)
        new_token_mask = torch.ones((combined_embs.shape[0],1), dtype=torch.int64).to(inputs['attention_mask'].device)
        new_attention_mask = torch.cat((new_token_mask, inputs['attention_mask']), dim=1).to('cuda')
        token_weights = torch.cat((new_token_mask.clone(), token_weights), dim=1)
        output = model.generate(inputs_embeds=combined_embs, max_new_tokens=args.max_len, attention_mask=new_attention_mask, token_weights=token_weights).detach()
        decode = tokenizer.decode(output[0], skip_special_tokens=True)
        predicted_p = colored('Generated: ' + decode, 'light_yellow')
        cprint(predicted_p)

        # Display the output in the GUI
        output_text_widget.delete("1.0", tk.END)
        output_text_widget.insert(tk.END, decode)

        if target_text is not None:
            target_p = colored("CIC Caption: " + target_text, 'light_red')
            cprint(target_p)

def load_default_image():
    global image_features, image_label
    # Load the image from the dataset
    image_path = default_image_path  # We'll set this variable when we initialize the dataset
    if image_path and os.path.exists(image_path):
        # Load and process the image
        image = Image.open(image_path)
        image.thumbnail((256, 256))  # Resize for display purposes

        # Update the image in the image_label
        img_display = ImageTk.PhotoImage(image)
        image_label.config(image=img_display)
        image_label.image = img_display  # Keep a reference

        # Extract image features
        image_features = extract_image_features(image_path)
        print("Default image features extracted.")
    else:
        print("Default image not found.")

def load_image():
    global image_features, image_label
    # Open file dialog to select an image
    image_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=[("Image Files", ("*.png", "*.jpg", "*.jpeg", "*.bmp"))]
    )
    if image_path:
        # Load and process the image
        image = Image.open(image_path)
        image.thumbnail((256, 256))  # Resize for display purposes

        # Update the image in the image_label
        img_display = ImageTk.PhotoImage(image)
        image_label.config(image=img_display)
        image_label.image = img_display  # Keep a reference

        # Extract image features
        image_features = extract_image_features(image_path)
        print("Image features extracted.")
    else:
        print("No image selected.")

def extract_image_features(image_path):
    # Load the image
    image = Image.open(image_path)
    image = convert_to_rgb(image)
    image = to_numpy_array(image)
    image = pad_to_square_np(image)

    # Preprocess the image
    image_input = processor(images=image, return_tensors="pt")['pixel_values'].to('cuda')

    # Extract features
    with torch.no_grad():
        outputs = img_model.vision_model(pixel_values=image_input, return_dict=True)
        pooler_output = outputs['pooler_output']
        image_features = img_model.visual_projection(pooler_output)

    image_features = image_features.detach().cpu()[0]
    return image_features

def clear_mask():
    global highlighted_tokens, mask
    highlighted_tokens.clear()
    mask.clear()
    # Update the highlighted words label
    highlighted_words_label.config(text="Highlighted words: ")
    print("Mask and highlighted tokens have been cleared.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default=None)
    parser.add_argument('--from_dataset', type=bool, default=True)
    parser.add_argument('--csv_name', type=str, default='test_image_dict_v7.csv') 
    parser.add_argument('--index', type=int, default=0) 
    parser.add_argument('--max_len', type=int, default=128) 
    parser.add_argument('--type_input', type=bool, default=False)  # Will I Implement in-program text input later?
    parser.add_argument('--selected_all_occurrences', type=bool, default=False)
    parser.add_argument('--run_id', type=str, default="231026-225939-pid693673-reweight_full-nonimmunities") 
    parser.add_argument('--ckpt_index', type=int, default=2930000) 
    parser.add_argument('--predictor_ckpt_index', type=int, default=660000) 
    
    args = parser.parse_args()
    selected_all_occurrences = args.selected_all_occurrences
    run_id = args.run_id
    ckpt_index = 'checkpoint-' + str(args.ckpt_index)
    MODEL_PATH = os.path.join(CKPT_DIR, 'CIC', 'experiments', run_id, f'checkpoint-{args.ckpt_index}')

    # Initialize tokenizer and add special tokens
    tokenizer = AutoTokenizer.from_pretrained("google/long-t5-tglobal-base")
    special_tokens = ['[ImageToken]', '[SectionIndex]', '[SectionTitle]', '[SectionText]', '[PageURL]', '[PageTitle]', '[ImageCaption]']
    tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})

    # Initialize models
    model = ReWeightLongT5.from_pretrained(MODEL_PATH)
    model = model.to('cuda')

    weight_predictor_path = os.path.join(CKPT_DIR, 'CIC', 'experiments', '231113-162802-pid1897475-weight_predictor-nonmystic', f'checkpoint-{args.predictor_ckpt_index}')
    predictor = LongT5forTokenRegression.from_pretrained(weight_predictor_path)
    predictor = predictor.to('cuda')

    # Initialize the image model and processor
    model_name = "openai/clip-vit-large-patch14"
    img_model = CLIPModel.from_pretrained(model_name).to('cuda')
    processor = CLIPProcessor.from_pretrained(model_name)

    if args.from_dataset:
        split = args.csv_name.split('_')[0]
        if split not in ['train', 'test', 'val']:
            print('Cannot infer the split of dataset, set to train, might cause data loading trouble')
            split = 'train'
        dataset = Web2MFTDataset(csv_name=args.csv_name, max_src_len=511, max_tgt_len=args.max_len, split=split,
                                 feature_matrix_name=f"{split}_image_feature_matrix.pt", mask_dir='t5-large_mean_512',
                                 normalise_score='fix_scale', tokenizer=tokenizer, add_input_mask=True, mask_as_labels=True)
        inputs = dataset.__getitem__(args.index, return_raw_text=True)
        default_image_path = inputs.get('image_path', None)
    else:
        raise NotImplementedError

    # Initial model input from dataset
    model_input = inputs['raw_texts']

    # Create the main window
    root = tk.Tk()
    root.title("Highlight Text")

    # Create and place UI elements
    paragraph_label = tk.Label(root, text="Enter a paragraph:")
    paragraph_label.pack()

    paragraph_text = tk.Text(root, height=10, width=100)
    paragraph_text.insert("1.0", model_input)
    paragraph_text.pack()

    # Initialize the image_label and pack it below the paragraph_text
    image_label = tk.Label(root)
    image_label.pack()

    load_default_image()

    highlight_button = tk.Button(root, text="Highlight Selection", command=highlight_selection)
    highlight_button.pack()

    highlighted_words_label = tk.Label(root, text="Highlighted words: ")
    highlighted_words_label.pack()

    generate_mask_button = tk.Button(root, text="Generate Caption", command=show_highlight_mask)
    generate_mask_button.pack()

    # Add Clear Mask button
    clear_mask_button = tk.Button(root, text="Clear Highlight Selection", command=clear_mask)
    clear_mask_button.pack()

    # Add Load Image button
    load_image_button = tk.Button(root, text="Load Image", command=load_image)
    load_image_button.pack()

    # Output display
    output_label = tk.Label(root, text="Generated Output:")
    output_label.pack()

    output_text_widget = tk.Text(root, height=10, width=100)
    output_text_widget.pack()

    root.mainloop()
