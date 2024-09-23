import os
from torch.utils.data import Dataset as TorchDataset

from cli.utils_registry import Registry
from PIL import Image
from transformers import T5Tokenizer, ViTImageProcessor, AutoTokenizer, CLIPProcessor
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoProcessor
from transformers.image_transforms import convert_to_rgb, pad
from utils.utils import pad_to_square, pad_to_square_np
from transformers.image_utils import to_numpy_array
import tensorflow.compat.v1 as tf
import pandas as pd
import torch
import numpy as np
from utils.utils_io import load_pkl, get_shard_index, load_json
from utils.utils import get_feature_path, get_save_string, pad_to_square, normalize_scores, split_into_sentences
from starter.env_getter import get_env
import warnings
import random
import string
from collections import OrderedDict

PREP_DIR = get_env('PREP')
Image.MAX_IMAGE_PIXELS = 933120000
warnings.filterwarnings("ignore", "Corrupt EXIF data", UserWarning)
warnings.filterwarnings("ignore", "Truncated File Read", UserWarning)
warnings.filterwarnings("ignore", "Palette images with Transparency expressed in bytes should be converted to RGBA images", UserWarning)
warnings.filterwarnings("ignore", "Image appears to be a malformed MPO file, it will be interpreted as a base JPEG file", UserWarning)
warnings.filterwarnings("ignore", "Metadata Warning", UserWarning)

MASK_DIR = "input_masks_full"

@Registry.register("DummyDataset")
class DummyDataset(TorchDataset):
    def __init__(self, processor="vit", max_src_len=64, max_tgt_len=32, **kwargs):

        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.src = 'this is source text.'
        self.tgt = 'that is target text.'
        tokenizer_name = kwargs['params']['pretrained_model_name_or_path']
        if 'tokenizer' in kwargs: self.tokenizer = kwargs['tokenizer']
        else: self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if processor == "vit": self.processor = ViTImageProcessor.from_pretrained(kwargs['image_model_name'])
        else: 
            print('Unsupported Processor Type')
            exit()


    def __len__(self):
        return 100

    def __getitem__(self, index):

        # return {'text': self.src, 'label': self.tgt}
        
        tokenized_inputs = self.tokenizer(
            [self.src], max_length=self.max_src_len, padding="max_length", return_tensors="pt", truncation=True
        )
        tokenized_targets = self.tokenizer(
            [self.tgt], max_length=self.max_tgt_len, padding="max_length", return_tensors="pt", truncation=True
        )

        source_ids = tokenized_inputs["input_ids"][0]
        target_ids = tokenized_targets["input_ids"][0]

        src_mask = tokenized_inputs["attention_mask"][0]
        target_mask = tokenized_targets["attention_mask"][0]

        image = Image.new('RGB', (512, 512))
        image = self.processor(images=image, return_tensors="pt")['pixel_values'][0]

        return {"input_ids": source_ids, "attention_mask": src_mask,
                    "labels": target_ids, "pixel_values": image}
        return {"source_ids": source_ids, "source_mask": src_mask,
                    "target_ids": target_ids, "target_mask": target_mask,
                    "source_text": self.src, "target_text": self.tgt}

    @property
    def classname(self):
        return self.__class__.__name__
    
@Registry.register("WikiWeb2MDataset")
class WikiWeb2MDataset(TorchDataset):
    def __init__(self, csv_name, processor="vit", max_src_len=64, max_tgt_len=32, **kwargs):
        self.split = kwargs.pop('split')
        print(f"WikiWeb2MDataset-{self.split} called")
        self.df = pd.read_csv(os.path.join(PREP_DIR, csv_name))
        self.df = self.df.drop(self.df[self.df['exist'] == 0].index)
        self.df = self.df.drop('exist', axis=1)
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.tokenizer = kwargs['tokenizer']
        # self.image_features = torch.zeros((2000000, 1024))
        if processor == "vit": self.processor = ViTImageProcessor.from_pretrained(kwargs['image_model_name'])
        else: 
            print('Unsupported Processor Type')
            exit()

        # self.split = 'val' # test purpose, remove when training data is ready

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):

        # return {'text': self.src, 'label': self.tgt}
        sample_idx, img_path, img_idx, sec_idx = self.df.iloc[index]

        pkl_path = os.path.join(PREP_DIR, 'extracted_texts', self.split, (str(sample_idx) + '.pkl'))
        text_data = load_pkl(pkl_path)

        # ['ImageToken', 'SectionIndex', 'SectionTitle', 'SectionText', 'PageURL', 'PageTitle', 'ImageCaption']
        sec_title, sec_text = text_data['section_dict'][sec_idx]
        page_url = text_data['page_url']
        page_title = text_data['page_title']
        # Image embeddings will be inserted after tokenization in the trainer
        input_text = " ".join(['[ImageToken]', '[SectionIndex]', str(sec_idx), 
                               '[SectionTitle]', sec_title, '[SectionText]', sec_text,
                               ])
        # check for non-target captions
        image_captions = text_data['image_captions'] # dict of dict
        for k, v in image_captions[sec_idx].items():
            if k != img_idx:
                input_text = " ".join([input_text, '[ImageCaption]', v])

        # append page content
        input_text = " ".join([input_text, '[PageURL]', page_url, '[PageTitle]', page_title])

        # append other section content -- confusing variable names...
        for k, (section_title, section_ctx) in text_data['section_dict'].items():
            if k == sec_idx: continue
            input_text = " ".join([input_text, '[SectionIndex]', str(k), '[SectionTitle]', section_title, 
                                   '[SectionText]', section_ctx])
            if k in image_captions:
                for kk, vv in image_captions[k].items(): # kk, vv are image idx and caption ctx
                    input_text = " ".join([input_text, '[ImageCaption]', vv])

        target_text = image_captions[sec_idx][img_idx]


        tokenized_inputs = self.tokenizer(
            input_text, max_length=self.max_src_len, padding="max_length", return_tensors="pt", truncation=True
        )
        tokenized_targets = self.tokenizer(
            target_text, max_length=self.max_tgt_len, padding="max_length", return_tensors="pt", truncation=True
        )

        # Is there any way, or should I put strings as batch input and tokenize them in batch during training?
        source_ids = tokenized_inputs["input_ids"][0]
        target_ids = tokenized_targets["input_ids"][0]

        src_mask = tokenized_inputs["attention_mask"][0]
        target_mask = tokenized_targets["attention_mask"][0]

        image = Image.open(img_path).convert('RGB')
        # print(image.size)
        image = self.processor(images=image, return_tensors="pt")['pixel_values'][0]
        return {"input_ids": source_ids, "attention_mask": src_mask,
                    "labels": target_ids, "pixel_values": image}
    
    @property
    def classname(self):
        return self.__class__.__name__
    
    def __repr__(self):
        return f"{self.classname}-{self.split}"

# Image Feature+Text Dataset
@Registry.register("Web2MFTDataset")
class Web2MFTDataset(TorchDataset):
    def __init__(self, csv_name, feature_matrix_name, split, max_src_len=64, max_tgt_len=32, 
                 add_input_mask = False, mask_dir = None, tokenizer = None, normalise_score = None, 
                 apply_to_attention_mask = False, mask_threshold = 0, mask_as_labels = False, pad_image = False, **kwargs):
        self.split = split
        print(f"Web2MFTDataset-{self.split} called")
        self.df = pd.read_csv(os.path.join(PREP_DIR, csv_name))
        self.df = self.df.drop(self.df[self.df['exist'] == 0].index)
        self.df = self.df.drop('exist', axis=1)
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.tokenizer = tokenizer
        if feature_matrix_name is None:
            self.image_features = torch.zeros((len(self.df), 768))
            print("No Image Feature Matrix is Used, Using Zero Matrix Instead")
        else:
            if pad_image: self.image_features = torch.load(os.path.join(PREP_DIR, "extracted_image_features", feature_matrix_name))
            else: self.image_features = torch.load(os.path.join(PREP_DIR, "extracted_image_features_unpadded", feature_matrix_name))
            print(self.image_features.shape)
        assert len(self.image_features) == len(self.df)
        self.add_input_mask = add_input_mask
        if self.add_input_mask:
            self.mask_dir = mask_dir
            assert os.path.exists(os.path.join(PREP_DIR, 'input_masks', self.mask_dir, split))
            self.normalise_mode = normalise_score
            self.apply_to_attention_mask = apply_to_attention_mask
            self.mask_threshold = mask_threshold
            self.mask_as_labels = mask_as_labels


    def is_train(self):
        return self.split == 'train'

    def __len__(self):
        return len(self.df)
    
    # The other two args are for visualizations only
    def __getitem__(self, index, override_mask = None, return_raw_text = False):

        sample_idx, img_path, img_idx, sec_idx = self.df.iloc[index]

        pkl_path = os.path.join(PREP_DIR, 'extracted_texts', self.split, (str(sample_idx) + '.pkl'))
        text_data = load_pkl(pkl_path)

        # ['ImageToken', 'SectionIndex', 'SectionTitle', 'SectionText', 'PageURL', 'PageTitle', 'ImageCaption']
        sec_title, sec_text = text_data['section_dict'][sec_idx]
        page_title = text_data['page_title']
        image_captions = text_data['image_captions'] # dict of dict
        image_feature = self.image_features[index].clone()
        target_text = image_captions[sec_idx][img_idx]
        tokenized_targets = self.tokenizer(
                target_text, max_length=self.max_tgt_len, padding="max_length", return_tensors="pt", truncation=True
        )
        target_ids = tokenized_targets["input_ids"][0]
        if self.add_input_mask:
            if len(sec_text) > 0:
                input_prefix = " ".join(['[ImageToken]', '[PageTitle]', page_title, 
                                '[SectionTitle]', sec_title, '[SectionText]'])
            else:
                input_prefix = " ".join(['[ImageToken]', '[PageTitle]', page_title, 
                                '[SectionTitle]', sec_title])
            # remove eos token here
            prefix_token_ids = self.tokenizer.encode(input_prefix, return_tensors="pt")[0, : -1]

            sec_token_ids = self.tokenizer.encode(sec_text, return_tensors="pt", max_length=self.max_src_len, truncation = True)[0, : -1]
            prefix_mask = torch.ones(len(prefix_token_ids))
            mask_path = os.path.join(PREP_DIR, MASK_DIR, self.mask_dir, self.split, str(get_shard_index(index, 1000)), str(index) + '.pt')
            sec_mask = torch.load(mask_path)


            if len(sec_mask) >= len(sec_token_ids):
                # print(len(sec_mask), len(sec_token_ids))
                sec_mask = sec_mask[:self.max_src_len]
                # print(sec_mask)
                if self.normalise_mode == 'minmax':
                    sec_mask = normalize_scores(sec_mask, 0, 1)
                    if isinstance(sec_mask, torch.Tensor):
                        pass
                    else:
                        sec_mask = torch.tensor(sec_mask)
                elif self.normalise_mode == None:
                    pass
                elif self.normalise_mode == 'fix_scale':
                    sec_mask = sec_mask/2 + 0.5
                else:
                    raise NotImplementedError
                # print(sec_mask)
                sec_mask = sec_mask.detach()
                sec_mask.requires_grad = False
                # exit()
            else:
                print(f'sec mask shorter than input at index {str(index)}, needs re-process mask with longer length')
                raise NotImplementedError
            
            if override_mask is not None:
                print('original mask', sec_mask, sec_mask.shape)
                assert len(override_mask) == len(sec_mask)
                sec_mask = override_mask

            
            caption_text = "" # Remaining captions in the section
            for k, v in image_captions[sec_idx].items():
                if k != img_idx and len(v) > 0:
                    caption_text = " ".join([caption_text, '[ImageCaption]', v])

            caption_ids = self.tokenizer.encode(caption_text, return_tensors="pt")[0]

            caption_mask = torch.zeros(len(caption_ids))
            # Fill with values to avoid total removal of this information
            if self.normalise_mode == 'minmax':
                caption_mask_val = 0.4
            elif self.normalise_mode == 'fix_scale':
                caption_mask_val = 0.5
            elif self.normalise_mode == None:
                caption_mask_val = 0.2 
            else:
                raise NotImplementedError
            caption_mask = torch.full((len(caption_ids), ), caption_mask_val)
            # change eos token weight to 1 
            caption_mask[-1] = 1

            input_ids = torch.cat([prefix_token_ids, sec_token_ids, caption_ids])
            input_cmask = torch.cat([prefix_mask, sec_mask, caption_mask])
            if self.mask_as_labels or return_raw_text:
                assert self.normalise_mode == 'fix_scale'
                token_weight_mask = torch.cat([torch.zeros(len(prefix_token_ids)), torch.ones(len(sec_token_ids)), torch.zeros(len(caption_ids))])
            if len(input_ids) >= self.max_src_len:

                input_ids = input_ids[:self.max_src_len]
                input_ids[-1] = self.tokenizer.eos_token_id
                input_cmask = input_cmask[:self.max_src_len]
                src_mask = torch.ones(self.max_src_len)
                if self.mask_as_labels or return_raw_text:
                    token_weight_mask = token_weight_mask[:self.max_src_len]

            else:
                src_mask = torch.cat([torch.ones(len(input_ids)), torch.zeros(self.max_src_len - len(input_ids))])
                input_ids = torch.nn.functional.pad(input_ids, (self.tokenizer.pad_token_id, self.max_src_len - len(input_ids)))
                input_cmask = torch.nn.functional.pad(input_cmask, (0, self.max_src_len - len(input_cmask)))
                if self.mask_as_labels or return_raw_text:
                    token_weight_mask = torch.nn.functional.pad(token_weight_mask, (0, self.max_src_len - len(token_weight_mask)))

            if self.mask_as_labels:
                # print(input_cmask)
                out = {"input_ids": input_ids, "attention_mask": src_mask,
                    "labels": target_ids, "image_feature": image_feature,
                    "token_weight_mask": token_weight_mask, 'token_weights': input_cmask}
                if return_raw_text:
                    out['raw_texts'] = sec_text
                    out['image_path'] = img_path
                return out
            if self.apply_to_attention_mask:
                bool_filter = input_cmask >= self.mask_threshold
                input_cmask[bool_filter] = 0
            
            if return_raw_text:
                
                # texts = " ".join([input_prefix, sec_text, caption_text])
                return {"input_ids": input_ids, "attention_mask": src_mask,
                    "labels": target_ids, "image_feature": image_feature,
                    "token_weights": input_cmask, 'raw_texts': sec_text, "token_weight_mask": token_weight_mask,}
            else:
                return {"input_ids": input_ids, "attention_mask": src_mask,
                    "labels": target_ids, "image_feature": image_feature,
                    "token_weights": input_cmask}
            # Input Test could be carried out here to validate whether input generated this way is identical
        else:
            # Image embeddings will be inserted after tokenization in the trainer
            if len(sec_text) > 0:
                input_text = " ".join(['[ImageToken]', '[PageTitle]', page_title, 
                                '[SectionTitle]', sec_title, '[SectionText]', sec_text,
                                ])
            else:
                input_text = " ".join(['[ImageToken]', '[PageTitle]', page_title, 
                                '[SectionTitle]', sec_title
                                ])


            for k, v in image_captions[sec_idx].items():
                if k != img_idx and len(v) > 0:
                    input_text = " ".join([input_text, '[ImageCaption]', v])

            tokenized_inputs = self.tokenizer(
                input_text, max_length=self.max_src_len, padding="max_length", return_tensors="pt", truncation=True
            )
            
            source_ids = tokenized_inputs["input_ids"][0]
            
            src_mask = tokenized_inputs["attention_mask"][0]
            target_mask = tokenized_targets["attention_mask"][0]


            return {"input_ids": source_ids, "attention_mask": src_mask,
                        "labels": target_ids, "image_feature": image_feature}
    
    @property
    def classname(self):
        return self.__class__.__name__
    
    def __repr__(self):
        return f"{self.classname}-{self.split}"


@Registry.register("SectionDataset")
class SectionDataset(TorchDataset): # Prompted Input for zero-shot models -- Adjust Prompt for different LLM/VLMs
    def __init__(self, csv_name, processor, max_src_len=64, max_tgt_len=32, prompt = '', split = 'train', 
                 return_target_text = False, print_input = [False, False], **kwargs):
        self.split = split
        print(f"SectionDataset-{self.split} called")
        self.df = pd.read_csv(os.path.join(PREP_DIR, csv_name))
        self.df = self.df.drop(self.df[self.df['exist'] == 0].index)
        self.df = self.df.drop('exist', axis=1)

        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.prompt = prompt
        self.processor = processor
        self.return_target_text = return_target_text
        self.print_input = print_input

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):

        sample_idx, img_path, img_idx, sec_idx = self.df.iloc[index]
        
        pkl_path = os.path.join(PREP_DIR, 'extracted_texts', self.split, (str(sample_idx) + '.pkl'))
        
        
        text_data = load_pkl(pkl_path)

        sec_title, sec_text = text_data['section_dict'][sec_idx]

        # input_text = 'Generate image caption given the following context: ' +  \
        #                 sec_title + ". " + sec_text + ' Image caption: ' # prompt v2
        
        page_title = text_data['page_title']

        
        input_text = 'we use \"<<<...>>>\" to represent a context. Generate the image caption given the following context <<<' + \
                        page_title + ". " + sec_title + ". " + sec_text + ">>>. This is a photo of " # v3.2


        image_captions = text_data['image_captions'] # dict of dict



        target_text = image_captions[sec_idx][img_idx]


        image = Image.open(img_path).convert('RGB')
        if self.print_input[1]:
            image.show()
        image = self.processor(images=image, return_tensors="pt")['pixel_values'][0]


        if self.print_input[0]:
            print('Input Text: ', input_text)

        tokenized_inputs = self.processor(
            text=input_text, max_length=self.max_src_len, padding="max_length", return_tensors="pt", truncation=True
        )
        tokenized_targets = self.processor(
            text=target_text, max_length=self.max_tgt_len, padding="max_length", return_tensors="pt", truncation=True
        )

        source_ids = tokenized_inputs["input_ids"][0]
        target_ids = tokenized_targets["input_ids"][0]

        src_mask = tokenized_inputs["attention_mask"][0]
        target_mask = tokenized_targets["attention_mask"][0]

        
        if not self.return_target_text:
            return {"input_ids": source_ids, "attention_mask": src_mask,
                    "labels": target_ids, "pixel_values": image, "index": index}
            # return {"input_ids": source_ids, "attention_mask": src_mask,
            #     "labels": target_ids, "pixel_values": image}
        else:
            return {"input_ids": source_ids, "attention_mask": src_mask,
                    "labels": target_ids, "pixel_values": image, "index": index,
                    'target_text': target_text}
    
    @property
    def classname(self):
        return self.__class__.__name__
    
    def __repr__(self):
        return f"{self.classname}-{self.split}"
    
import gc
@Registry.register("InstructBlipDataset")
class InstructBlipDataset(SectionDataset): # Prompted Input for zero-shot models
    def __init__(self, csv_name, processor, max_src_len=64, max_tgt_len=32, prompt='', split='train', return_target_text=False, print_input=[False, False], **kwargs):
        super().__init__(csv_name, processor, max_src_len, max_tgt_len, prompt, split, return_target_text, print_input, **kwargs)

    def __getitem__(self, index):

        if index < 8600 * 8:
            return torch.zeros(1)

        # return {'text': self.src, 'label': self.tgt}
        sample_idx, img_path, img_idx, sec_idx = self.df.iloc[index]
        # print(sample_idx, img_path, img_idx, sec_idx)
        
        pkl_path = os.path.join(PREP_DIR, 'extracted_texts', self.split, (str(sample_idx) + '.pkl'))
        text_data = load_pkl(pkl_path)

        sec_title, sec_text = text_data['section_dict'][sec_idx]

        input_text = 'Generate image caption given the following context: ' +  \
                        sec_title + ". " + sec_text + ' Image caption: ' # prompt v2
        
        page_title = text_data['page_title']
        input_text = 'we use \"<<<...>>>\" to represent a context. Generate the image caption given the following context <<<' + \
                        page_title + ". " + sec_title + ". " + sec_text + ">>>. The image describes " 


        image_captions = text_data['image_captions'] # dict of dict

        target_text = image_captions[sec_idx][img_idx]

        with Image.open(img_path).convert('RGB') as image:
            inputs = self.processor(images=image, text=input_text, return_tensors="pt", max_length=self.max_src_len, padding="max_length", truncation = True)

        for k, v in inputs.items():
            inputs[k] = v[0]

        gc.collect()

        inputs['labels'] = target_text
        return inputs
    
@Registry.register("LlavaDataset")
class LlavaDataset(SectionDataset): # Prompted Input for zero-shot models
    def __init__(self, csv_name, processor, max_src_len=64, max_tgt_len=32, prompt='', split='train', return_target_text=False, print_input=[False, False], **kwargs):
        super().__init__(csv_name, processor, max_src_len, max_tgt_len, prompt, split, return_target_text, print_input, **kwargs)

    def __getitem__(self, index):


        sample_idx, img_path, img_idx, sec_idx = self.df.iloc[index]
        
        pkl_path = os.path.join(PREP_DIR, 'extracted_texts', self.split, (str(sample_idx) + '.pkl'))
        text_data = load_pkl(pkl_path)

        sec_title, sec_text = text_data['section_dict'][sec_idx]

        input_text = 'Generate image caption given the following context: ' +  \
                        sec_title + ". " + sec_text + ' Image caption: ' # prompt v2
        
        page_title = text_data['page_title']
        # input_text = 'we use \"<<<...>>>\" to represent a context. Generate the image caption given the following context <<<' + \
        #                 page_title + ". " + sec_title + ". " + sec_text + ">>>. The image describes " 

        # input_text = page_title + ". " + sec_title + ". " + sec_text + "Generate the image caption given the above context:"
        input_text = 'USER: <image>\nGenerate a short image caption that aligns with the provided context.\n' + \
                     'CONTEXT:\n' + page_title + ". " + sec_title + ". " + sec_text
        # print(input_text)
        # exit()

        with Image.open(img_path).convert('RGB') as image:
            inputs = self.processor(images=image, text=input_text, return_tensors="pt", max_length=self.max_src_len, padding="max_length", truncation = True)
            

        for k, v in inputs.items():
            inputs[k] = v[0]
        
        tail = self.processor(text = "\nASSISTANT:",return_tensors="pt")
        
        # print(inputs['input_ids'], tail['input_ids'])
        inputs['input_ids'] = torch.cat([inputs['input_ids'], tail['input_ids'][0, 1:]])
        inputs['attention_mask'] = torch.cat([inputs['attention_mask'], tail['attention_mask'][0, 1:]])


        image_captions = text_data['image_captions'] # dict of dict

        target_text = image_captions[sec_idx][img_idx]
        tokenized_targets = self.processor(
            text=target_text, max_length=self.max_tgt_len, padding="max_length", return_tensors="pt", truncation=True
        )

        target_ids = tokenized_targets["input_ids"][0]

        inputs['labels'] = target_ids

        gc.collect()

        return inputs


class Web2MImageDataset(TorchDataset): # Image Dataset for caching image features locally
    def __init__(self, csv_name, split, model_name, pad = False):
        self.split = split
        print(f"Web2MImageDataset-{self.split} called")
        self.df = pd.read_csv(os.path.join(PREP_DIR, csv_name))
        self.df = self.df.drop(self.df[self.df['exist'] == 0].index)
        self.df = self.df.drop('exist', axis=1)

        print(f'entries: {len(self.df)}')
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.save_string = get_save_string(model_name)
        self.pad = pad
        

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        sample_idx, img_path, img_idx, sec_idx = self.df.iloc[index]

        try:
            image = Image.open(img_path).convert('RGB')
            image = convert_to_rgb(image)
        except:
            print(index, img_path)
            image = Image.new('RGB', (512, 512))
        if self.pad:
            # first resize then pad to save memory
            width, height = image.size
            aspect_ratio = width / height

            if width > height:
                new_width = 224
                new_height = int(new_width / aspect_ratio)
            else:
                new_height = 224
                new_width = int(new_height * aspect_ratio)

            image = image.resize((new_width, new_height))

            image = to_numpy_array(image)

            image = pad_to_square_np(image)

            
        new_path = get_feature_path(img_path, self.save_string, PREP_DIR)
        pooler_path = new_path.replace(self.save_string, self.save_string + '_')
        image = self.processor(images=image, return_tensors="pt")['pixel_values'][0]
        return {"pixel_values": image, "new_path": new_path, "pooler_path": pooler_path}
     
    @property
    def classname(self):
        return self.__class__.__name__
    
    def __repr__(self):
        return f"{self.classname}-{self.split}"
    

class Web2MTextDataset(TorchDataset):
    def __init__(self, tokenizer, csv_name, split, max_src_len=64, max_tgt_len=32, return_sentence_list = False, add_cap = False, all_inputs = False):
        self.df = pd.read_csv(os.path.join(PREP_DIR, csv_name))
        self.df = self.df.drop(self.df[self.df['exist'] == 0].index)
        self.df = self.df.drop('exist', axis=1)
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.return_sentence_list = return_sentence_list
        self.split = split
        self.add_cap = add_cap # variable for word score mask generation test purpose, remove later
        self.all_inputs = all_inputs

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):

        sample_idx, img_path, img_idx, sec_idx = self.df.iloc[index]
        pkl_path = os.path.join(PREP_DIR, 'extracted_texts', self.split, (str(sample_idx) + '.pkl'))
        text_data = load_pkl(pkl_path)

        sec_title, sec_text = text_data['section_dict'][sec_idx]
        page_title = text_data['page_title']

        image_captions = text_data['image_captions'] # dict of dict

        target_text = image_captions[sec_idx][img_idx]
        if self.all_inputs:
            raise NotImplementedError
        else:
            if self.add_cap:
                sec_text = target_text + '.' + sec_text
            else: 
                sec_text = " " + sec_text # make it consistent with the real training format, where there will be space between special token and this context
            tokenized_inputs = self.tokenizer(
                sec_text, max_length=self.max_src_len, padding="max_length", return_tensors="pt", truncation=True
            )
            tokenized_targets = self.tokenizer(
                target_text, max_length=self.max_tgt_len, padding="max_length", return_tensors="pt", truncation=True
            )
            source_ids = tokenized_inputs["input_ids"][0]
            target_ids = tokenized_targets["input_ids"][0]

            src_mask = tokenized_inputs["attention_mask"][0]
            target_mask = tokenized_targets["attention_mask"][0]

            if not self.return_sentence_list:
                return {"input_ids": source_ids, "attention_mask": src_mask,
                        "target_ids": target_ids, "target_mask": target_mask}
            else:
                pass



def is_symbol(char):
    return char in string.punctuation

def isalnum_mine(char):
    return not is_symbol

def token2word(tokens, token_masks):
    # Convert token masks to word-level masks
    word_level_masks = OrderedDict()
    current_word = ''
    current_word_mask_sum = 0.0
    token_count = 0

    for idx, (token, mask) in enumerate(zip(tokens, token_masks)):
        # Remove the '▁' (whitespace) prefix from tokens for alignment
        clean_token = token.replace("▁", " ").strip()
        
        # Check if the next token starts a new word
        next_token_starts_new_word = (idx < len(tokens) - 1 and tokens[idx+1].startswith("▁"))
        
        # If the token is alphanumeric, aggregate it
        if clean_token.isalnum():
            current_word += clean_token
            current_word_mask_sum += mask
            token_count += 1

        # If token is standalone punctuation/symbol or the next token starts a new word
        if not clean_token.isalnum() or next_token_starts_new_word:
            if current_word:  # Check if there's any word to finalize
                word_level_masks[current_word] = current_word_mask_sum / token_count
                current_word = ''
                current_word_mask_sum = 0.0
                token_count = 0

    # Handle any remaining words after the loop
    if current_word:
        word_level_masks[current_word] = current_word_mask_sum / token_count
    return word_level_masks


@Registry.register("Web2MMaskPrefixDataset")
class Web2MMaskPrefixDataset(TorchDataset):
    def __init__(self, csv_name, feature_matrix_name, split, max_src_len=64, max_tgt_len=32, 
                 add_input_mask = False, mask_dir = None, tokenizer = None, normalise_score = None, 
                 apply_to_attention_mask = False, mask_threshold = 0, mask_as_labels = False,
                return_prefix = False, drop_rate = 0, vary_rate = 0, pad_image = False,
                 token2word = False, prefix_dir = None, subset = False, subset_json = None, 
                 extract_sentence = False, max_extract_len = 32, **kwargs):
        self.split = split
        print(f"Web2MMaskPrefixDataset-{self.split} called")
        self.df = pd.read_csv(os.path.join(PREP_DIR, csv_name))
        # if 'exist' in self.df.columns:
        self.df = self.df.drop(self.df[self.df['exist'] == 0].index)
        self.df = self.df.drop('exist', axis=1)
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.tokenizer = tokenizer
        if feature_matrix_name is None:
            self.image_features = torch.zeros((len(self.df), 768))
            print("No Image Feature Matrix is Used, Using Zero Matrix Instead")
        else:
            # TODO Switch to padded if full set
            if pad_image: self.image_features = torch.load(os.path.join(PREP_DIR, "extracted_image_features", feature_matrix_name))
            else: self.image_features = torch.load(os.path.join(PREP_DIR, "extracted_image_features_unpadded", feature_matrix_name))
            print(self.image_features.shape)
        assert len(self.image_features) == len(self.df)
        self.add_input_mask = add_input_mask
        # if self.add_input_mask:
        self.mask_dir = mask_dir
        assert os.path.exists(os.path.join(PREP_DIR, MASK_DIR, self.mask_dir, split))
        self.normalise_mode = normalise_score
        self.apply_to_attention_mask = apply_to_attention_mask
        self.mask_threshold = mask_threshold
        self.mask_as_labels = mask_as_labels
        self.return_prefix = return_prefix   
        self.drop_rate = drop_rate
        self.vary_rate = vary_rate
        self.token2word = token2word
        self.prefix_dir = prefix_dir
        self.extract_sentence = extract_sentence
        self.max_extract_len = max_extract_len
        if subset_json is not None:
            subset_json = load_json(os.path.join(PREP_DIR, subset_json))
            subset_list = subset_json['list']
            self.df = self.df.iloc[subset_list]
            self.image_features = self.image_features[subset_list]
            self.subset_list = subset_list
        else:
            self.subset_list = None


        if subset:
            self.df = self.df[:256]
            self.image_features = self.image_features[:256]

    def is_train(self):
        return self.split == 'train'
    
    def get_mask_prefix_word(self, sec_token_ids, prefix_token_ids, sec_mask, override_mask = None, ret_word = False):
        valid_context_len = min(len(sec_token_ids), self.max_src_len - len(prefix_token_ids))
        
        sec_tokens = self.tokenizer.convert_ids_to_tokens(sec_token_ids)
        if len(sec_token_ids) <= self.max_src_len - len(prefix_token_ids):
            # no truncation
            sec_tokens = sec_tokens[:valid_context_len]
            tmp_mask = sec_mask[:valid_context_len]
        else:
            truncate_idx = valid_context_len
            # ensure the completeness of last token
            while truncate_idx > 0 and not sec_tokens[truncate_idx].startswith("▁") or is_symbol(sec_tokens[truncate_idx]):
                truncate_idx -= 1
            sec_tokens = sec_tokens[:truncate_idx]
            tmp_mask = sec_mask[:truncate_idx]
        
        if override_mask is not None:
            tmp_mask = override_mask

        word_weights = token2word(sec_tokens, tmp_mask.tolist())
        word_weights = {k: v for k, v in word_weights.items() if not is_symbol(k) and v > self.mask_threshold and not k == ''}


        upper_n = 40
        if len(word_weights) > upper_n:
            # Find the keys of the items with the smallest weight values that need to be removed
            items_to_remove = sorted(word_weights.items(), key=lambda x: x[1])[:-upper_n]
            keys_to_remove = set(key for key, _ in items_to_remove)

            prefix_words = [key for key in word_weights.keys() if key not in keys_to_remove]
        else:
            prefix_words = list(word_weights.keys())

        if ret_word:
            return prefix_words
        
        if len(prefix_words) == 0:
            return "<MSK> <CPT>"
        else:
            mask_prefix = "<MSK> " + prefix_words[0]
            for wd in prefix_words[1:]:
                mask_prefix = " ".join([mask_prefix, "<SEP>", wd])
            mask_prefix = " ".join([mask_prefix, "<CPT>"])
        return mask_prefix



    def get_mask_prefix(self, sec_token_ids, prefix_token_ids, sec_mask, override_mask = None):

        # len candidate > k
        def top_k_in_candidates(tensor, candidate_indices, k):
            sorted_indices = torch.argsort(tensor, descending=True)

            # Initialize an empty list to store selected indices
            selected_indices = []

            # Iterate through the sorted indices and add elements to the list if they are in candidate_indices
            candidate_list = candidate_indices.tolist()
            for index in sorted_indices:
                if index.item() in candidate_list and len(selected_indices) < k:
                    selected_indices.append(index.item())
                
                if len(selected_indices) == k:
                    break

            # Convert the list of selected indices to a tensor
            selected_indices = torch.tensor(selected_indices)
            return selected_indices

        # NOTE: Truncate considering max_src_length - len(prefix_token_ids)
        # This ensures only tokens present in the context are selected.
        valid_context_len = min(len(sec_token_ids), self.max_src_len - len(prefix_token_ids))
        tmp_mask = sec_mask[:valid_context_len]
        if override_mask is not None:
            tmp_mask = override_mask
        indices = torch.nonzero(tmp_mask > self.mask_threshold).squeeze()
        candidate_token_ids = sec_token_ids[indices]
        if candidate_token_ids.dim() == 0:
            return "<MSK> <CPT>"
        
        special_space = (b'\xe2\x96\x81').decode('utf-8')
        space_token_id = self.tokenizer.convert_tokens_to_ids(special_space)
        valid_indices = []
        seen_tokens = set()
        seen_tokens.add(space_token_id)
        # seen_tokens.add()
        for i, candidate_index in enumerate(indices.tolist()):
            token_id = sec_token_ids[candidate_index].item()
            if token_id not in seen_tokens:
                valid_indices.append(candidate_index)
                seen_tokens.add(token_id)
        if len(valid_indices) == 0:
            return "<MSK> <CPT>"
        # valid_indices are all indices > t without duplicates and space
        valid_indices = torch.tensor(valid_indices)

        # we try to keep this n same with/without dropout
        n_selected = min(len(valid_indices), 50) 
        # dropout several prefix for better training -- though this is not used for final model 
        if self.drop_rate > 0 and self.is_train():
            # if number correction is not needed, just change n_candi to n_select, and trunc selected_ind with n_sele * droprate
            n_dropout_candidates = int(n_selected / self.drop_rate)
            # here we try to select n unique largest element from sec_token_ids
            sub_token_ids = sec_token_ids[:len(tmp_mask)]
            token_mask_mapping = {token_id: mask_value for token_id, mask_value in zip(sub_token_ids.tolist(), tmp_mask.tolist())}

            sorted_token_ids = sorted(enumerate(sub_token_ids.tolist()), key=lambda x: token_mask_mapping[x[1]], reverse=True)
            # Select the top k non-duplicate elements
            selected_token_ids = []
            selected_indices = []
            for index, token_id in sorted_token_ids:
                if token_id not in selected_token_ids and token_id != space_token_id and len(selected_token_ids) < n_dropout_candidates:
                    selected_token_ids.append(token_id)
                    selected_indices.append(index)
            # Convert the selected_token_ids list to a tensor

            indices = random.sample(selected_indices, n_selected)
            indices.sort()
            final_token_ids = sub_token_ids[indices]

            prefix_tokens = self.tokenizer.convert_ids_to_tokens(final_token_ids)


            mask_prefix = "<MSK> " + prefix_tokens[0]
            for tk in prefix_tokens[1:]:
                mask_prefix = " ".join([mask_prefix, "<SEP>", tk])
            mask_prefix = " ".join([mask_prefix, "<CPT>"])
            return mask_prefix
        elif self.vary_rate > 0 and self.is_train():
            n_dropout_candidates = int(n_selected * (1 +self.vary_rate))
            if n_dropout_candidates == 0:
                return "<MSK> <CPT>"
            sub_token_ids = sec_token_ids[:len(tmp_mask)]
            token_mask_mapping = {token_id: mask_value for token_id, mask_value in zip(sub_token_ids.tolist(), tmp_mask.tolist())}
            sorted_token_ids = sorted(enumerate(sub_token_ids.tolist()), key=lambda x: token_mask_mapping[x[1]], reverse=True)
            selected_token_ids = []
            selected_indices = []
            for index, token_id in sorted_token_ids:
                if token_id not in selected_token_ids and token_id != space_token_id and len(selected_token_ids) < n_dropout_candidates:
                    selected_token_ids.append(token_id)
                    selected_indices.append(index)
            selected_indices.sort()
            final_token_ids = sub_token_ids[selected_indices]

            prefix_tokens = self.tokenizer.convert_ids_to_tokens(final_token_ids)
            mask_prefix = "<MSK> " + prefix_tokens[0]
            for tk in prefix_tokens[1:]:
                mask_prefix = " ".join([mask_prefix, "<SEP>", tk])
            mask_prefix = " ".join([mask_prefix, "<CPT>"])
            return mask_prefix
        else: 
            # top_n_values, top_n_indices = torch.topk(tmp_mask, n_selected)
            if len(valid_indices > 50):
                top_n_indices = top_k_in_candidates(tmp_mask, valid_indices, 50)
                indices, _ = torch.sort(top_n_indices)
            else:
                indices = valid_indices
            selected_token_ids = sec_token_ids[indices]
            prefix_tokens = self.tokenizer.convert_ids_to_tokens(selected_token_ids)
            mask_prefix = "<MSK> " + prefix_tokens[0]
            for tk in prefix_tokens[1:]:
                mask_prefix = " ".join([mask_prefix, "<SEP>", tk])
            mask_prefix = " ".join([mask_prefix, "<CPT>"])
            return mask_prefix


    def __len__(self):
        return len(self.df)
    
    # The other two args are for visualizations only -- not used during train/test
    def __getitem__(self, index, override_mask = None, return_raw_text = False):

        sample_idx, img_path, img_idx, sec_idx = self.df.iloc[index]

        pkl_path = os.path.join(PREP_DIR, 'extracted_texts', self.split, (str(sample_idx) + '.pkl'))
        text_data = load_pkl(pkl_path)

        # ['ImageToken', 'SectionIndex', 'SectionTitle', 'SectionText', 'PageURL', 'PageTitle', 'ImageCaption']
        sec_title, sec_text = text_data['section_dict'][sec_idx]
        page_title = text_data['page_title']
        image_captions = text_data['image_captions'] # dict of dict
        image_feature = self.image_features[index].clone()
        target_text = image_captions[sec_idx][img_idx]
        tokenized_targets = self.tokenizer(
                target_text, max_length=self.max_tgt_len, padding="max_length", return_tensors="pt", truncation=True
        )
        target_ids = tokenized_targets["input_ids"][0]
        if len(sec_text) > 0:
            input_prefix = " ".join(['[ImageToken]', '[PageTitle]', page_title, 
                            '[SectionTitle]', sec_title, '[SectionText]'])
        else:
            input_prefix = " ".join(['[ImageToken]', '[PageTitle]', page_title, 
                            '[SectionTitle]', sec_title])
        # remove eos token here
        prefix_token_ids = self.tokenizer.encode(input_prefix, return_tensors="pt")[0, : -1]
        # padding = 'max_length' is not use here so its fine
        sec_token_ids = self.tokenizer.encode(sec_text, return_tensors="pt", max_length=self.max_src_len, truncation = True)[0, : -1]
        prefix_mask = torch.ones(len(prefix_token_ids))
        if self.subset_list is not None:
            mask_path = os.path.join(PREP_DIR, MASK_DIR, self.mask_dir, self.split, str(get_shard_index(self.subset_list[index], 1000)), str(self.subset_list[index]) + '.pt')
        else:
            mask_path = os.path.join(PREP_DIR, MASK_DIR, self.mask_dir, self.split, str(get_shard_index(index, 1000)), str(index) + '.pt')
        sec_mask = torch.load(mask_path)

        if override_mask is not None:
            print('original mask', sec_mask)
            assert len(override_mask) == len(sec_mask)
            sec_mask = override_mask

        if len(sec_mask) >= len(sec_token_ids):
            sec_mask = sec_mask[:self.max_src_len]
            if self.normalise_mode == 'minmax':
                sec_mask = normalize_scores(sec_mask, 0, 1)
                if isinstance(sec_mask, torch.Tensor):
                    pass
                else:
                    sec_mask = torch.tensor(sec_mask)
            elif self.normalise_mode == None:
                pass
            elif self.normalise_mode == 'fix_scale':
                sec_mask = sec_mask/2 + 0.5
            else:
                raise NotImplementedError
            # print(sec_mask)
            sec_mask = sec_mask.detach()
            sec_mask.requires_grad = False
            
            # print(sec_text)
            if self.prefix_dir is not None:
                prefix_file_name = os.path.join(self.prefix_dir, f"{index}_prefix.txt")
                with open(prefix_file_name, "r") as file:
                    mask_prefix = file.read()
                # print(index, mask_prefix)
            elif self.extract_sentence:
                candidate_words = self.get_mask_prefix_word(sec_token_ids, prefix_token_ids, sec_mask, override_mask, ret_word=True)
            elif self.token2word:
                mask_prefix = self.get_mask_prefix_word(sec_token_ids, prefix_token_ids, sec_mask, override_mask)
            else:
                mask_prefix = self.get_mask_prefix(sec_token_ids, prefix_token_ids, sec_mask, override_mask)

            if self.extract_sentence:
                shard_index = get_shard_index(index, 1000)
                save_dir = os.path.join(PREP_DIR, 'filtered_sentences', self.split, str(shard_index))
                import pathlib
                save_path= pathlib.Path(save_dir)
                save_path.mkdir(exist_ok=True, parents=True)
                save_path = os.path.join(save_dir, str(index) + '.txt')
                
                with open(save_path, 'r') as file:
                    # Read the entire content of the file as a single string
                    new_sec_text = file.read()
                
                sec_token_ids = self.tokenizer.encode(new_sec_text, return_tensors="pt", max_length=self.max_extract_len, truncation = True)[0, : -1]
                input_ids = torch.cat([prefix_token_ids, sec_token_ids])
                if len(input_ids) >= self.max_extract_len:
                    input_ids = input_ids[:self.max_extract_len]
                    input_ids[-1] = self.tokenizer.eos_token_id
                    src_mask = torch.ones(self.max_extract_len)
                else:
                    src_mask = torch.cat([torch.ones(len(input_ids)), torch.zeros(self.max_extract_len - len(input_ids))])
                    input_ids = torch.nn.functional.pad(input_ids, (self.tokenizer.pad_token_id, self.max_extract_len - len(input_ids)))

                return {"input_ids": input_ids, "attention_mask": src_mask,
                    "labels": target_ids, "image_feature": image_feature,
                    }
            
            
            # For T5, pad token is used as the start of setence token, this might have problem is other tokenizer is used
            eos_token_id = self.tokenizer.eos_token_id
            pad_token_id = self.tokenizer.pad_token_id
            if self.return_prefix: # we assume single element batch here, so no padding

                decoder_prefix_ids = self.tokenizer(
                    "<pad> " + mask_prefix, max_length=self.max_tgt_len+1, return_tensors="pt", truncation=True
                )["input_ids"][0][:-1] # Get rid of the eos token


            # Get prefix mask
            loss_prefix_tmp = self.tokenizer(
                mask_prefix, max_length=self.max_tgt_len, padding = 'max_length', return_tensors="pt", truncation=True
            )
            loss_prefix_ids = loss_prefix_tmp["input_ids"][0]
            eos_positions = (loss_prefix_ids== eos_token_id).nonzero()
            loss_prefix_mask = loss_prefix_tmp["attention_mask"][0]
            for position in eos_positions:
                loss_prefix_mask[position[0]] = 0

            # https://github.com/huggingface/transformers/issues/10478
            target_text = " ".join([mask_prefix, target_text])
            tokenized_targets = self.tokenizer(
                target_text, max_length=self.max_tgt_len, padding="max_length", return_tensors="pt", truncation=True
            )
            target_ids = tokenized_targets["input_ids"][0]

        else:
            print(f'sec mask shorter than input at index {str(index)}, needs re-process mask with longer length')
            raise NotImplementedError

        caption_text = "" # Remaining captions in the section
        for k, v in image_captions[sec_idx].items():
            if k != img_idx and len(v) > 0:
                caption_text = " ".join([caption_text, '[ImageCaption]', v])

        caption_ids = self.tokenizer.encode(caption_text, return_tensors="pt")[0]

        caption_mask = torch.zeros(len(caption_ids))
        # Fill with values to avoid total removal of this information
        if self.normalise_mode == 'minmax':
            caption_mask_val = 0.4
        elif self.normalise_mode == 'fix_scale':
            caption_mask_val = 0.5
        elif self.normalise_mode == None:
            caption_mask_val = 0.2 
        else:
            raise NotImplementedError
        caption_mask = torch.full((len(caption_ids), ), caption_mask_val)
        # change eos token weight to 1 
        caption_mask[-1] = 1

        input_ids = torch.cat([prefix_token_ids, sec_token_ids, caption_ids])
        input_cmask = torch.cat([prefix_mask, sec_mask, caption_mask])
        if self.mask_as_labels:
            assert self.normalise_mode == 'fix_scale'
            token_weight_mask = torch.cat([torch.zeros(len(prefix_token_ids)), torch.ones(len(sec_token_ids)), torch.zeros(len(caption_ids))])
        if len(input_ids) >= self.max_src_len:
            input_ids = input_ids[:self.max_src_len]
            input_ids[-1] = self.tokenizer.eos_token_id
            input_cmask = input_cmask[:self.max_src_len]
            src_mask = torch.ones(self.max_src_len)
            if self.mask_as_labels:
                token_weight_mask = token_weight_mask[:self.max_src_len]
        else:
            src_mask = torch.cat([torch.ones(len(input_ids)), torch.zeros(self.max_src_len - len(input_ids))])
            input_ids = torch.nn.functional.pad(input_ids, (self.tokenizer.pad_token_id, self.max_src_len - len(input_ids)))
            input_cmask = torch.nn.functional.pad(input_cmask, (0, self.max_src_len - len(input_cmask)))
            if self.mask_as_labels:
                token_weight_mask = torch.nn.functional.pad(token_weight_mask, (0, self.max_src_len - len(token_weight_mask)))

        if self.mask_as_labels:
            return {"input_ids": input_ids, "attention_mask": src_mask,
                "labels": target_ids, "image_feature": image_feature,
                "token_weight_mask": token_weight_mask, 'token_weights': input_cmask}
        if self.apply_to_attention_mask:
            bool_filter = input_cmask >= self.mask_threshold
            input_cmask[bool_filter] = 0

        out = {"input_ids": input_ids, "attention_mask": src_mask,
                    "labels": target_ids, "image_feature": image_feature,
                    }
        
        if return_raw_text:
            out['raw_texts'] = sec_text
            image = Image.open(img_path).convert('RGB')
            image = convert_to_rgb(image)
            image.show()

        if self.return_prefix:
            out['decoder_input_ids'] = decoder_prefix_ids
            return out

        else:
            return {"input_ids": input_ids, "attention_mask": src_mask,
                "labels": target_ids, "image_feature": image_feature,
                "prefix_mask":loss_prefix_mask}
           
    @property
    def classname(self):
        return self.__class__.__name__
    
    def __repr__(self):
        return f"{self.classname}-{self.split}"
    
# Dataset class for ctrl-cic eval, there will be a subset df of samples feed to this
@Registry.register("CCICInferenceDataset")
class CCICInferenceDataset(TorchDataset):
    def __init__(self, csv_name, feature_matrix_name, split, max_src_len=64, max_tgt_len=32, 
                 mask_dir = None, tokenizer = None, normalise_score = None, mode = 'word_prefix',
                 pad_image = False, ccic_json_path = None, highlight_weight = 0.7, 
                 return_highlight_sents = False, return_clip_image = False, processor = None, **kwargs):
        # return_highlight_sents and return_clip_image are for clip-based evaluation metrics 
        self.split = split
        print(f"CCICInferenceDataset-{self.split} called")
        self.df = pd.read_csv(os.path.join(PREP_DIR, csv_name))

        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        if processor is None:
            self.tokenizer = tokenizer
            assert tokenizer is not None
        else:
            if mode == 'zeroshot-llm':
                pass

            elif mode == 'zeroshot-qwen':
                self.processor = processor
                self.tokenizer = processor
            else:
                self.processor = processor
                self.tokenizer = processor.tokenizer

        if feature_matrix_name is None:
            self.image_features = torch.zeros((len(self.df), 768))
            print("No Image Feature Matrix is Used, Using Zero Matrix Instead")
        else:
            if pad_image: self.image_features = torch.load(os.path.join(PREP_DIR, "extracted_image_features", feature_matrix_name))
            else: self.image_features = torch.load(os.path.join(PREP_DIR, "extracted_image_features_unpadded", feature_matrix_name))
            print(self.image_features.shape)
        # assert len(self.image_features) == len(self.df)
        self.mask_dir = mask_dir
        assert os.path.exists(os.path.join(PREP_DIR, MASK_DIR, self.mask_dir, split))
        self.normalise_mode = normalise_score
        ccic_json_path = os.path.join(PREP_DIR,'selected_highlights', ccic_json_path)
        self.highlight_samples = load_json(ccic_json_path)
        self.highlight_samples = {int(key): value for key, value in self.highlight_samples.items()}
        self.mode = mode
        if self.mode == 'reweight':
            self.highlight_weight = highlight_weight

        self.return_highlight_sents = return_highlight_sents
        self.return_clip_image = return_clip_image
        if self.return_clip_image or self.return_highlight_sents: # return clip-preprocessed image for clipscore
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        self.debug = kwargs['debug']
        if self.debug:
            self.df = self.df[:10]

    def is_train(self):
        return self.split == 'train'

    def __len__(self):
        return len(self.df)
    
    # The other two args are for visualizations only
    def __getitem__(self, index, override_mask = None, return_raw_inputs = False):

        sample_idx, img_path, img_idx, sec_idx, original_index, key = self.df.iloc[index]

        pkl_path = os.path.join(PREP_DIR, 'extracted_texts', self.split, (str(sample_idx) + '.pkl'))
        text_data = load_pkl(pkl_path)

        # ['ImageToken', 'SectionIndex', 'SectionTitle', 'SectionText', 'PageURL', 'PageTitle', 'ImageCaption']
        sec_title, sec_text = text_data['section_dict'][sec_idx]
        page_title = text_data['page_title']
        image_captions = text_data['image_captions'] # dict of dict
        image_feature = self.image_features[original_index].clone()

        target_text = image_captions[sec_idx][img_idx]
        if self.tokenizer is not None:
            if self.tokenizer.pad_token == None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.padding_side = "right"
            tokenized_targets = self.tokenizer(
                    target_text, max_length=self.max_tgt_len, padding="max_length", return_tensors="pt", truncation=True
            )
        target_ids = tokenized_targets["input_ids"][0]
        if len(sec_text) > 0:
            input_prefix = " ".join(['[ImageToken]', '[PageTitle]', page_title, 
                            '[SectionTitle]', sec_title, '[SectionText]'])
        else:
            input_prefix = " ".join(['[ImageToken]', '[PageTitle]', page_title, 
                            '[SectionTitle]', sec_title])
        # remove eos token here
        prefix_token_ids = self.tokenizer.encode(input_prefix, return_tensors="pt")[0, : -1]
        sec_token_ids = self.tokenizer.encode(sec_text, return_tensors="pt", max_length=self.max_src_len, truncation = True)[0, : -1]

        prefix_mask = torch.ones(len(prefix_token_ids))

        # load from dict                
        highlight_segments_list = self.highlight_samples[original_index]
        # key is something like 0001_1 -- index_ith-higlight

        ith_highlight = int(key[-1])
        # highlight segments will be a list of one-five element
        highlight_segments = highlight_segments_list[ith_highlight]

        caption_text = "" # Remaining captions in the section
        for k, v in image_captions[sec_idx].items():
            if k != img_idx and len(v) > 0:
                caption_text = " ".join([caption_text, '[ImageCaption]', v])
        caption_ids = self.tokenizer.encode(caption_text, return_tensors="pt")[0]

        sec_string = sec_text # cannot reduce the sec_text length first here, as that will turn many non-english tokens to unk and untracable
        highlight_in_title = False

        replaced_flag = False # this flag should only work for multiple highlights        

        for i_can, candidate_word in enumerate(highlight_segments):
            if candidate_word not in sec_string:
                if candidate_word.lower() in page_title.lower():
                    # usually we would not like the highlight to be only in the title itself, but if that happens.. 
                    sent_to_include = [page_title] 
                    highlight_in_title = True
                elif candidate_word.lower() in sec_title.lower():
                    sent_to_include = [sec_title]
                    highlight_in_title = True
                else:
                    print('bad highlight in', index)
                    print(page_title, sec_title)
                    print(sec_string)
                    print(candidate_word)
                    exit()
            else:
                if len(highlight_segments) == 1:
                    # replace highlights to avoid being segmented to different sentences.
                    sec_string = sec_string.replace(candidate_word, f'SPC_HOLDER_{i_can}')
                else: 
                    if len(split_into_sentences(candidate_word)) > 1:
                        sec_string = sec_string.replace(candidate_word, f'SPC_HOLDER_{i_can}')
                        replaced_flag = True
        if highlight_in_title:
            if self.debug: 
                print('highlight in title', index)
            pass
        else:
            if self.debug:
                print(replaced_flag)
                print(highlight_segments)
            sentences = split_into_sentences(sec_string)
            sent_to_include = []
            if len(highlight_segments) == 1: 
                for sent in sentences:
                    for i_can, candidate_word in enumerate(highlight_segments):
                        sent = sent.replace(f'SPC_HOLDER_{i_can}', candidate_word)
                        if candidate_word in sent:
                            assert 'SPC_HOLDER' not in sent
                            sent_to_include.append(sent)
                            break
            else: 
                if replaced_flag:
                    for ith_sent, sent in enumerate(sentences):
                        # space_holder_present = False
                        for i_can, candidate_word in enumerate(highlight_segments):
                            # if 'SPC_HOLDER' in sent:
                            #     space_holder_present = True
                            sent = sent.replace(f'SPC_HOLDER_{i_can}', candidate_word)
                            sentences[ith_sent] = sent
                            # if space_holder_present:
                            #     print(sentences[ith_sent])
                for ith_sent, sent in enumerate(sentences):
                    for i_can, candidate_word in enumerate(highlight_segments):
                        if candidate_word in sent:
                            if 'SPC_HOLDER' in sent:
                                print('bad spotted')
                                print(sent)
                                print()
                                exit()
                            # assert 'SPC_HOLDER' not in sent
                            sent_to_include.append(sent)
                            break
        if len(sent_to_include) == 0:
            print(page_title, sec_title)
            print(sec_text)
            print(highlight_segments)
            print('Zero Highlight Sent, Unexpected Case at', index) 


        if self.mode == 'extractive':
            
            new_sec_text= " ".join(sent_to_include)
            sec_token_ids = self.tokenizer.encode(new_sec_text, return_tensors="pt", max_length=self.max_src_len, truncation = True)[0, : -1]
            input_ids = torch.cat([prefix_token_ids, sec_token_ids])
            if len(input_ids) >= self.max_src_len:
                input_ids = input_ids[:self.max_src_len]
                input_ids[-1] = self.tokenizer.eos_token_id
                src_mask = torch.ones(self.max_src_len)
            else:
                src_mask = torch.cat([torch.ones(len(input_ids)), torch.zeros(self.max_src_len - len(input_ids))])
                input_ids = torch.nn.functional.pad(input_ids, (self.tokenizer.pad_token_id, self.max_src_len - len(input_ids)))

            if self.debug:
                print('highlight:', ",".join(highlight_segments))
                print('input string:', input_prefix, new_sec_text)
                if self.return_highlight_sents:
                    print('sent_to_include:', sent_to_include)
                print()
                # print('input id', input_ids)
            out = {"input_ids": input_ids, "attention_mask": src_mask,
                "labels": target_ids, "image_feature": image_feature, 
                "key": key}
            

            if self.return_clip_image:
                image = Image.open(img_path)
                image_processed = self.clip_processor(images=image, return_tensors='pt')['pixel_values'][0]
                out['image'] = image_processed


            if self.return_highlight_sents:
                # CLIP have a 77 token limits.
                batch_tokenized = self.clip_processor(
                    sent_to_include,
                    max_length=77,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt',
                )
                # print('batch_tokenized', batch_tokenized)
                out['highlight_sents'] = batch_tokenized

            return out
        
        if self.mode == 'zeroshot':

            highlight_merged = ''
            if len(highlight_segments) == 1:
                highlight_merged = highlight_segments[0] + '\n'
            else:
                for i, highlight_phrases in enumerate(highlight_segments):
                    highlight_merged = "". join([highlight_merged, f"{i+1}. " , highlight_phrases, '\n'])
            input_text = 'USER: <image>\n Generate a short image caption that aligns with the provided context, particularly focusing on the highlighted part.\n' + \
                        'HIGHLIGHT:\n' + highlight_merged + '\n' + 'CONTEXT:\n' + page_title + ". " + sec_title + ". " + sec_text


            with Image.open(img_path).convert('RGB') as image:

                inputs = self.processor(images=image, text=input_text, return_tensors="pt", max_length=self.max_src_len, truncation = True)


            for k, v in inputs.items():
                inputs[k] = v[0]
            
            tail = self.processor(text = "\nASSISTANT:",return_tensors="pt")
            
            inputs['input_ids'] = torch.cat([inputs['input_ids'], tail['input_ids'][0, 1:]])
            inputs['attention_mask'] = torch.cat([inputs['attention_mask'], tail['attention_mask'][0, 1:]])

            out = {"labels": target_ids, "key": key}
            
            out.update(inputs)
            
            if self.return_clip_image:
                image = Image.open(img_path)
                image_processed = self.clip_processor(images=image, return_tensors='pt')['pixel_values'][0]
                out['image'] = image_processed


            if self.return_highlight_sents:
                # CLIP have a 77 token limits.
                batch_tokenized = self.clip_processor(
                    sent_to_include,
                    max_length=77,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt',
                )
                # print('batch_tokenized', batch_tokenized)
                out['highlight_sents'] = batch_tokenized
            return out
        
        if self.mode == 'zeroshot-llm':
            highlight_merged = ''
            if len(highlight_segments) == 1:
                highlight_merged = highlight_segments[0] + '\n'
            else:
                for i, highlight_phrases in enumerate(highlight_segments):
                    highlight_merged = "". join([highlight_merged, f"{i+1}. " , highlight_phrases, '\n'])
            
            
            sys_prompt = "<s>[INST] <<SYS>>\nYou are a helpful, respectful and honest assistant. Given a context and some highlights, without any specific image descriptions, please generate a short image caption that aligns with the provided context, particularly focusing on the highlighted part. (one sentence without any explanation or details)\n\n <</SYS>>"
            
            input_text = sys_prompt + 'HIGHLIGHT:\n' + highlight_merged + '\n' + 'CONTEXT:\n' + page_title + ". " + sec_title + ". " + sec_text + "\nCaption (do not add quotations surrounding the response):"

            inputs = self.tokenizer(text=input_text, return_tensors="pt", max_length=self.max_src_len - 1, truncation = True)

            for k, v in inputs.items():
                inputs[k] = v[0]
            
            tail = self.tokenizer(text = "[/INST]",return_tensors="pt")
            inputs['input_ids'] = torch.cat([inputs['input_ids'], tail['input_ids'][0, 1:]])
            inputs['attention_mask'] = torch.cat([inputs['attention_mask'], tail['attention_mask'][0, 1:]])

            out = {"labels": target_ids, "key": key}
            
            out.update(inputs)
            
            if self.return_clip_image:
                image = Image.open(img_path)
                image_processed = self.clip_processor(images=image, return_tensors='pt')['pixel_values'][0]
                out['image'] = image_processed

            # batch processing of highlight sents is actually possible, just padd the sentences return another mask
            # with n_padded sentences similar to attention mask
            if self.return_highlight_sents:
                # CLIP have a 77 token limits.
                batch_tokenized = self.clip_processor(
                    sent_to_include,
                    max_length=77,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt',
                )
                # print('batch_tokenized', batch_tokenized)
                out['highlight_sents'] = batch_tokenized
            return out

        if self.mode == 'reweight': # recalibration
            # we will not need gt relevance weight during ccic, a placeholder should suffice
            # sec_mask = torch.ones(len(sec_token_ids))
            sec_mask = torch.full((len(sec_token_ids),), 0.5)
            sec_mask.requires_grad = False

            # length -1 inaccount for the eos token that will be overwritting the last token
            tokenized_context = self.tokenizer.tokenize(sec_text, max_length=self.max_src_len - 1, truncation = True)
            assert len(tokenized_context) == len(sec_token_ids)
            # Check each phrase in the phrases list
            for phrase in highlight_segments:
                tokenized_phrase = self.tokenizer.tokenize(phrase)
                # Find the start index of the tokenized phrase in the tokenized context
                for i in range(len(tokenized_context) - len(tokenized_phrase) + 1):
                    if tokenized_context[i:i+len(tokenized_phrase)] == tokenized_phrase:
                        # Set the weights for tokens in the phrase
                        for j in range(len(tokenized_phrase)):
                            sec_mask[i+j] = self.highlight_weight

            if override_mask is not None:
                print('original mask', sec_mask)
                assert len(override_mask) == len(sec_mask)
                sec_mask = override_mask

            caption_mask = torch.zeros(len(caption_ids))

            if self.normalise_mode == 'minmax':
                caption_mask_val = 0.4
            elif self.normalise_mode == 'fix_scale': # only this will be used
                caption_mask_val = 0.5
            elif self.normalise_mode == None:
                caption_mask_val = 0.2 
            else:
                raise NotImplementedError
            caption_mask = torch.full((len(caption_ids), ), caption_mask_val)
            # change eos token weight to 1 
            caption_mask[-1] = 1

            input_ids = torch.cat([prefix_token_ids, sec_token_ids, caption_ids])
            input_cmask = torch.cat([prefix_mask, sec_mask, caption_mask])
            token_weight_mask = torch.cat([torch.zeros(len(prefix_token_ids)), torch.ones(len(sec_token_ids)), torch.zeros(len(caption_ids))])
            if len(input_ids) >= self.max_src_len:
                input_ids = input_ids[:self.max_src_len]
                input_ids[-1] = self.tokenizer.eos_token_id
                input_cmask = input_cmask[:self.max_src_len]
                src_mask = torch.ones(self.max_src_len)
                token_weight_mask = token_weight_mask[:self.max_src_len]
            else:
                src_mask = torch.cat([torch.ones(len(input_ids)), torch.zeros(self.max_src_len - len(input_ids))])
                input_ids = torch.nn.functional.pad(input_ids, (self.tokenizer.pad_token_id, self.max_src_len - len(input_ids)))
                input_cmask = torch.nn.functional.pad(input_cmask, (0, self.max_src_len - len(input_cmask)))
                token_weight_mask = torch.nn.functional.pad(token_weight_mask, (0, self.max_src_len - len(token_weight_mask)))

            out = {"input_ids": input_ids, "attention_mask": src_mask,
                    "labels": target_ids, "image_feature": image_feature,
                    "token_weights": input_cmask, "key": key, "token_weight_mask": token_weight_mask,}
            if return_raw_inputs:
                out['raw_image'] = Image.open(img_path)
                # out['raw_texts'] = 
            if self.return_clip_image:
                image = Image.open(img_path)
                image_processed = self.clip_processor(images=image, return_tensors='pt')['pixel_values'][0]
                out['image'] = image_processed

            # batch processing of highlight sents is actually possible, just padd the sentences return another mask
            # with n_padded sentences similar to attention mask
            if self.return_highlight_sents:
                # CLIP have a 77 token limits.
                batch_tokenized = self.clip_processor(
                    sent_to_include,
                    max_length=77,
                    truncation=True,
                    padding='max_length',
                    return_tensors='pt',
                )
                out['highlight_sents'] = batch_tokenized
            return out


        if self.mode == 'word_prefix':

            mask_prefix = "<MSK> " + highlight_segments[0]
            for wd in highlight_segments[1:]:
                mask_prefix = " ".join([mask_prefix, "<SEP>", wd])
            mask_prefix = " ".join([mask_prefix, "<CPT>"])

        elif self.mode == 'token_prefix':
            prefix_tokens = []
            for phrase in highlight_segments:
                tokenized_phrase = self.tokenizer.tokenize(phrase)
                for token in tokenized_phrase:
                    prefix_tokens.append(token)
            mask_prefix = "<MSK> " + prefix_tokens[0]
            for tk in prefix_tokens[1:]:
                mask_prefix = " ".join([mask_prefix, "<SEP>", tk])
            mask_prefix = " ".join([mask_prefix, "<CPT>"])


        input_ids = torch.cat([prefix_token_ids, sec_token_ids, caption_ids])
        
        if len(input_ids) >= self.max_src_len:
            input_ids = input_ids[:self.max_src_len]
            input_ids[-1] = self.tokenizer.eos_token_id
            src_mask = torch.ones(self.max_src_len)
        else:
            src_mask = torch.cat([torch.ones(len(input_ids)), torch.zeros(self.max_src_len - len(input_ids))])
            input_ids = torch.nn.functional.pad(input_ids, (self.tokenizer.pad_token_id, self.max_src_len - len(input_ids)))


        decoder_prefix_ids = self.tokenizer(
            "<pad> " + mask_prefix, max_length=self.max_tgt_len+1, return_tensors="pt", truncation=True
        )["input_ids"][0][:-1] # Get rid of the eos token

        target_text = " ".join([mask_prefix, target_text])
        tokenized_targets = self.tokenizer(
            target_text, max_length=self.max_tgt_len, padding="max_length", return_tensors="pt", truncation=True
        )
        target_ids = tokenized_targets["input_ids"][0]
        # print(target_text, target_ids)
        
        out = {"input_ids": input_ids, "attention_mask": src_mask,
                    "labels": target_ids, "image_feature": image_feature,
                "decoder_input_ids": decoder_prefix_ids, "key": key}
        
        if self.return_clip_image:
            image = Image.open(img_path)
            image_processed = self.clip_processor(images=image, return_tensors='pt')['pixel_values'][0]
            out['image'] = image_processed

        # batch processing of highlight sents is actually possible, just padd the sentences return another mask
        # with n_padded sentences similar to attention mask
        if self.return_highlight_sents:
            # CLIP have a 77 token limits.
            batch_tokenized = self.clip_processor(
                sent_to_include,
                max_length=77,
                truncation=True,
                padding='max_length',
                return_tensors='pt',
            )
            # print('batch_tokenized', batch_tokenized)
            out['highlight_sents'] = batch_tokenized
        return out
                    

    @property
    def classname(self):
        return self.__class__.__name__
    
    def __repr__(self):
        return f"{self.classname}-{self.split}"