import os
import evaluate
import numpy as np
import torch
from transformers import AutoProcessor, CLIPModel, CLIPProcessor
from transformers import logging, TrainingArguments, AutoTokenizer
import transformers.utils.logging as tlogging
from transformers.trainer_utils import EvalPrediction
from comm_ddp import comm
from tqdm import tqdm
import pathlib
from pycocoevalcap.cider.cider import Cider
import time

from cli.utils_registry import Registry, import_class_by_full_name
from utils.utils_io import save_json, load_json
# from huggingface_hub import login


def remove_prefix_by_ending_token(input_string, ending_token = '<CPT>'):
    index = input_string.rfind(ending_token)
    if index != -1:
        return input_string[index + len(ending_token):]
    else:
        return ' '
    
def remove_prefix_from_list(string_list):
    out_list = []
    for string in string_list:
        out_list.append(remove_prefix_by_ending_token(string))
    return out_list

# As for other metrics, can I import them from https://github.com/tylin/coco-caption/blob/master/pycocoevalcap/spice/spice.py 
# Or the transform and tell paper https://github.com/alasdairtran/transform-and-tell/blob/f1cc8382ea27bb508f4cbf5b519b18538d84e74e/scripts/compute_metrics.py#L4
def compute_metrics(pred, tokenizer = None, remove_prefix = False):
    # https://huggingface.co/spaces/evaluate-metric/bleu bleu defualt 4 grams
    # metrics = ['bleu', '']
    # cannot do it using a loop due to different input parameters
    metric_bleu = evaluate.load("bleu")
    metric_rougle = evaluate.load("rouge")
    metric_meteor = evaluate.load('meteor')
    metric_bert = evaluate.load('bertscore')

    if isinstance(pred, EvalPrediction):
        labels_ids = pred.label_ids 
        pred_ids = pred.predictions
    elif isinstance(pred, dict):
        labels_ids = pred['labels_ids']
        pred_ids = pred['predictions']
    elif isinstance(pred, tuple):
        labels_ids = pred[1]
        pred_ids = pred[0]

    if isinstance(tokenizer, AutoProcessor):
        tokenizer = tokenizer.tokenizer
    pred_ids[pred_ids == -100] = tokenizer.pad_token_id
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    

    if remove_prefix:
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=False)
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=False)
        pred_str = remove_prefix_from_list(pred_str)
        label_str = remove_prefix_from_list(label_str)
        pred_ids = tokenizer(pred_str)['input_ids']
        labels_ids = tokenizer(label_str)['input_ids']
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)


    else:
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    # print(pred_str)
    # print(label_str)
    bleu4 = metric_bleu.compute(predictions=pred_str, references=label_str)["bleu"]
    # bleu1 = metric_bleu.compute(predictions=pred_str, references=label_str, max_order = 1)["bleu"]
    bleu1 = 0
    rougeL = metric_rougle.compute(predictions=pred_str, references=label_str)["rougeL"]
    meteor = metric_meteor.compute(predictions=pred_str, references=label_str)["meteor"]

    bertscore = 0

    references = {index: [value] for index, value in enumerate(label_str)}
    candidate = {index: [value] for index, value in enumerate(pred_str)}
    # print(references)
    # print(candidate)
    scorer = Cider()
    cider = scorer.compute_score(references, candidate)[0]


    return {"bleu4": bleu4, "rougeL": rougeL, "meteor":meteor, 
            "bleu1":bleu1, "bertscore":bertscore, "cider": cider}
    # return {"bleu4": bleu4, "rougeL": rougeL, "meteor":meteor, "bleu1":bleu1, "bertscore":bertscore}

# https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13
def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may tracks all logits and consume excessive RAM. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    print(logits[0].shape)
    pred_ids = torch.argmax(logits[0], dim=-1)
    print(pred_ids.shape)
    print(pred_ids)
    # exit()
    return pred_ids, labels

import wandb
@Registry.register("SimpleEvaluator")
class EvaluatorWrapper:
    def __init__(self, **kwargs):

        run_id = kwargs['run_id']
        # print(run_id)
        tlogging.set_verbosity_error()

        self.run_id = run_id
        self.eval_steps = kwargs['eval_steps']
        self.max_steps = kwargs['max_steps']
        self.fp16 = kwargs['fp16']
        self.output_dir = kwargs['output_dir']
        # self.output_dir = os.path.join(kwargs['output_dir'], kwargs['run_id'])
        assert kwargs['do_val'] or kwargs['do_test'], 'choose at least do eval or test'
        self.do_val = kwargs['do_val']
        self.do_test = kwargs['do_test']
        self.test_model_index = kwargs['test_model_index']
        self.eval_batch_size = kwargs['per_device_eval_batch_size']
        self.save_model_output = kwargs['save_model_output']
        self.dataloader_num_workers = kwargs['dataloader_num_workers']
        if 'read_prefix' in kwargs: self.read_prefix = kwargs['read_prefix']
        else: self.read_prefix = False
        if 'generation_length_correction' in kwargs: self.generation_length_correction = kwargs['generation_length_correction']
        else: self.generation_length_correction = False
        if 'ccic_model' in kwargs: self.ccic_model = kwargs['ccic_model']
        else: self.ccic_model = None
        # length correction should only works at singular batch size
        if self.generation_length_correction: assert self.eval_batch_size == 1
        # self.use_image_embs = kwargs['use_image_embs']
        if 'wandb' in kwargs: self.wandb = kwargs['wandb']
        else: self.wandb = True
        if self.wandb:
            if self.read_prefix or self.ccic_model is not None:
                wandb.init(project="MMWebpage", id=run_id + '-extra', resume= False, dir = kwargs['output_dir'])
            else:
                wandb.init(project="MMWebpage", id=run_id, resume= True, dir = kwargs['output_dir'])
        if 'debug' in kwargs: self.debug = kwargs['debug']
        else: self.debug = False
        if 'load_predictions' in kwargs: self.load_predictions = kwargs['load_predictions']
        else: self.load_predictions = False
        # print(self.debug)
        # exit()
    
    def compile(self, dataset, **kwargs):
        print("SimpleEvaluator.compile() called")

        tokenizer_name = kwargs['model']['params']['pretrained_model_name_or_path']
        if 'tokenizer' in kwargs: self.tokenizer = kwargs['tokenizer']
        else: self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        special_tokens = ['[ImageToken]', '[SectionIndex]', '[SectionTitle]', '[SectionText]', '[PageURL]', '[PageTitle]', '[ImageCaption]']
        if 'extra_special_tokens' in kwargs:
            print(kwargs['extra_special_tokens'])
            special_tokens.extend(kwargs['extra_special_tokens'])
            # exit()
        token_dict = {'additional_special_tokens': special_tokens}
        num_added_tokens = self.tokenizer.add_special_tokens(token_dict)
        self.use_image_embs = kwargs['use_image_embs']
        if 'add_input_mask' in kwargs: self.add_input_mask = kwargs['add_input_mask']
        else: self.add_input_mask = False
        if 'mask_as_labels' in kwargs: self.mask_as_labels = kwargs['mask_as_labels']
        else: self.mask_as_labels = False
        if 'reweight_embs' in kwargs: self.reweight_embs = kwargs['reweight_embs']
        else: self.reweight_embs = False

        if 'use_local_model' in kwargs: self.use_local_model = kwargs['use_local_model']
        else: self.use_local_model = True
        if not self.use_local_model:
            self.pretrained_model_name = tokenizer_name

        self.model_class = import_class_by_full_name(kwargs['model']['model_class'])
        # if kwargs['model']['model_class'] in ['src.model.CICModel.TwoStageLongT5', 'src.model.CICModel.MaskedLongT5'] :
        # This is a not so decent checker for customised model class with extra args required
        has_other_params = any(key != 'pretrained_model_name_or_path' for key in kwargs['model']['params'].keys())
        if has_other_params:
            self.extra_model_args = kwargs['model']['params']
            self.extra_model_args.pop('pretrained_model_name_or_path')
        else:
            self.extra_model_args = None
        ### build dataset

        if 'return_prefix' in dataset['testset']['params']:
            self.remove_prefix = True
        else:
            self.remove_prefix = False

        if 'use_clip_score' in kwargs: self.use_clip_score = kwargs['use_clip_score']
        else: self.use_clip_score = False
        if 'use_sent_score' in kwargs: self.use_sent_score = kwargs['use_sent_score']
        else: self.use_sent_score = False
        if self.use_sent_score:
            model_name = "openai/clip-vit-large-patch14"
            self.clip_model = CLIPModel.from_pretrained(model_name).to('cuda')
            self.processor = CLIPProcessor.from_pretrained(model_name)

        if self.do_val:

            if self.wandb:
                wandb.define_metric("eval/global_step")
                wandb.define_metric("eval/*", step_metric="eval/global_step")
            
            self.validset = Registry.build_instance_from_cfg_node(dataset['validset'], tokenizer = self.tokenizer, add_input_mask = self.add_input_mask, mask_as_labels = self.mask_as_labels)
        if self.do_test:
            if self.wandb:
                wandb.define_metric("test/step")
                wandb.define_metric("test/*", step_metric="test/step")
            if self.use_local_model:
                if self.read_prefix: # this var should be used exclusively for ungrounded token moval
                    prefix_dir = os.path.join(self.output_dir, 'checkpoint-' + str(self.test_model_index), 'grounded')
                    # prefix_dir = os.path.join(self.output_dir, 'checkpoint-' + str(self.test_model_index), 'filtered_prefix')
                    self.testset = Registry.build_instance_from_cfg_node(dataset['testset'], tokenizer = self.tokenizer, add_input_mask = self.add_input_mask, mask_as_labels = self.mask_as_labels, prefix_dir = prefix_dir)
                else:
                    self.testset = Registry.build_instance_from_cfg_node(dataset['testset'], tokenizer = self.tokenizer, add_input_mask = self.add_input_mask, mask_as_labels = self.mask_as_labels, return_highlight_sents = self.use_sent_score, return_clip_image = self.use_clip_score)
            # if self.use_local_model:
                if isinstance(self.test_model_index, list):
                    self.test_model_paths = []
                    for text_index in self.test_model_index:
                        self.test_model_paths.append(os.path.join(self.output_dir, 'checkpoint-' + str(text_index)))
                else:
                    self.test_model_path = os.path.join(self.output_dir, 'checkpoint-' + str(self.test_model_index))
                    assert os.path.exists(self.test_model_path), 'Invalid Checkpoint Index'
            # print(self.test_model_path)
            else:
                if self.ccic_model == 'zeroshot-llm':
                    tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
                    self.testset = Registry.build_instance_from_cfg_node(dataset['testset'], tokenizer = tokenizer, add_input_mask = self.add_input_mask, mask_as_labels = self.mask_as_labels, return_highlight_sents = self.use_sent_score, return_clip_image = self.use_clip_score)
                    self.test_model_path = os.path.join(self.output_dir, str(self.pretrained_model_name))
                    os.makedirs(self.test_model_path, exist_ok=True)
                    self.tokenizer = tokenizer

                else:
                    processor = AutoProcessor.from_pretrained(self.pretrained_model_name)
                    self.testset = Registry.build_instance_from_cfg_node(dataset['testset'], processor = processor, add_input_mask = self.add_input_mask, mask_as_labels = self.mask_as_labels, return_highlight_sents = self.use_sent_score, return_clip_image = self.use_clip_score)
                    self.test_model_path = os.path.join(self.output_dir, str(self.pretrained_model_name))
                    os.makedirs(self.test_model_path, exist_ok=True)
                    if 'Qwen' in self.pretrained_model_name:
                        self.tokenizer = processor
                    else:
                        self.tokenizer = processor.tokenizer

        self.use_cached_feature = kwargs['use_cached_feature']
        
        
        
    def eval(self):
        if self.do_val: 
            best_model_index = self.eval_valid()
            print('Best Model Index', best_model_index)
        if self.do_test:
            if isinstance(self.test_model_index, list):
                for path in self.test_model_paths:
                    self.test_model_path = path
                    self.eval_test()
            else:
                self.eval_test()

    def eval_valid(self):
        # eval all ckpt in folder
        if self.eval_steps == 0:
            ckpt_list = os.listdir(self.output_dir)
            ckpt_idx_list = [int(x[11:]) for x in ckpt_list if 'checkpoint-' in x]
            ckpt_idx_list.sort()

            # ckpt_list = [prefix + str(x) for x in ckpt_idx_list]
            ckpt_list = [os.path.join(self.output_dir, 'checkpoint-' + str(x)) for x in ckpt_idx_list]
        else:
            ckpt_list = []
            for ckpt_index in range(self.eval_steps, self.max_steps, self.eval_steps):
                ckpt_path = os.path.join(self.output_dir, 'checkpoint-' + str(ckpt_index))
                if os.path.exists(ckpt_path): ckpt_list.append(ckpt_path)
                else: continue

        best_bleu = -1
        best_index = -1
        
        for ckpt_folder_path in ckpt_list:
            print(ckpt_folder_path)
            val_step = int(ckpt_folder_path[len(self.output_dir)+12:]) # "/checkpoint-"
            # exit()
            # continue
            model = self.model_class.from_pretrained(ckpt_folder_path)
            metric_values = self.inner_eval_loop(model, self.validset, ckpt_dir=os.path.join(ckpt_folder_path, 'val_results'))
            log_dict = {}
            for k, v in metric_values.items():
                log_dict['eval/' + k] = v
                print(k, v)
            log_dict['eval/global_step'] = val_step
            if self.wandb:
                wandb.log(log_dict)
            if metric_values['bleu4'] > best_bleu:
                best_bleu = metric_values['bleu4']
                best_index = val_step
        return best_index


    def eval_test(self):
        if self.use_local_model:
            test_step = int(self.test_model_path[len(self.output_dir)+12:])
            if self.extra_model_args is None:
                model = self.model_class.from_pretrained(self.test_model_path)
            else:
                model = self.model_class.from_pretrained(self.test_model_path, **self.extra_model_args)
        # print(self.test_model_path)
        else:
            # print(self.extra_model_args)
            # exit()
            torch_dtype = torch.float16 if self.fp16 else 'auto'  
            if self.extra_model_args is None:
                model = self.model_class.from_pretrained(self.pretrained_model_name, torch_dtype = torch_dtype)
            else:
                model = self.model_class.from_pretrained(self.pretrained_model_name, **self.extra_model_args, torch_dtype = torch_dtype)
                
            
        if self.read_prefix:
            metric_values = self.inner_eval_loop(model, self.testset, ckpt_dir=os.path.join(self.test_model_path, 'read_prefix_results'))
        else:
            metric_values = self.inner_eval_loop(model, self.testset, ckpt_dir=os.path.join(self.test_model_path, 'test_results'))
        log_dict = {}
        for k, v in metric_values.items():
            log_dict['test/' + k] = v
            print(k, v)
        if self.extra_model_args is not None:
            test_step += 1 # very hacky and bad way to prevent two stage model from writing to the same step, will see how to modify this
        if 'prefix' in self.run_id:
            test_step += 1
        log_dict['test/step'] = test_step

        print('----------------')
        # print(log_dict)
        # exit()
        if self.wandb:
            wandb.log(log_dict)

    def inner_eval_loop(self, model, dataset, ckpt_dir = None):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model.to(device)
        model.eval()
        # change pin_memory to True after the CPU RAM memory is confirmed enough
        eval_loader = torch.utils.data.DataLoader(dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.dataloader_num_workers, pin_memory=False)
        if self.ccic_model is not None:
            ccic_save_dict = {}

        if self.load_predictions:
            exp_dir = os.path.dirname(ckpt_dir)

            # modify this path 
            output_save_path = os.path.join(exp_dir, 'ccic_outputs_llama.json')

            all_predictions = load_json(output_save_path)

        pred = {'label_ids':[], 'predictions':[]}

        allow_overwrite_flag = False

        clip_scores = []
        highlight_scores = []

        if self.debug:
            time_cnt = []
            print(model.num_parameters())

        for idx, inputs in tqdm(enumerate(eval_loader), total=len(eval_loader)):

            if self.ccic_model is not None:
                keys = inputs.pop('key')
                if self.use_clip_score:
                    image_for_clip = inputs.pop('image').to('cuda')
                if self.use_sent_score:
                    sent_for_clip = inputs.pop('highlight_sents').to('cuda')

            for k, v in inputs.items():
                inputs[k] = v.to(device)


            labels = inputs['labels']

            if self.add_input_mask:
                token_weights = inputs.pop('token_weights')
            decoder_input_ids = None
            if self.remove_prefix or self.ccic_model is not None:
                if 'decoder_input_ids' in inputs:
                    decoder_input_ids = inputs.pop('decoder_input_ids')


            if self.debug:
                start_time = time.time()

            if self.load_predictions:
                output = None
            else:
                if self.use_local_model:
                    if self.use_cached_feature:
                        image_feature = inputs.pop('image_feature')
                    image_embs = image_feature.unsqueeze(1)
                    input_embedding_layer = model.get_input_embeddings()
                    input_embeddings = input_embedding_layer(inputs['input_ids'])

                    combined_embs = torch.cat((input_embeddings[:, :1, :], image_embs, input_embeddings[:, 1:, :]), dim = 1)
                    new_token_mask = torch.ones((combined_embs.shape[0],1), dtype=torch.int64).to(inputs['attention_mask'].device)
                    new_attention_mask = torch.cat((new_token_mask, inputs['attention_mask']), dim = 1)

                    with torch.no_grad():
                        if self.use_image_embs:
                            if self.add_input_mask:
                                # needs to add one to mask, in account for the image embs.
                                token_weights = torch.cat((new_token_mask.clone(), token_weights), dim = 1)
                                if self.mask_as_labels:
                                    zero_mask = torch.zeros((combined_embs.shape[0],1), dtype=torch.int64).to(inputs['attention_mask'].device)
                                    token_weight_mask = torch.cat((zero_mask.clone(), inputs['token_weight_mask']), dim = 1)
                                    output = model.generate(inputs_embeds=combined_embs, labels = inputs['labels'], attention_mask = new_attention_mask, token_weights = token_weights, token_weight_mask = token_weight_mask, max_new_tokens=inputs['labels'].shape[1]).detach()
                                elif self.reweight_embs:
                                    combined_embs = combined_embs * token_weights.unsqueeze(-1)
                                    output = model.generate(inputs_embeds=combined_embs, max_new_tokens=inputs['labels'].shape[1], attention_mask = new_attention_mask, token_weights = token_weights, decoder_input_ids = decoder_input_ids).detach()
                                else:
                                    output = model.generate(inputs_embeds=combined_embs, max_new_tokens=inputs['labels'].shape[1], attention_mask = new_attention_mask, token_weights = token_weights, decoder_input_ids = decoder_input_ids).detach()
                            else:
                                # prefix method here
                                if self.read_prefix:
                                    if self.generation_length_correction:
                                        max_new_tokens = inputs['labels'].shape[1] - len(decoder_input_ids[0])
                                        # print(idx, max_new_tokens)
                                        if max_new_tokens < 0:
                                            print('wrong number of new tokens', idx)
                                            exit()
                                        output = model.generate(inputs_embeds=combined_embs, max_new_tokens=max_new_tokens, attention_mask = new_attention_mask, decoder_input_ids = decoder_input_ids).detach()
                                    else:
                                        output = model.generate(inputs_embeds=combined_embs, max_new_tokens=inputs['labels'].shape[1], attention_mask = new_attention_mask, decoder_input_ids = decoder_input_ids).detach()

                                else:
                                    output = model.generate(inputs_embeds=combined_embs, max_new_tokens=inputs['labels'].shape[1], attention_mask = new_attention_mask, decoder_input_ids = decoder_input_ids).detach()
                        else:
                            if self.add_input_mask:
                                raise NotImplementedError
                            else:
                                output = model.generate(inputs_embeds=input_embeddings, max_new_tokens=inputs['labels'].shape[1], attention_mask = inputs['attention_mask']).detach()
                else:

                    with torch.no_grad():
                        if self.load_predictions:
                            output = None
                        else:
                            output = model.generate(**inputs, max_new_tokens=inputs['labels'].shape[1])
            if self.debug:
                end_time = time.time()
                elapsed_time = end_time - start_time
                time_cnt.append(elapsed_time)
            
            if not self.load_predictions:
                predictions, label_ids = output.to('cpu'), labels.to('cpu')

            if self.save_model_output:
                if self.remove_prefix: # This var indicates whether prefix method is used
                    pred_str = self.tokenizer.batch_decode(predictions, skip_special_tokens=False)
                    label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=False)

                else:
                    pred_str = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
                    label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
                for i in range(predictions.shape[0]):
                    if not self.use_local_model and 'llava' in self.pretrained_model_name:
                        # in this case the input is predicted together with output, we have to post-process it
                        output_line = pred_str[i].split('\n')[-1]
                        if output_line.startswith('ASSISTANT: '):
                            pred_str[i] = output_line[11:]
                        else:
                            pred_str[i] = output_line
                            print('ASSISTANT NOT FOUND FOR', idx * self.eval_batch_size + i)
                    assert ckpt_dir is not None
                    global_index = idx * self.eval_batch_size + i
                    if self.read_prefix: json_save_path = os.path.join(ckpt_dir, str(global_index)+'_.json')
                    # elif dataset.return_prefix :json_save_path = os.path.join(ckpt_dir, str(global_index)+'_.json') # this case we do not read prefix from a file... ccic prefix does not have this
                    else: json_save_path = os.path.join(ckpt_dir, str(global_index)+'.json')

                    if os.path.exists(json_save_path) and not allow_overwrite_flag:
                        while True:
                            answer = input(f"Model Output Json already exists: {json_save_path}, double check you are not overwritting anything\n").strip().lower()
                            if answer in ['yes', 'y']:
                                allow_overwrite_flag = True
                                break
                            elif answer in ['no', 'n']:
                                exit()
                            else:
                                print("Please enter 'yes/y' or 'no/n'.")
                    save_dir = pathlib.Path(json_save_path)
                    save_dir.parent.mkdir(exist_ok=True, parents=True)
                    to_log = {'predicted': pred_str[i], 'target':label_str[i]}
                    save_json(json_save_path, to_log)
            if self.ccic_model is not None:
                if self.debug:
                    # print('debug 1')
                    pass
                if not self.load_predictions:
                    if self.ccic_model == 'prefix': # Need to remove prefix before save
                        pred_str = self.tokenizer.batch_decode(predictions, skip_special_tokens=False)
                        pred_str = remove_prefix_from_list(pred_str)
                        pred_ids = self.tokenizer(pred_str)['input_ids']
                        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
                    else:
                        pred_str = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
                    for i in range(predictions.shape[0]):
                        dict_key = keys[i]
                        if not self.use_local_model:
                            if 'llava' in self.pretrained_model_name:
                                # in this case the input is predicted together with output, we have to post-process it
                                output_line = pred_str[i].split('\n')[-1]
                                if output_line.startswith('ASSISTANT: '):
                                    pred_str[i] = output_line[11:]
                                else:
                                    pred_str[i] = output_line
                                    print('ASSISTANT NOT FOUND FOR', dict_key)
                            if 'llama' in self.pretrained_model_name:
                                # print('original', pred_str[i])
                                pred_str[i] = pred_str[i].split('\n')[-1]
                                start_index = pred_str[i].find('[/INST]')
    
                                # Check if the token was found
                                if start_index == -1:
                                    pred_str[i] = ""
                                
                                # Calculate the starting position of the substring after the token
                                start_index += len('[/INST]')
                                pred_str[i] = pred_str[i][start_index:].strip().replace('\\"', '"').strip('"')


                        ccic_save_dict[dict_key] = pred_str[i]
                        
                        if self.debug:
                            pass
                            print('predicted', dict_key, pred_str[i])
                            print('++++++++++')
                            print(ccic_save_dict[dict_key])
                            # print(predictions)
                            # print()
                            exit()
                else:
                    dict_key = keys[0]
                    pred_str = [all_predictions[dict_key]]
                if self.debug:
                    pass
                if len(pred_str) == 1 and (self.use_clip_score or self.use_sent_score):
                    
                    with torch.no_grad():
                        text_features = None
                        if self.use_clip_score:
                        
                            clip_text = pred_str[0]

                            if self.debug:
                                # print('clip_text',clip_text)
                                pass
                            text_inputs = self.processor(text=clip_text, max_length=77, truncation=True, return_tensors="pt", padding=True).to('cuda')

                            text_features = self.clip_model.get_text_features(**text_inputs)
                            # print(text_features.shape)

                            if self.use_clip_score:
                                # print(inputs['image'].shape)
                                image_feature = self.clip_model.get_image_features(image_for_clip)

                            # normalized features
                            image_embeds = image_feature / image_feature.norm(p=2, dim=-1, keepdim=True)
                            text_embeds = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

                            # cosine similarity as logits
                            logit_scale = self.clip_model.logit_scale.exp()
                            logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
                            logits_per_image = logits_per_text.t()
                            clip_score = logits_per_image.item() * 2.5
                            if self.debug:
                                print('clip_score', clip_score)  
                                pass 
                            clip_scores.append(clip_score)                         

                        if self.use_sent_score:
                            if text_features is None:
                                clip_text = pred_str[0]
                                text_inputs = self.processor(text=clip_text, return_tensors="pt", padding=True).to('cuda')
                                text_features = self.clip_model.get_text_features(**text_inputs)

                            sents_ids = sent_for_clip['input_ids'][0]
                            sents_mask = sent_for_clip['attention_mask'][0]

                            highlight_features = self.clip_model.get_text_features(input_ids= sents_ids, 
                                                                                   attention_mask=sents_mask)
                            highlight_avg_feat = torch.mean(highlight_features, dim=0).unsqueeze(0)

                            text_embeds = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
                            highlight_avg_embs = highlight_avg_feat / highlight_avg_feat.norm(p=2, dim=-1, keepdim=True)

                            text_similarity = torch.matmul(text_embeds, highlight_avg_embs.t()).item()
                            if self.debug:
                                print('sent_score', text_similarity)
                            highlight_scores.append(text_similarity)

                if self.debug and idx > 100: 
                    print(sum(time_cnt) / len(time_cnt))
                    exit()

                    print(clip_scores)
                    print(highlight_scores)

                    final_clip_score = sum(clip_scores) / len(clip_scores)
                    final_highlight_score = sum(highlight_scores) / len(highlight_scores)

                    print('clip score', final_clip_score)
                    print('highlight_score', final_highlight_score)
                    print('debug exiting in evaluator wrapper')
                    exit()
            
            if not self.load_predictions and self.ccic_model is None:
                pred['label_ids'].append(label_ids)
                pad_amount = label_ids.size(1) - predictions.size(1) + 1
                # print(predictions)
                predictions = torch.nn.functional.pad(predictions, (0, pad_amount), value=self.tokenizer.pad_token_id)

                pred['predictions'].append(predictions)


        if self.ccic_model is not None:
            if self.load_predictions:
                if self.use_clip_score:

                    print(len(clip_scores), len(all_predictions))

                final_clip_score = sum(clip_scores) / len(clip_scores)
                final_highlight_score = sum(highlight_scores) / len(highlight_scores)
                all_predictions['clip_score'] = final_clip_score
                all_predictions['highlight_score'] = final_highlight_score

                print('clip score', final_clip_score)
                print('highlight_score', final_highlight_score)
                save_json(output_save_path, all_predictions)
                exit()

            else:
                if self.use_clip_score:

                    print(len(clip_scores), len(ccic_save_dict))

                    if len(clip_scores) != len(ccic_save_dict):
                        print('unequal length clip scores and entries, need debug')

                exp_dir = os.path.dirname(ckpt_dir)
                # output_save_path = os.path.join(exp_dir, f'ccic_outputs_{len(ccic_save_dict)}.json')
                if self.use_clip_score:
                    ccic_save_dict['clip_score'] = sum(clip_scores) / len(clip_scores)
                    print('clip score', ccic_save_dict['clip_score'])

                if self.use_sent_score:
                    ccic_save_dict['highlight_score'] = sum(highlight_scores) / len(highlight_scores)
                    print('highlight score', ccic_save_dict['highlight_score'])
                   
                from datetime import datetime

                # Get the current time
                now = datetime.now()

                # Format the current time as a string
                current_time = now.strftime('%H:%M')

                # print(current_time)
                output_save_path = os.path.join(exp_dir, f'ccic_outputs_{len(ccic_save_dict)}_{current_time}.json')
                save_json(output_save_path, ccic_save_dict)
                exit()
            return {}

        predictions = torch.cat(pred['predictions'])
        label_ids = torch.cat(pred['label_ids'])

        eval_pred = EvalPrediction(predictions=predictions, label_ids=label_ids)

        metrics = compute_metrics(eval_pred, self.tokenizer, remove_prefix = self.remove_prefix )
        return metrics
        


@Registry.register("Blip2EvaluatorWrapper")
class Blip2EvaluatorWrapper:
    def __init__(self, trainer_class, **kwargs):

        self.trainer_class = import_class_by_full_name(trainer_class)
        self.training_args = TrainingArguments(**kwargs)
        # print(self.training_args)
        # exit()

    def eval(self):
        print("Blip2EvaluatorWrapper.eval() called")
        self.trainer.evaluate()


    def compile(self, dataset, **kwargs):
        print("Blip2EvaluatorWrapper.compile() called")
        tlogging.set_verbosity_error()

        local_rank = comm.get_local_rank()
        processor = AutoProcessor.from_pretrained(kwargs['model']['image_model_name'],
                                        model_max_length=kwargs['model']['image_model_name'])
        ### build dataset

        testset = Registry.build_instance_from_cfg_node(dataset['testset'], **kwargs['model'], 
                                                        prompt = dataset['prompt'], processor = processor)


        ### build model (unfinished)
        model = Registry.build_instance_from_cfg_node(kwargs['model'])
        model.eval()
        model.requires_grad = False

        ### build trainer (unfinished). refer to: https://huggingface.co/docs/transformers/main_classes/trainer
        self.trainer = self.trainer_class( 
                                          eval_dataset=testset, 
                                          model=model,
                                          args=self.training_args,
                                          compute_metrics = compute_metrics,
                                          preprocess_logits_for_metrics = preprocess_logits_for_metrics,
                                          processor = processor)
        self.model = model

