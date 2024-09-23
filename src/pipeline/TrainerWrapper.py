from cli.utils_registry import Registry, import_class_by_full_name
from transformers import logging, TrainingArguments
from comm_ddp import comm

from transformers import AdamW, Adafactor
from transformers import ViTModel, AutoTokenizer
import tensorflow.compat.v1 as tf
import wandb
import os
from starter.env_getter import get_env

CKPT_DIR = get_env('CKPT')

@Registry.register("TrainerWrapper")
class TrainerWrapper:
    def __init__(self, trainer_class, **kwargs):
        self._logger = logging.get_logger('transformers')

        # print(kwargs['output_dir'])
        if kwargs['report_to'] == "wandb":
            if 'resume_from_checkpoint' in kwargs and kwargs['resume_from_checkpoint']:

                wandb.init(project="MMWebpage", id=kwargs['run_name']+ '_resume_2', dir = kwargs['output_dir'])

            else:
                wandb.init(project="MMWebpage", name=kwargs['run_name'], id=kwargs['run_name'], dir = kwargs['output_dir'])

        self.trainer_class = import_class_by_full_name(trainer_class)
        self.training_args = TrainingArguments(**kwargs)
        tf.get_logger().setLevel(3)

    def train(self):
        print("HFTrainerWrapper.train() called")
        if self.resume_from_checkpoint:
            self.trainer.train(self.MODEL_PATH)
        else:
            self.trainer.train()


    def compile(self, dataset, **kwargs):
        print("HFTrainerWrapper.compile() called")

        local_rank = comm.get_local_rank()

        tokenizer_name = kwargs['model']['params']['pretrained_model_name_or_path']
        if 'tokenizer' in kwargs: tokenizer = kwargs['tokenizer']
        else: tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Or should I use add_special_tokens which will never be split instead
        # num_added_tokens = tokenizer.add_tokens(['[ImageToken]', '[SectionIndex]', '[SectionTitle]', '[SectionText]', '[PageURL]', '[PageTitle]', '[ImageCaption]'])
        special_tokens = ['[ImageToken]', '[SectionIndex]', '[SectionTitle]', '[SectionText]', '[PageURL]', '[PageTitle]', '[ImageCaption]']
        if 'extra_special_tokens' in kwargs:
            print(kwargs['extra_special_tokens'])
            # exit()
            special_tokens.extend(kwargs['extra_special_tokens'])
        
        token_dict = {'additional_special_tokens': special_tokens}

        num_added_tokens = tokenizer.add_special_tokens(token_dict)

        if kwargs['resume_name'] is None:
            self.resume_from_checkpoint =  False
        else:
            self.resume_from_checkpoint =  True
            self.MODEL_PATH = os.path.join(CKPT_DIR, 'CIC', 'experiments', kwargs['resume_name'], f"checkpoint-{kwargs['resume_id']}")


        ### build dataset
        if 'add_input_mask' in kwargs: add_input_mask = kwargs['add_input_mask']
        else: add_input_mask = False
        if 'use_image_embs' in kwargs: use_image_embs = kwargs['use_image_embs']
        else: use_image_embs = False
        if 'mask_as_labels' in kwargs: mask_as_labels = kwargs['mask_as_labels']
        else: mask_as_labels = False
        if 'reweight_embs' in kwargs: reweight_embs = kwargs['reweight_embs']
        else: reweight_embs = False
        

        trainset = Registry.build_instance_from_cfg_node(dataset['trainset'], **kwargs['model'], tokenizer = tokenizer, add_input_mask = add_input_mask, mask_as_labels = mask_as_labels)
        validset = Registry.build_instance_from_cfg_node(dataset['validset'], **kwargs['model'], tokenizer = tokenizer, add_input_mask = add_input_mask, mask_as_labels = mask_as_labels) if 'validset' in dataset else None
        self._logger.info(f"[rank:{local_rank}] built trainset: {trainset}")
        self._logger.info(f"[rank:{local_rank}] built validset: {validset}")


        ### build model (unfinished)
        if self.resume_from_checkpoint:

            model_class = kwargs['model']['callable']

            model = model_class.from_pretrained(self.MODEL_PATH)

        else:
            model = Registry.build_instance_from_cfg_node(kwargs['model'])
            model.resize_token_embeddings(len(tokenizer))

        # from transformers import T5Tokenizer
        if kwargs['use_cached_feature']: image_model = None
        else:    
            if kwargs['model']['image_model'] == 'vit': image_model = ViTModel.from_pretrained(kwargs['model']['image_model_name']).cuda()

        ### build trainer (unfinished). refer to: https://huggingface.co/docs/transformers/main_classes/trainer
        self.trainer = self.trainer_class(train_dataset=trainset, 
                                          eval_dataset=validset, 
                                          model=model,
                                          args=self.training_args,
                                          image_model = image_model,
                                          use_cached_feature = kwargs['use_cached_feature'],
                                          add_input_mask = add_input_mask,
                                          use_image_embs = use_image_embs,
                                          mask_as_labels = mask_as_labels,
                                          reweight_embs = reweight_embs)
        self.model = model

if __name__ == '__main__':

    a = TrainerWrapper()
    print(Registry.getter("SimpleTrainer"))