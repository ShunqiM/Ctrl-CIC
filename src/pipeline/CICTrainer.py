import os
import numpy as np
import torch
from torch import nn
from transformers import Trainer
from pydprint import dprint
from utils.utils import sdprint
from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from starter.env_getter import get_env
PREP_DIR = get_env('PREP')


class CustomTrainer(Trainer):

    def __init__(self, **kwargs):
        self.use_cached_feature =  kwargs.pop('use_cached_feature')
        self.use_image_embs = kwargs.pop('use_image_embs')
        # if not self.use_cached_feature:
        self.image_model = kwargs.pop('image_model')
        self.add_input_mask = kwargs.pop('add_input_mask')
        self.mask_as_labels = kwargs.pop('mask_as_labels')
        self.reweight_embs = kwargs.pop('reweight_embs')
        Trainer.__init__(self, **kwargs)

    # Implemented based on the parent class implementation of compute_loss 
    def compute_loss(self, model, inputs, return_outputs=False):

        if self.use_cached_feature:
            assert 'image_feature' in inputs, 'Expected Image Features'
        else:
            assert 'pixel_values' in inputs, 'Expected processed image data'
            pixel_values = inputs.pop('pixel_values')

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        # Hack the model input
        if self.add_input_mask:
            token_weights = inputs.pop('token_weights')
        # outputs = model(**inputs)
        if self.use_cached_feature:
            image_embs = inputs.pop('image_feature')
        else:
            with torch.no_grad():

                image_embs = self.image_model(pixel_values)['pooler_output']

        image_embs = image_embs.unsqueeze(1)

        input_embedding_layer = model.get_input_embeddings()
        input_embeddings = input_embedding_layer(inputs['input_ids'])

        combined_embs = torch.cat((input_embeddings[:, :1, :], image_embs, input_embeddings[:, 1:, :]), dim = 1)

        new_token_mask = torch.ones((combined_embs.shape[0],1), dtype=torch.int64).to(inputs['attention_mask'].device)
        new_attention_mask = torch.cat((new_token_mask, inputs['attention_mask']), dim = 1)

        if self.use_image_embs:
            if self.add_input_mask:
                # needs to add one to mask, in account for the image embs.
                token_weights = torch.cat((new_token_mask.clone(), token_weights), dim = 1)
                if self.mask_as_labels:
                    zero_mask = torch.zeros((combined_embs.shape[0],1), dtype=torch.int64).to(inputs['attention_mask'].device)
                    token_weight_mask = torch.cat((zero_mask.clone(), inputs['token_weight_mask']), dim = 1)
                    outputs = model(inputs_embeds=combined_embs, labels = inputs['labels'], attention_mask = new_attention_mask, token_weights = token_weights, token_weight_mask = token_weight_mask)
                elif self.reweight_embs:
                    combined_embs = combined_embs * token_weights.unsqueeze(-1)
                    outputs = model(inputs_embeds=combined_embs, labels = inputs['labels'], attention_mask = new_attention_mask)
                else:
                    outputs = model(inputs_embeds=combined_embs, labels = inputs['labels'], attention_mask = new_attention_mask, token_weights = token_weights)
            else:
                outputs = model(inputs_embeds=combined_embs, labels = inputs['labels'], attention_mask = new_attention_mask)
        else:
            if self.add_input_mask:
                raise NotImplementedError
            else:
                outputs = model(inputs_embeds=input_embeddings, labels = inputs['labels'], attention_mask = inputs['attention_mask'])
        # End of my edit

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            # print(loss)

        return (loss, outputs) if return_outputs else loss

class MaskPrefixTrainer(Trainer):

    def __init__(self, **kwargs):
        self.use_cached_feature =  kwargs.pop('use_cached_feature')
        self.use_image_embs = kwargs.pop('use_image_embs')
        # if not self.use_cached_feature:
        self.image_model = kwargs.pop('image_model')
        self.add_input_mask = kwargs.pop('add_input_mask')
        self.mask_as_labels = kwargs.pop('mask_as_labels')
        self.reweight_embs = kwargs.pop('reweight_embs')
        self.token_sum = 0
        self.eval_token_sum = 0
        Trainer.__init__(self, **kwargs)
        self.token_loss = torch.tensor(0.0).to(self.args.device)
        self.eval_token_loss = torch.tensor(0.0).to(self.args.device)

    def evaluate(
        self,
        eval_dataset = None,
        ignore_keys = None,
        metric_key_prefix = "eval",
    ):
        self.eval_token_loss -= self.eval_token_loss
        self.eval_token_sum = 0
        super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        token_loss_scalar = self.eval_token_loss.item()
        token_loss_scalar = token_loss_scalar / self.eval_token_sum
        metrics = {'eval_true_loss':round(token_loss_scalar, 4)}
        self.log(metrics)
        self.eval_token_loss -= self.eval_token_loss
        self.eval_token_sum = 0
    
    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            # here I remove code related to TPU

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            # Only code for single process
            token_loss_scalar = self.token_loss.item()
            self.token_loss -= self.token_loss
            logs["token_loss"] = round(token_loss_scalar / self.token_sum, 4)
            self.token_sum = 0
            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    # Implemented based on the parent class implementation of compute_loss 
    def compute_loss(self, model, inputs, return_outputs=False):

        if self.use_cached_feature:
            assert 'image_feature' in inputs, 'Expected Image Features'
        else:
            assert 'pixel_values' in inputs, 'Expected processed image data'
            pixel_values = inputs.pop('pixel_values')

        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None

        prefix_mask = inputs.pop("prefix_mask")

        # Hack the model input
        if self.add_input_mask:
            token_weights = inputs.pop('token_weights')
        # outputs = model(**inputs)
        if self.use_cached_feature:
            image_embs = inputs.pop('image_feature')
        else:
            with torch.no_grad():

                image_embs = self.image_model(pixel_values)['pooler_output']


        image_embs = image_embs.unsqueeze(1)

        input_embedding_layer = model.get_input_embeddings()
        input_embeddings = input_embedding_layer(inputs['input_ids'])

        combined_embs = torch.cat((input_embeddings[:, :1, :], image_embs, input_embeddings[:, 1:, :]), dim = 1)

        new_token_mask = torch.ones((combined_embs.shape[0],1), dtype=torch.int64).to(inputs['attention_mask'].device)
        new_attention_mask = torch.cat((new_token_mask, inputs['attention_mask']), dim = 1)

        if self.use_image_embs:
            if self.add_input_mask:
                raise NotImplementedError
            else:
                outputs, loss_sum, token_sum = model(inputs_embeds=combined_embs, labels = inputs['labels'], attention_mask = new_attention_mask, prefix_mask=prefix_mask)
        else:
            if self.add_input_mask:
                raise NotImplementedError
            else:
                outputs = model(inputs_embeds=input_embeddings, labels = inputs['labels'], attention_mask = inputs['attention_mask'])
        if model.training:
            self.token_loss += loss_sum
            self.token_sum += token_sum
        else:
            self.eval_token_loss += loss_sum
            self.eval_token_sum += token_sum
        # End of my edit

        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            # print(loss)

        return (loss, outputs) if return_outputs else loss

import time
import math
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, Dataset
from transformers.trainer_utils import speed_metrics, EvalLoopOutput, has_length, EvalPrediction, denumpify_detensorize
from transformers.debug_utils import DebugOption
from transformers.utils import logging
from transformers.trainer_pt_utils import (
    find_batch_size, nested_concat, nested_numpify, nested_truncate, nested_detach, IterableDatasetShard)
logger = logging.get_logger(__name__)
from utils.utils_io import save_json


class CustomEvaluator(Trainer):
    def __init__(self, **kwargs):
        self.processor = kwargs.pop('processor')
        self.is_fp_16 = kwargs['args'].fp16
        Trainer.__init__(self, **kwargs)
        print('CustomEvaluator Constructed')


    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0



        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            input_index = 'fake_index'
            if 'index' in inputs.keys():
                input_index = inputs.pop('index').cpu()
            inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None
            
            
            for k, v in inputs.items():
                if k=='pixel_values':
                    if self.is_fp_16: inputs[k] = v.to("cuda", torch.float16)
                    else: inputs[k] = v.to("cuda")
                else:
                    inputs[k] = v.to("cuda")
            with torch.no_grad():
                logits = model.generate(**inputs, max_new_tokens=inputs['labels'].shape[1]).detach()

            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
            loss = torch.tensor(-1)

            # Here I remove the propcessing
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if inputs_decode is not None:
                inputs_decode = self._pad_across_processes(inputs_decode)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                # if self.preprocess_logits_for_metrics is not None:
                #     logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            # preds_host = None
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)
            # Modified
            # # print(logits.shape)
            pred_str = self.processor.batch_decode(logits, skip_special_tokens=True)
            label_str = self.processor.batch_decode(labels, skip_special_tokens=True)
            for i in range(logits.shape[0]):
                json_save_dir = os.path.join(PREP_DIR, 'results_v3', str(input_index[i]))
                to_log = {'predicted': pred_str[i], 'target':label_str[i]}
                save_json(json_save_dir, to_log)
            # exit()
            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None



        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_inputs is not None:
            all_inputs = nested_truncate(all_inputs, num_samples)


        # Metrics
        all_losses = None
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels), tokenizer = self.processor)
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)


        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
