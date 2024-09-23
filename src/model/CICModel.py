import os
from transformers.models.longt5.configuration_longt5 import LongT5Config
from cli.utils_registry import Registry, import_class_by_full_name
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LongT5ForConditionalGeneration, T5ForConditionalGeneration, Blip2ForConditionalGeneration, LongT5EncoderModel, LlavaForConditionalGeneration
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.modeling_utils import PreTrainedModel

# refer to https://huggingface.co/models?sort=trending&search=google%2Flongt5 for avaliable models
@Registry.register("LongT5")
def LongT5(**kwargs):
    return LongT5ForConditionalGeneration.from_pretrained(**kwargs)

@Registry.register("T5")
def T5(**kwargs):
    return T5ForConditionalGeneration.from_pretrained(**kwargs)

@Registry.register("Blip2")
def Blip2(**kwargs):
    if kwargs.pop('torch_dtype') == "torch.float16":
        return Blip2ForConditionalGeneration.from_pretrained(torch_dtype=torch.float16, **kwargs)
    else:
        return Blip2ForConditionalGeneration.from_pretrained(**kwargs)
    
@Registry.register("Llama")
def Llama(**kwargs):
    if kwargs.pop('torch_dtype') == "torch.float16":
        print(1)
        exit()
    else:
        print(2)
        exit()
    
@Registry.register("Qwen")
def Qwen(**kwargs):
    if kwargs.pop('torch_dtype') == "torch.float16":
        print(1)
        exit()
        return LlavaForConditionalGeneration.from_pretrained(torch_dtype=torch.float16, **kwargs)
    else:
        print(2)
        exit()
        return LlavaForConditionalGeneration.from_pretrained(**kwargs)


@Registry.register("Llava")
def Llava(**kwargs):
    if kwargs.pop('torch_dtype') == "torch.float16":
        print(1)
        exit()
        return LlavaForConditionalGeneration.from_pretrained(torch_dtype=torch.float16, **kwargs)
    else:
        print(2)
        exit()
        return LlavaForConditionalGeneration.from_pretrained(**kwargs)


# dont need this
@Registry.register("TGAttention")
class TGAttention(nn.Module):
    def __init__(self, **kwargs):
        super(TGAttention, self).__init__()
        
        print('TGAttention called')
        
    
# Do everthing on input_embings so I don't have change the inner model specifics.
@Registry.register("PretrainedMaskProjectionLongT5")
def get_mask_projection_longt5(**kwargs):
    return MaskProjectionLongT5.from_pretrained(**kwargs)

@Registry.register("MaskProjectionLongT5")
class MaskProjectionLongT5(LongT5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        
        embedding_dim = self.get_input_embeddings().embedding_dim
        self.weight_projection = torch.nn.Linear(1, embedding_dim)
        
    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, token_weights = None, labels=None, **kwargs):
        if kwargs['inputs_embeds'] is None:
            raise NotImplementedError
        else:
            inputs_embeds = kwargs.pop('inputs_embeds')

        if token_weights is None:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels = labels, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, **kwargs)
        else:
            # assume input will only be b, seq, emb, token weight will be b, seq
            token_weights.unsqueeze_(-1)
            # print(token_weights.shape)
            token_weights = self.weight_projection(token_weights)
            # print(token_weights.shape)
            # print(inputs_embeds.shape)
            inputs_embeds = inputs_embeds + token_weights
            # exit()
            outputs = super().forward(inputs_embeds = inputs_embeds, input_ids=input_ids, labels = labels, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, **kwargs)
            
            return outputs
    
    def generate(self, token_weight = None, **kwargs):
        if kwargs['inputs_embeds'] is None:
            raise NotImplementedError
        else:
            inputs_embeds = kwargs.pop('inputs_embeds')
        if token_weight is None:
            return super().generate(**kwargs)
        else:
            # assume input will only be b, seq, emb, token weight will be b, seq
            token_weights.unsqueeze_(-1)
            # print(token_weights.shape)
            token_weights = self.weight_projection(token_weights)
            # print(token_weights.shape)
            # print(inputs_embeds.shape)
            inputs_embeds = inputs_embeds + token_weights
            # exit()
            outputs = super().generate(inputs_embeds = inputs_embeds, **kwargs)
            return outputs
            

@Registry.register("PretrainedMaskAsTokenLongT5")
def get_mask_as_token_longt5(**kwargs):
    return MaskAsTokenLongT5.from_pretrained(**kwargs)

@Registry.register("MaskAsTokenLongT5")
class MaskAsTokenLongT5(LongT5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        
        embedding_dim = self.get_input_embeddings().embedding_dim
        self.weight_projection = torch.nn.Linear(512, embedding_dim)
        
    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, token_weights = None, labels=None, **kwargs):
        if kwargs['inputs_embeds'] is None:
            raise NotImplementedError
        else:
            inputs_embeds = kwargs.pop('inputs_embeds')

        if token_weights is None:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels = labels, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, **kwargs)
        else:
            # assume input will only be b, seq, emb, token weight will be b, seq
            # print(token_weights.shape)
            token_weights = self.weight_projection(token_weights)
            token_weights = token_weights.unsqueeze(1)
            # print(token_weights.shape)
            # print(inputs_embeds.shape)
            inputs_embeds = torch.cat([token_weights, inputs_embeds], dim = 1)
            # print(inputs_embeds.shape)
            new_token_mask = torch.ones((inputs_embeds.shape[0],1), dtype=torch.int64).to(attention_mask.device)
            # print('new', new_token_mask)
            attention_mask = torch.cat((new_token_mask, attention_mask), dim = 1)
            # print(attention_mask.shape)
            # exit()
            outputs = super().forward(inputs_embeds = inputs_embeds, input_ids=input_ids, labels = labels, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, **kwargs)
            
            return outputs
    
    def generate(self, token_weight = None, **kwargs):
        if kwargs['inputs_embeds'] is None:
            raise NotImplementedError
        else:
            inputs_embeds = kwargs.pop('inputs_embeds')
        if token_weight is None:
            return super().generate(**kwargs)
        else:
            token_weights = self.weight_projection(token_weights)
            token_weights = token_weights.unsqueeze(1)

            inputs_embeds = torch.cat([token_weights, inputs_embeds], dim = 1)
            new_token_mask = torch.ones((inputs_embeds.shape[0],1), dtype=torch.int64).to(attention_mask.device)
            attention_mask = torch.cat((new_token_mask, attention_mask), dim = 1)
            outputs = super().generate(inputs_embeds = inputs_embeds, attention_mask=attention_mask, **kwargs)
            return outputs
            
@Registry.register("PretrainedReWeightLongT5")
def get_reweight_longt5(**kwargs):
    return ReWeightLongT5.from_pretrained(**kwargs)

import warnings

@Registry.register("ReWeightLongT5")
class ReWeightLongT5(LongT5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    # This is needed for preperving token weights
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        token_weights = None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "token_weights": token_weights
        }
        
    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, token_weights = None, labels=None, inputs_embeds = None, **kwargs):
        
        use_cache, return_dict = None, None
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        head_mask = None
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn("HEAD MASK WARNING TRIGGERED", FutureWarning)
                decoder_head_mask = head_mask

        if 'encoder_outputs' in kwargs and kwargs['encoder_outputs'] is not None:
            encoder_outputs = kwargs['encoder_outputs']
        else:
            encoder_outputs = None
        

        # During generate, encoder outputs will be prepared by the framework and inputs_embeds will be popped
        if inputs_embeds is None:
            if encoder_outputs is None:
                raise NotImplementedError
            else:
                if token_weights is None:
                    print('note that no token weights are used')
                    return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels = labels, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, **kwargs)
                else:
                    # print(333, token_weights.shape)
                    token_weights = token_weights.unsqueeze(2)

                    encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state * token_weights
                    kwargs.pop('encoder_outputs')
                    outputs = super().forward(encoder_outputs = encoder_outputs, input_ids=input_ids, labels = labels, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, **kwargs)
                        
                    return outputs

        if token_weights is None:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels = labels, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, **kwargs)
        else:
            
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=return_dict,
            )
            token_weights = token_weights.unsqueeze(2)

            encoder_outputs.last_hidden_state = encoder_outputs.last_hidden_state * token_weights

            outputs = super().forward(encoder_outputs = encoder_outputs, input_ids=input_ids, labels = labels, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, **kwargs)
            
            return outputs
        

@Registry.register("PretrainedMaskedLongT5")
def get_masked_longt5(t, **kwargs):
    model = MaskedLongT5.from_pretrained(t=t, **kwargs)
    return model


@Registry.register("MaskedLongT5")
class MaskedLongT5(LongT5ForConditionalGeneration):
    def __init__(self, config):
        self.t = 0.6
        super().__init__(config)

    # This is needed for preperving token weights
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        token_weights = None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "token_weights": token_weights
        }
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, t, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        model.t = t
        print("Threshold Loaded:", t)
        return model
    
    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, token_weights = None, labels=None, inputs_embeds = None, **kwargs):
        
        use_cache, return_dict = None, None
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        head_mask = None
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn("HEAD MASK WARNING TRIGGERED", FutureWarning)
                decoder_head_mask = head_mask

        if 'encoder_outputs' in kwargs and kwargs['encoder_outputs'] is not None:
            encoder_outputs = kwargs['encoder_outputs']
        else:
            encoder_outputs = None
        

        if inputs_embeds is None:
            if encoder_outputs is None:
                raise NotImplementedError
            else:
                if token_weights is None:
                    print('note that no token weights are used')
                    return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels = labels, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, **kwargs)
                else:
                    weight_mask = token_weights >= self.t
                    combined_mask = torch.logical_and(weight_mask, attention_mask)

                    kwargs.pop('encoder_outputs')
                    outputs = super().forward(encoder_outputs = encoder_outputs, input_ids=input_ids, labels = labels, attention_mask=combined_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, **kwargs)
                        
                    return outputs

        if token_weights is None:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, labels = labels, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, **kwargs)
        else:
            # assume input will only be b, seq, emb, token weight will be b, seq

            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=return_dict,
            )
            weight_mask = token_weights >= self.t
            combined_mask = torch.logical_and(weight_mask, attention_mask)

            outputs = super().forward(encoder_outputs = encoder_outputs, input_ids=input_ids, labels = labels, attention_mask=combined_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, **kwargs)
            
            return outputs
        
@Registry.register("PretrainedLongT5forTokenRegression")
def get_token_regression_longt5(loss = 'cross_entropy', **kwargs):
    model = LongT5forTokenRegression.from_pretrained(**kwargs)
    model.loss = loss
    return model
    
@Registry.register("LongT5forTokenRegression")
class LongT5forTokenRegression(LongT5EncoderModel):
    def __init__(self, config):
        super(LongT5forTokenRegression, self).__init__(config)
        self.regression_head = torch.nn.Linear(config.d_model, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.loss = 'cross_entropy'

    def forward(self, input_ids=None, inputs_embeds = None, attention_mask=None, token_weights = None, token_weight_mask=None, decoder_input_ids=None, decoder_attention_mask=None, labels=None):
            
        encoder_outputs = self.encoder.forward(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask)

        logits = self.regression_head(encoder_outputs.last_hidden_state) # b, s, 1

        logits = logits * token_weight_mask.unsqueeze(-1)
        if labels is not None:
            labels = token_weights * token_weight_mask


        scores = self.sigmoid(logits)
        scores = scores.squeeze(2)
        scores = scores * token_weight_mask
        if labels is None:
            return {'scores': scores}
        flatten_labels = labels.reshape(-1, 1)
        flatten_scores = scores.reshape(-1, 1)
        if self.loss == 'cross_entropy':
            # loss = F.binary_cross_entropy (flatten_scores, flatten_labels, reduction='mean')
            loss = F.binary_cross_entropy(flatten_scores, flatten_labels, reduction="none")


            loss = torch.sum(loss) / (torch.sum(token_weight_mask).float() + 1e-8)
            
        elif self.loss == 'mse': 
            loss = F.mse_loss(flatten_scores, flatten_labels, reduction="none")

            loss = loss * token_weight_mask.view(loss.shape)
            loss = torch.sum(loss) / (torch.sum(token_weight_mask).float() + 1e-8)

        return {'loss': loss, 'scores': scores}


@Registry.register("TwoStageLongT5")
class TwoStageLongT5(ReWeightLongT5):
    def __init__(self, config):
        super().__init__(config)
        self.mask_regressor = None
        self.ccic_mode = None
        self.highlight_val = 0.7

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, base_model_path = None, base_model_index = None, ccic_mode = None, highlight_val = 0.7, **kwargs):

        if base_model_path is None:
            # Not sure if this will work, we can test it
            raise NotImplementedError
            model = super().from_pretrained(pretrained_model_name_or_path)
            model.mask_regressor = LongT5forTokenRegression.from_pretrained(pretrained_model_name_or_path)

        else:
            base_model_path = os.path.join(base_model_path, 'checkpoint-' + str(base_model_index))
            model = super().from_pretrained(base_model_path, **kwargs)
            model.mask_regressor = LongT5forTokenRegression.from_pretrained(pretrained_model_name_or_path)
            print(f"Loaded Pretrained Two Stage Model, Base from {base_model_path}, mask predictor from {pretrained_model_name_or_path}")
            model.ccic_mode = ccic_mode
            model.highlight_val = highlight_val
        return model

    
    def generate(self,  input_ids=None, inputs_embeds=None, max_new_tokens=None, token_weight_mask = None, attention_mask = None, token_weights = None, **kwargs):
        if 'labels' in kwargs:
            labels = kwargs.pop('labels')
        else:
            labels = None
        # print(token_weight_mask)
        if self.ccic_mode == None:
            # The token weight here would not leak label information as the unmasked values are hardcoded
            predicted_weights = self.mask_regressor(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask = attention_mask, token_weight_mask=token_weight_mask)['scores'].detach()
            mask_indices = token_weight_mask == 1
            # print(mask_indices)
            token_weights[mask_indices] = predicted_weights[mask_indices]
            # How to overwrite to the correct token indices?

            generation_output = super().generate(token_weights = token_weights, inputs_embeds=inputs_embeds, attention_mask = attention_mask, max_new_tokens=max_new_tokens, **kwargs)
            # print(generation_output)
            return generation_output
        elif self.ccic_mode == 'fixed':
            generation_output = super().generate(token_weights = token_weights, inputs_embeds=inputs_embeds, attention_mask = attention_mask, max_new_tokens=max_new_tokens, **kwargs)
            return generation_output
        elif self.ccic_mode == 'overwrite':
            predicted_weights = self.mask_regressor(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask = attention_mask, token_weight_mask=token_weight_mask)['scores'].detach()
            mask_indices = (token_weights != self.highlight_val) & (token_weight_mask == 1)
            token_weights[mask_indices] = predicted_weights[mask_indices]

            generation_output = super().generate(token_weights = token_weights, inputs_embeds=inputs_embeds, attention_mask = attention_mask, max_new_tokens=max_new_tokens, **kwargs)
            # print(generation_output)
            return generation_output
        elif self.ccic_mode == 'add':
            predicted_weights = self.mask_regressor(input_ids=input_ids, inputs_embeds=inputs_embeds, attention_mask = attention_mask, token_weight_mask=token_weight_mask)['scores'].detach()
            mask_indices = token_weight_mask == 1
            increment_mask = token_weights == self.highlight_val
            token_weights[mask_indices] = predicted_weights[mask_indices]
            increment_val = self.highlight_val - 0.5

            token_weights[increment_mask] += increment_val

            generation_output = super().generate(token_weights = token_weights, inputs_embeds=inputs_embeds, attention_mask = attention_mask, max_new_tokens=max_new_tokens, **kwargs)
            # print(generation_output)
            return generation_output
        else:
            assert "unrecognised ccic mode"

@Registry.register("PretrainedMaskPrefixLongT5")
def get_mask_prefix_longt5(**kwargs):
    model = MaskPrefixLongT5.from_pretrained(**kwargs)
    return model

from torch.nn import CrossEntropyLoss
@Registry.register("MaskPrefixLongT5")
class MaskPrefixLongT5(LongT5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    # This is needed for preperving token weights
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        prefix_mask = None,
        **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
            "prefix_mask": prefix_mask
        }
    
    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, prefix_mask = None, labels=None, inputs_embeds = None, **kwargs):
        
        if labels is None:
            return super().forward(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, labels=labels, inputs_embeds=inputs_embeds, **kwargs)
        else:
            if prefix_mask is None:
                raise NotImplementedError
            outputs = super().forward(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask, labels=labels, inputs_embeds=inputs_embeds, **kwargs)
            loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='none')
            lm_logits = outputs.logits
            labels = labels.to(lm_logits.device)
            with torch.no_grad():
                loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
                true_token_mask = 1 - prefix_mask
                masked_loss = loss * true_token_mask.view(loss.shape)

                # Calculate the sum of unmasked losses along the appropriate dimension (usually dim=1 for sequence length)
                sum_loss = masked_loss.sum()
            # sum_loss = masked_loss.sum(dim=1)
            # average_loss = sum_loss/true_token_mask.sum(dim=1)
            return outputs, sum_loss, true_token_mask.sum().item()