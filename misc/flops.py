# Compute FLOPs with deepspeed for BERT
# pip install deepspeed transformers

from deepspeed.profiling.flops_profiler import get_model_profile
from functools import partial
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss


# self-customized module
class UM(nn.Module):
    def forward(self, logits):
        return torch.log(F.softmax(logits, dim=-1)) * F.softmax(logits, dim=-1)


class MyBert(BertForSequenceClassification):
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=False,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        uncertainty = torch.log(F.softmax(logits, dim=-1)) * F.softmax(logits, dim=-1)
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
        return ((loss,) + output) if loss is not None else output


def bert_input_constructor(input_shape, tokenizer):
    fake_seq = ""
    for _ in range(input_shape[1] - 2):  # ignore the two special tokens [CLS] and [SEP]
        fake_seq += tokenizer.pad_token
    inputs = tokenizer([fake_seq] * input_shape[0],
                       padding=True,
                       truncation=True,
                       return_tensors="pt")
    labels = torch.tensor([1] * input_shape[0])
    inputs = dict(inputs)
    inputs.update({"labels": labels})
    return inputs


with torch.cuda.device(0):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = MyBert.from_pretrained('bert-base-uncased') # BertForSequenceClassification.from_pretrained('bert-base-uncased')
    batch_size = 4 # Related to FLOPs
    seq_len = 128 # Related to FLOPs
    enable_profile = True
    # following code will print the detailed FLOPs / MACs
    # Note: GFLOPs = 2 * GMACs
    macs, params = get_model_profile(
            model,
            (batch_size, seq_len),
            input_constructor=partial(bert_input_constructor,
                                      tokenizer=tokenizer),
            print_profile=True,
            detailed=True,
        )

