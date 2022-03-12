import os
from os.path import dirname, abspath
root_dir = dirname(abspath(__file__))
import sys
sys.path.append(root_dir)
from transformers import BertModel, BertPreTrainedModel
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from losses.focal_loss import FocalLoss
from losses.label_smoothing import LabelSmoothingCrossEntropy


class BertForBinaryClassify(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForBinaryClassify, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_type = config.loss_type
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,labels=None):
        outputs = self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0][:, 0, :]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            assert self.loss_type in ['lsr', 'focal', 'ce', "bce"]
            if self.loss_type == 'lsr':
                loss_fct = LabelSmoothingCrossEntropy(ignore_index=0)
            elif self.loss_type == 'focal':
                loss_fct = FocalLoss(ignore_index=0)
            elif self.loss_type == "ce":
                loss_fct = CrossEntropyLoss()
            elif self.loss_type == "bce":
                loss_fct = BCEWithLogitsLoss()
            # Only keep active parts of the loss
            # if attention_mask is not None:
            #     active_loss = attention_mask.view(-1) == 1
            #     active_logits = logits.view(-1, self.num_labels)[active_loss]
            #     active_labels = labels.view(-1)[active_loss]
            #     loss = loss_fct(active_logits, active_labels)
            # else:
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)
