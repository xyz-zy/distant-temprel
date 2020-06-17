import os

import numpy as np

import torch
from torch.utils.data import TensorDataset

# TODO: revise imports 
from transformers import *

class BertForTRC(BertPreTrainedModel):
    
    def __init__(self, config):
        super(BertForTRC, self).__init__(config)
        self.config = config

        self.bert = BertModel(config)
        self.classifier = torch.nn.Linear(4*768, 14)
        self.loss = torch.nn.CrossEntropyLoss()

        self.init_weights()

    def get_positions(self, sequence_output, positions):
        output_tensors = []
        for sample_idx, pos in enumerate(positions):
            position_tensor = sequence_output[sample_idx][pos]
            output_tensors.append(position_tensor)
        return torch.stack(output_tensors, dim=0)

    def eval_sequence_output(self, input_ids, attention_mask=None, token_type_ids=None,
                             tre_labels=None, e1_pos=None, e2_pos=None):
        sequence_output, _ = self.bert(input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids)
        return sequence_output

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, tre_labels=None, e1_pos=None, e2_pos=None):
        #print(input_ids.shape, attention_mask.shape, token_type_ids.shape)
        sequence_output, _ = self.bert(input_ids,
                                       attention_mask=attention_mask,
                                       token_type_ids=None)
        e1_hidden = self.get_positions(sequence_output, e1_pos)
        e2_hidden = self.get_positions(sequence_output, e2_pos)
        el_mul = e1_hidden * e2_hidden
        hidden_diff = (e1_hidden - e2_hidden).abs()
        cls_tensor = torch.cat((e1_hidden,e2_hidden,el_mul, hidden_diff),1)

        out = self.classifier(cls_tensor)
        loss = self.loss(out, tre_labels)
        return loss, out, (e1_hidden, e2_hidden)


class BertForTBD(BertForTRC):
    """
    BERT for TimeBank-Dense
    """
    def __init__(self, config):
        super(BertForTBD, self).__init__(config)
        self.classifier = torch.nn.Linear(4*768, 6)


class BertForMatres(BertForTRC):
    
    def __init__(self, config):
        super(BertForMatres, self).__init__(config)
        self.classifier = torch.nn.Linear(4*config.hidden_size, 4)


class ElectraForMatres(BertPreTrainedModel):
    config_class = ElectraConfig
    retrained_model_archive_map = ELECTRA_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_electra
    base_model_prefix = "electra"

    def __init__(self, config):
        super().__init__(config)

        self.electra = ElectraModel(config)
        self.classifier = torch.nn.Linear(4*config.hidden_size, 4)
        self.loss = torch.nn.CrossEntropyLoss()

    def get_positions(self, sequence_output, positions):
        output_tensors = []
        for sample_idx, pos in enumerate(positions):
            position_tensor = sequence_output[sample_idx][pos]
            output_tensors.append(position_tensor)
        return torch.stack(output_tensors, dim=0)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, tre_labels=None, e1_pos=None, e2_pos=None):
        outputs = self.electra(input_ids=input_ids,
                               token_type_ids=token_type_ids,
                               attention_mask=attention_mask)
        sequence_output = outputs[0]
        e1_hidden = self.get_positions(sequence_output, e1_pos)
        e2_hidden = self.get_positions(sequence_output, e2_pos)
        el_mul = e1_hidden * e2_hidden
        hidden_diff = (e1_hidden - e2_hidden).abs()
        cls_tensor = torch.cat((e1_hidden,e2_hidden,el_mul, hidden_diff),1)
        out = self.classifier(cls_tensor)
        loss = self.loss(out, tre_labels)

        return loss, out, (e1_hidden, e2_hidden)


class RobertaForMatres(BertPreTrainedModel):
    config_class=RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RobertaModel(config)
        self.classifier = torch.nn.Linear(4*config.hidden_size, 4)
        self.loss = torch.nn.CrossEntropyLoss()

        self.init_weights()

    def set_loss_weights(self, weights_list):
        """
        Sets custom loss weights.
        ---
        Ideally should be used to increase the weight of losses on
        rare labels and decrease weight of losses on frequent labels.
        """
        self.train_class_loss_weights=np.array(weights_list)
        self.loss = torch.nn.CrossEntropyLoss(
            weight=torch.from_numpy(
              self.train_class_loss_weights)
                  .float()
        )

    def get_positions(self, sequence_output, positions):
        output_tensors = []
        for sample_idx, pos in enumerate(positions):
            position_tensor = sequence_output[sample_idx][pos]
            output_tensors.append(position_tensor)
        return torch.stack(output_tensors, dim=0)

    def predict(self, input_ids, attention_mask=None,
                             e1_pos=None, e2_pos=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask)
        sequence_output = outputs[0]
        e1_hidden = self.get_positions(sequence_output, e1_pos)
        e2_hidden = self.get_positions(sequence_output, e2_pos)
        el_mul = e1_hidden * e2_hidden
        hidden_diff = (e1_hidden - e2_hidden).abs()
        cls_tensor = torch.cat((e1_hidden,e2_hidden,el_mul, hidden_diff),1)
        out = self.classifier(cls_tensor)
        return out

    def forward(self, input_ids, attention_mask=None, tre_labels=None, e1_pos=None, e2_pos=None):
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask)
        sequence_output = outputs[0]
        e1_hidden = self.get_positions(sequence_output, e1_pos)
        e2_hidden = self.get_positions(sequence_output, e2_pos)
        el_mul = e1_hidden * e2_hidden
        hidden_diff = (e1_hidden - e2_hidden).abs()
        cls_tensor = torch.cat((e1_hidden,e2_hidden,el_mul, hidden_diff),1)
        out = self.classifier(cls_tensor)
        loss = self.loss(out, tre_labels)

        return loss, out, (e1_hidden, e2_hidden)

class InputFeatures(object):
    """Object containing the features for one example/data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 tokens,
                 input_ids,
                 token_type_ids,
                 attention_mask,
                 label,
                 e1_position=None,
                 e2_position=None
                 ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.tokens = tokens
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        
        # Indicates sentence A or sentence B.
        self.token_type_ids = token_type_ids
        self.label = label
        self.e1_position = e1_position
        self.e2_position = e2_position

        
def make_tensor_dataset(train_features, model='roberta'):
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_mask for f in train_features], dtype=torch.long)
    
    if model == 'bert' or model.startswith('electra'):
      all_token_type_ids = torch.tensor([f.token_type_ids for f in train_features], dtype=torch.long)

    all_label_ids = torch.tensor([e.label for e in train_features], dtype=torch.long)
    all_e1_pos = torch.tensor([e.e1_position for e in train_features], dtype=torch.long)
    all_e2_pos = torch.tensor([e.e2_position for e in train_features], dtype=torch.long)
    if model == 'roberta':
      train_data = TensorDataset(all_input_ids, all_attention_masks, all_label_ids, all_e1_pos, all_e2_pos)
    else:
      train_data = TensorDataset(all_input_ids, all_token_type_ids, all_attention_masks, all_label_ids, all_e1_pos, all_e2_pos)
    
    return train_data

def get_tensors(features, model='roberta'):
    #print(model)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    
    if model == 'bert' or model.startswith('electra'):
      all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    all_label_ids = torch.tensor([e.label for e in features], dtype=torch.long)
    all_e1_pos = torch.tensor([e.e1_position for e in features], dtype=torch.long)
    all_e2_pos = torch.tensor([e.e2_position for e in features], dtype=torch.long)
    if model == 'roberta':
      return tuple([all_input_ids, all_attention_masks, all_label_ids, all_e1_pos, all_e2_pos])
    else:
      return tuple([all_input_ids, all_attention_masks, all_token_type_ids, all_label_ids, all_e1_pos, all_e2_pos])


def load_model_and_tokenizer(lm, model_dir=None):
    if lm == 'roberta':
       model_path = model_dir if model_dir else 'roberta-base'
       model = RobertaForMatres.from_pretrained(model_path)
       tokenizer = RobertaTokenizer.from_pretrained(model_path)
    elif lm == 'bert':
       model_path = model_dir if model_dir else 'bert-base-uncased'
       model = BertForMatres.from_pretrained(model_path)
       tokenizer = BertTokenizer.from_pretrained(model_path)
    elif lm == 'bert-large':
       model_path = model_dir if model_dir else 'bert-large-uncased'
       model = BertForMatres.from_pretrained(model_path)
       tokenizer = BertTokenizer.from_pretrained(model_path)
    elif lm == 'electra':
       model_path = model_dir if model_dir else 'google/electra-base-discriminator'
       model = ElectraForMatres.from_pretrained(model_path)
       tokenizer = ElectraTokenizer.from_pretrained(model_path)
    elif lm == 'electra-large':
       model_path = model_dir if model_dir else 'google/electra-large-discriminator'
       model = ElectraForMatres.from_pretrained(model_path)
       tokenizer = ElectraTokenizer.from_pretrained(model_path)
    else:
       raise RuntimeError("Please specify valid model from {bert, bert-large, roberta, electra, electra-large}.")
    return model, tokenizer
 
