import itertools
import torch

from typing import Iterable

import IPython
from IPython.core.display import display, HTML
from IPython.display import Image

from constants import CLASSES

from captum.attr import visualization as viz
from captum.attr import LayerIntegratedGradients


def save_viz(datarecords: Iterable[viz.VisualizationDataRecord]):
    dom = ["<table width: 100%>"]
    rows = [
        "<tr><th>True Label</th>"
        "<th>Predicted Label</th>"
        "<th>Attribution Label</th>"
        "<th>Attribution Score</th>"
        "<th>Word Importance</th>"
    ]
    for datarecord in datarecords:
        rows.append(
            "".join(
                [
                    "<tr>",
                    viz.format_classname(datarecord.true_class),
                    viz.format_classname(
                        "{0} ({1:.2f})".format(
                            datarecord.pred_class, datarecord.pred_prob
                        )
                    ),
                    viz.format_classname(datarecord.attr_class),
                    viz.format_classname("{0:.2f}".format(datarecord.attr_score)),
                    viz.format_word_importances(
                        datarecord.raw_input, datarecord.word_attributions
                    ),
                    "<tr>",
                ]
            )
        )

    dom.append("".join(rows))
    dom.append("</table>")
    return HTML("".join(dom))


def summarize_attributions(attributions):
    '''
    A helper function to summarize attributions for each word token in
    the sequence.
    '''
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions

class Attributor(object):
    def __init__(self, model, tokenizer, device):
        '''
        model: RobertaForMatres model. Will add hooks to model.roberta.embeddings
        '''
        self.model = model
        self.tokenizer = tokenizer
        #self.device = device
        self.device = torch.device("cpu")
        self.model.to(self.device)
        #print(self.device)
        
        self.ref_token_id = tokenizer.mask_token_id # A token used for generating token reference
        self.sep_token_id = tokenizer.sep_token_id # A token used as a separator between question and text and it is also added to the end of the text.
        self.cls_token_id = tokenizer.cls_token_id # A token used for prepending to the concatenated question-text word sequence
    
    def predict(self, input_ids, token_type_ids=None, attention_mask=None, 
            labels=None, e1_pos=None, e2_pos=None):
        '''
        A helper function to perform forward pass of the model and make predictions.
        '''
        #print(input_ids.device, attention_mask.device, labels.device, e1_pos.device, e2_pos.device)
        return self.model(input_ids, #token_type_ids=token_type_ids, 
                     attention_mask=attention_mask, 
                     tre_labels=labels, e1_pos=e1_pos, e2_pos=e2_pos)
    
    def trc_forward_func(self, inputs, token_type_ids=None, attention_mask=None, labels=0, e1_pos=None, e2_pos=None):
        loss, out, _ = self.predict(inputs,
                                    #token_type_ids=token_type_ids,
                                    attention_mask=attention_mask,
                                    labels=labels,
                                    e1_pos=e1_pos,
                                    e2_pos=e2_pos)
        pred = out.max(1).values
        return pred
    
    def _construct_input_ref_pair(self, input_ids):
        # construct reference token ids 
        ref_input_ids = [self.ref_token_id if tok != self.sep_token_id and tok != self.cls_token_id else tok for tok in input_ids]

        return torch.tensor([input_ids], device=self.device), torch.tensor([ref_input_ids], device=self.device)

    def _construct_input_ref_token_type_pair(self, token_type_ids):
        token_type_ids = torch.tensor([token_type_ids], device=self.device)
        ref_token_type_ids = torch.zeros_like(token_type_ids, device=self.device)# * -1
        return token_type_ids, ref_token_type_ids
                
    def _encode(self, example):
        is_same_sentence = example.sent1 == example.sent2

        if is_same_sentence:
            text = example.sent1.copy()
        else:
            text = example.sent1 + example.sent2

        sent1_tok = list(itertools.chain.from_iterable([self.tokenizer.tokenize(w) for w in example.sent1]))
        sent2_tok = list(itertools.chain.from_iterable([self.tokenizer.tokenize(w) for w in example.sent2]))
        if is_same_sentence: 
            inputs = self.tokenizer.encode_plus(sent1_tok, add_special_tokens=True)
        else: 
            inputs = self.tokenizer.encode_plus(sent1_tok, sent2_tok, add_special_tokens=True)
        
        return text, inputs
    
    def get_scores_and_attributions(self, inputs, tok_e1_idx, tok_e2_idx, str_label):
        input_ids, attention_mask = inputs["input_ids"], \
                                    inputs["attention_mask"]
    
        input_ids_tensor, ref_input_ids_tensor = self._construct_input_ref_pair(input_ids)
        #token_type_ids_tensor, ref_token_type_ids_tensor = self._construct_input_ref_token_type_pair(token_type_ids)
        attention_mask_tensor = torch.tensor([attention_mask],device=self.device)
        e1_pos_tensor = torch.tensor([tok_e1_idx], device=self.device)
        e2_pos_tensor = torch.tensor([tok_e2_idx], device=self.device)
        labels_tensor = torch.tensor([CLASSES.index(str_label)], device=self.device)
        

        indices = input_ids_tensor[0].detach().tolist()
        all_tokens = self.tokenizer.convert_ids_to_tokens(indices)

        _, pred_scores, _ = self.predict(input_ids_tensor,
                                         #token_type_ids=token_type_ids_tensor,
                                         attention_mask=attention_mask_tensor,
                                         labels=labels_tensor,
                                         e1_pos=e1_pos_tensor,
                                         e2_pos=e2_pos_tensor)

        lig = LayerIntegratedGradients(self.trc_forward_func, self.model.roberta.embeddings)

        attributions, delta = lig.attribute(inputs=input_ids_tensor,
                                      baselines=ref_input_ids_tensor,
                                      additional_forward_args=(None,#token_type_ids_tensor,
                                                               attention_mask_tensor,
                                                               labels_tensor,
                                                               e1_pos_tensor,
                                                               e2_pos_tensor),
                                      return_convergence_delta=True)

        attributions_sum = summarize_attributions(attributions)

        return pred_scores, all_tokens, attributions_sum, delta
    
    def visualize_attribution(self, example, save=False):
        e1_pos = example.e1_idx
        e2_pos = example.e2_idx
        label = example.label
        is_same_sentence = example.sent1 == example.sent2
        
        text, inputs = self._encode(example)

        input_ids, attention_mask = inputs["input_ids"], \
                                    inputs["attention_mask"]

        # Creates mappings from words in original text to tokens.
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, word) in enumerate(text):
          orig_to_tok_index.append(len(all_doc_tokens))
          tokens = self.tokenizer.tokenize(word)
          for token in tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(token)

        tok_e1_idx = orig_to_tok_index[e1_pos] + 1
        tok_e2_idx = orig_to_tok_index[e2_pos] + 1
        if not is_same_sentence:
          tok_e2_idx += 2

        pred_scores, all_tokens, attributions_sum, delta = self.get_scores_and_attributions(inputs,
                                                                                     tok_e1_idx,
                                                                                     tok_e2_idx,
                                                                                     label)

        # storing couple samples in an array for visualization purposes
        cls_vis = viz.VisualizationDataRecord(
                                word_attributions=attributions_sum,
                                pred_prob=torch.max(torch.softmax(pred_scores[0], dim=0)),
                                pred_class=CLASSES[torch.argmax(pred_scores)],
                                true_class=label,
                                attr_class=CLASSES[torch.argmax(pred_scores)],
                                attr_score=attributions_sum.sum(),       
                                raw_input=all_tokens,
                                convergence_score=delta)

        print('\033[1m', 'Visualizations For Prediction', '\033[0m')
        viz.visualize_text([cls_vis])
        print(example.id)
        print(example.sent1[example.e1_idx])
        print(example.sent2[example.e2_idx])
        if save:
            attributions_per_example[example.id] = AttributionPerExample(example, input_ids, self.tokenizer.convert_ids_to_tokens(input_ids), attributions_sum)

            obj = save_viz([cls_vis])
            with open(model_path + example.id + ".html", "w") as png:
                png.write(obj.data)
    
    def get_span_attributions(self, example, spans):
        e1_pos = example.e1_idx
        e2_pos = example.e2_idx
        label = example.label
        is_same_sentence = example.sent1 == example.sent2
        
        text, inputs = self._encode(example)

        input_ids, attention_mask = inputs["input_ids"], \
                                    inputs["attention_mask"]

        # Creates mappings from words in original text to tokens.
        tok_to_orig_index = []
        orig_to_tok_index = []
        orig_to_tok_index_end = []
        all_doc_tokens = []
        for (i, word) in enumerate(text):
          orig_to_tok_index.append(len(all_doc_tokens))
          tokens = self.tokenizer.tokenize(word)
          for token in tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(token)
          orig_to_tok_index_end.append(len(all_doc_tokens))

        tok_e1_idx = orig_to_tok_index[e1_pos] + 1
        tok_e2_idx = orig_to_tok_index[e2_pos] + 1
        if not is_same_sentence:
          tok_e2_idx += 2

        _, _, attributions, _ = self.get_scores_and_attributions(inputs,
                                                                 tok_e1_idx,
                                                                 tok_e2_idx,
                                                                 label)

        span_attrs = []
        for span_tokens in spans:
            total_attr = 0
            #print(span_tokens)
            for t in span_tokens:
                st = orig_to_tok_index[t]
                nd = orig_to_tok_index_end[t]
                total_attr += sum(attributions[st:nd]).item()
            span_attrs.append(total_attr)
        return span_attrs

