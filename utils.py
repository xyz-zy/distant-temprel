import itertools
import random
import re

#from transformers import RobertaTokenizer

from constants import CLASSES

class IndexedExamplePartial(object):
    def __init__(self, label, sent1, sent2, tags1, tags2, e1_idx, e2_idx, doc_name=None):
        self.label = label
        self.sent1 = sent1
        self.sent2 = sent2
        self.tags1 = tags1
        self.tags2 = tags2
        self.e1_idx = int(e1_idx)
        self.e2_idx = int(e2_idx)
        self.doc_name = doc_name

    def __repr__(self):
        return str(self.sent1) + "\n" + str(self.sent2) + "\n" + str(self.e1_idx) + "  " + str(self.e2_idx) + "\n" + self.sent1[self.e1_idx] + "  " + self.sent2[self.e2_idx]

    def from_json(json_obj, doc_name=None):
        return IndexedExamplePartial(label=json_obj["label"],
                                     sent1=json_obj["tokens"],
                                     sent2=json_obj["tokens"],
                                     tags1=None, tags2=None,
                                     e1_idx=json_obj["e1_pos"],
                                     e2_idx=json_obj["e2_pos"],
                                     doc_name=doc_name)

class InputFeatures(object):
    """Object containing the features for one example/data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 tokens,
                 input_ids,
                 attention_mask,
                 token_type_ids,
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

    def __repr__(self):
        return str(self.tokens) + "\n" + str(self.e1_position) + ", " + str(self.e2_position) + ", " + str(self.label)


def convert_examples_to_features(examples, tokenizer, max_seq_length,   
                                 doc_stride, mask=False,                
                                 mask_events=False, mask_context=False,
                                 id_prefix=None):
    """Loads a data file into a list of InputFeatures."""                       

    unique_id = 1

    features = []
    processed_examples = []
    # Generates features from examples.
    for (example_index, example) in enumerate(examples):                        
        is_same_sentence = example.sent1 == example.sent2                       
        sent1 = example.sent1.copy()
        sent2 = example.sent2.copy()
        if mask or mask_events or mask_context:                                 
            if mask == 'beforeafter':
                sent1 = generate_explicit_mask(
                    example.sent1, example.label, tokenizer)
                sent2 = generate_explicit_mask(
                    example.sent2, example.label, tokenizer)
            elif mask:
                sent1 = generate_timex_mask(
                    example.sent1, example.tags1, tokenizer)
                sent2 = generate_timex_mask(
                    example.sent2, example.tags2, tokenizer)
            if mask_events:
                sent1[example.e1_idx] = tokenizer.mask_token
                if is_same_sentence:
                    sent1[example.e2_idx] = tokenizer.mask_token
                sent2[example.e2_idx] = tokenizer.mask_token
            elif mask_context:
                sent1 = [tokenizer.mask_token] * len(example.sent1)
                sent1[example.e1_idx] = example.sent1[example.e1_idx]
                if is_same_sentence:
                    sent1[example.e2_idx] = example.sent2[example.e2_idx]
                sent2 = [tokenizer.mask_token] * len(example.sent2)
                sent2[example.e2_idx] = example.sent2[example.e2_idx]
        sent1_tokens = list(itertools.chain.from_iterable(
            [tokenizer.tokenize(w) for w in sent1]))
        sent2_tokens = list(itertools.chain.from_iterable(
            [tokenizer.tokenize(w) for w in sent2]))
        example.text = sent1.copy() if is_same_sentence else sent1 + sent2

        if is_same_sentence:
            inputs = tokenizer.encode_plus(
                sent1_tokens, add_special_tokens=True, pad_to_max_length=True, max_length=max_seq_length)
        else:
            inputs = tokenizer.encode_plus(
                sent1_tokens, sent2_tokens, max_length=max_seq_length, add_special_tokens=True, pad_to_max_length=True)
        input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]
        if 'token_type_ids' in inputs:
            token_type_ids = inputs['token_type_ids']
            assert len(token_type_ids) == max_seq_length
        else:
            token_type_ids = None

        example.tokens = tokenizer.convert_ids_to_tokens(input_ids)
        max_seq_length = len(input_ids)
        # print(max_seq_length)
        # Maximum number of tokens that an example may have. This is equal to
        # the maximum token length less 3 tokens for [CLS], [SEP], [SEP].
        max_tokens_for_doc = max_seq_length - 5

        # Skips this example if it is too long.
        if len(sent1_tokens + sent2_tokens) > max_tokens_for_doc:
            unique_id += 1
            continue

        e1_pos = example.e1_idx
        e2_pos = example.e2_idx if is_same_sentence else example.e2_idx + \
            len(example.sent1)

        # Creates mappings from words in original text to tokens.
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, word) in enumerate(example.text):
            orig_to_tok_index.append(len(all_doc_tokens))
            tokens = tokenizer.tokenize(word)
            for token in tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(token)

        example.tok_e1_idx = orig_to_tok_index[e1_pos] + 1
        example.tok_e2_idx = orig_to_tok_index[e2_pos] + 1
        if not is_same_sentence:
            example.tok_e2_idx += 2

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert example.tok_e1_idx < max_seq_length
        assert example.tok_e2_idx < max_seq_length

        label = example.label.upper()
        label = "EQUAL" if label == "SIMULTANEOUS" or label == "DURING" else label
        if label == 'IS_INCLUDED':
            label = 'AFTER'
        if label == 'INCLUDES':
            label = 'BEFORE'
        # if label in ['IS_INCLUDED', 'INCLUDES']:
        #   continue
        label = CLASSES.index(label)

        features.append(
            InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                tokens=example.tokens,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                label=label,
                e1_position=example.tok_e1_idx,
                e2_position=example.tok_e2_idx
            )
        )
        if id_prefix:
            example.id = id_prefix + str(unique_id)
        unique_id += 1
        processed_examples.append(example)
    return processed_examples, features

def generate_explicit_mask(sent, label, tokenizer):
    masked_sent = sent.copy()
    rel_idx = sent.index(label)
    masked_sent[rel_idx] = tokenizer.mask_token
    return masked_sent


def generate_timex_mask(sent, tags, tokenizer):
    masked_sent = sent.copy()
    for i, tag in enumerate(tags):
        if tag and tag[0] == 't':
            masked_sent[i] = tokenizer.mask_token
    return masked_sent


def apply_random_mask_to_sentence(tokenizer, sentence, event_idx):
    mask_idx = random.randint(0, len(sentence)-1)
    while mask_idx == event_idx:
        mask_idx = random.randint(0, len(sentence)-1)
    sent = sentence.copy()
    sent[mask_idx] = tokenizer.mask_token
    return sent


def apply_random_mask(examples, tokenizer, threshold=0.5):
    new_examples = []
    for ex in examples:
        # new_examples.append(ex)
        if random.random() <= threshold:
            ex_copy = copy.copy(ex)
            sent_num = random.randrange(0, 1)
            if ex.sent1 == ex.sent2 or sent_num == 0:
                ex_copy.sent1 = apply_random_mask_to_sentence(
                    tokenizer, ex.sent1, ex.e1_idx)
                ex_copy.sent2 = ex.sent2.copy()
            else:
                ex_copy.sent2 = apply_random_mask_to_sentence(
                    tokenizer, ex.sent2, ex.e2_idx)
                ex_copy.sent1 = ex.sent1.copy()
            new_examples.append(ex_copy)
        else:
            new_examples.append(ex)
    return new_examples


class ExampleWithEmbeddings(object):
    def __init__(self, example, e1_hidden, e2_hidden, label, guess):
        self.example = example
        self.e1_hidden = e1_hidden
        self.e2_hidden = e2_hidden
        self.label = label
        self.guess = guess


def count_labels(exs, file=None):
    labels = {}
    for ex in exs:
        if ex.label in labels:
            labels[ex.label] += 1
        else:
            labels[ex.label] = 1
    print(labels, file=file)
    print("total\t", len(exs), file=file)

