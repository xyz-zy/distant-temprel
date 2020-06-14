import glob
import itertools
import json

## TODO: this code is taken from distant.py
class Example:
    def __init__(self, tokens, e1, e1_pos, e2, e2_pos, label, doc_name=None):
        self.valid = e1 is not None and e1_pos is not None and e2 is not None and e2_pos is not None
        self.tokens = tokens
        self.e1 = e1
        self.e1_pos = e1_pos
        self.e2 = e2
        self.e2_pos = e2_pos
        self.label = label
        self.doc_name = doc_name

    def to_json(self):
        out_obj = {"tokens": self.tokens, "e1_text" : self.e1, "e1_pos": self.e1_pos,
                   "e2_text" : self.e2, "e2_pos" : self.e2_pos, "label" : self.label}
        return json.dumps(out_obj)

    def from_json(json_obj, doc_name=None):
        return Example(json_obj["tokens"],
                       json_obj["e1_text"],
                       json_obj["e1_pos"],
                       json_obj["e2_text"],
                       json_obj["e2_pos"],
                       json_obj["label"],
                       doc_name=doc_name)

    def __repr__(self):
        e1 = self.e1 if self.e1 else "None"
        e1_pos = str(self.e1_pos) if self.e1_pos is not None else "None"
        e2 = self.e2 if self.e2 else "None"
        e2_pos = str(self.e2_pos) if self.e2_pos is not None else "None"

        return str(self.tokens) + "\n (" + e1 + ", " + e1_pos + \
               ") (" + e2+ ", " + e2_pos + ") " + self.label

    def __bool__(self):
        # print(self.valid)
        return self.valid

classes = ["AFTER", "BEFORE", "EQUAL", "VAGUE"]

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
                                 doc_stride, is_training, segment_ids=False, mask=False):
    """Loads a data file into a list of InputFeatures."""

    unique_id = 1000000000

    features = []

    # Generates features from examples.
    for (example_index, example) in enumerate(examples):
        input_tokens = list(itertools.chain.from_iterable([tokenizer.tokenize(w) for w in example.tokens]))

        # Maximum number of tokens that an example may have. This is equal to 
        # the maximum token length less 3 tokens for [CLS], [SEP], [SEP].
        max_tokens_for_doc = max_seq_length - 3

        # Skips this example if it is too long.
        if len(input_tokens) > max_tokens_for_doc:
            unique_id += 1
            continue
 
        if mask:
            rel_idx = input_tokens.index(example.label)
            input_tokens[rel_idx] = "[MASK]"

        # Creates mappings from words in original text to tokens.
        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, word) in enumerate(example.tokens):
            orig_to_tok_index.append(len(all_doc_tokens)) 
            tokens = tokenizer.tokenize(word)
            for token in tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(token)

        # + 1 accounts for CLS token
        tok_e1_pos = orig_to_tok_index[example.e1_pos] + 1
        tok_e2_pos = orig_to_tok_index[example.e2_pos] + 1


        # The -3 accounts for [CLS], [SEP] and [SEP]
        segment = 0

        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(segment)
        for token in input_tokens:
            tokens.append(token)
            segment_ids.append(segment)
            if token == '[SEP]':
                segment += 1

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        # Zero-pads up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert tok_e1_pos < max_seq_length
        assert tok_e2_pos < max_seq_length

        label=-1
        if example.label == "before":
            label = classes.index("BEFORE")
        elif example.label == "after":
            label = classes.index("AFTER")
        elif example.label == "during":
            label = classes.index("EQUAL")

        features.append(
            InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                label=label,
                e1_position=tok_e1_pos,
                e2_position=tok_e2_pos
            )
        )
        unique_id += 1

    return features


def get_examples(EXAMPLE_DIR="examples/", num_examples=None, ratio=False, during=True):
    example_files = glob.glob(EXAMPLE_DIR + "*.json")

    d = 0.04
    a = 0.48
    b = 0.48

    exs = []

    after_exs = []
    before_exs = []
    during_exs = []

    for FILE in example_files:
        print(FILE)
        if not during and "during" in FILE:
            continue
        if ratio:
            if "after" in FILE:
                exs = after_exs
            elif "before" in FILE:
                exs = before_exs
            elif "during" in FILE:
                exs = during_exs
            else:
                continue

        with open(FILE) as file:
            exs_list = json.load(file)

            for ex_json in exs_list:
                example = Example.from_json(ex_json, doc_name=FILE)
                exs.append(example)
        if num_examples and not ratio and len(exs) >= num_examples:
             break
    if ratio:
        if num_examples:
            d = int(d * num_examples)
            a = int(a * num_examples)
            b = int(b * num_examples)
            #print(d, a, b)
            return during_exs[:d] + after_exs[:a] + before_exs[:b]
        else:
            ab_cap = min(len(after_exs), len(before_exs))
            d_cap = int(ab_cap * 0.05)
            #print(ab_cap, d_cap)
            return during_exs[:d_cap] + after_exs[:ab_cap] + before_exs[:ab_cap]
    else:
        return exs[:num_examples]
