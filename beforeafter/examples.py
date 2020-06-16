import glob
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
