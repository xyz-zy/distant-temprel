import glob
import json

from utils import IndexedExamplePartial

def get_examples(EXAMPLE_DIR="examples/", num_examples=None, ratio=False, during=False):
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
                example = IndexedExamplePartial.from_json(ex_json, doc_name=FILE)
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
