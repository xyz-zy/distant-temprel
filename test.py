import json
import pickle

from utils import convert_examples_to_features
from load_data import *
#from ne import NEReplacer
from transformers import *

#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


def load_exs_data_from_model(model_dir):
    exs = pickle.load(open(model_dir+"/exs.pkl", 'rb'))
    data = pickle.load(open(model_dir+"/data.pkl", 'rb'))
    return exs, data


def add_count(el, count_dict):
    if el in count_dict:
        count_dict[el] += 1
    else:
        count_dict[el] = 1


def sorted_out(d, filename):
    d_sorted = list(d.items())
    d_sorted.sort(key=lambda x: x[1], reverse=True)
    json.dump(d_sorted, open(filename, "w"))


def count_distant_events(distant_exs, out_prefix):
    events = {}
    e1s = {}
    e2s = {}
    for ex in distant_exs:
        e1 = ex.sent1[ex.e1_idx]
        e2 = ex.sent2[ex.e2_idx]
        add_count(e1, events)
        add_count(e2, events)
        add_count(e1, e1s)
        add_count(e2, e2s)
    sorted_out(events, out_prefix+"_events.json")
    sorted_out(e1s, out_prefix+"_e1s.json")
    sorted_out(e2s, out_prefix+"_e2s.json")
    return events, e1s, e2s


def count_labels(exs, file=None):
    labels = {}
    for ex in exs:
        if ex.label in labels:
            labels[ex.label] += 1
        else:
            labels[ex.label] = 1
    print(labels, file=file)
    print("total\t", len(exs), file=file)


def examples_to_csv(exs, filename):
    df = [[ex.text, ex.sent1[ex.e1_idx], ex.sent2[ex.e2_idx], ex.label]
          for ex in exs]
    df = pd.DataFrame(df, columns=["text", "e1", "e2", "label"])
    df.to_csv(filename)
    return df

def check_source(distant_exs, source):
     for ex in distant_exs:
          assert hasattr(ex, "doc_name")
          assert ex.doc_name[:3].lower() == source


def test_load_udst_maj(split="dev"):
    exs, data = udst_majority(tokenizer, lm='roberta', split=split)
    return exs, data


def test_load_udst_dev():
    exs, data = udst(tokenizer, split="dev")
    return exs, data


def test_load_udst_train():
    exs, data = udst(tokenizer, split="train")
    return exs, data


def test_load_udst_dev_maj_conf_nt():
    exs, data = udst(tokenizer, split="dev", example_dir="udst/DecompTime/maj_conf_nt/")
    return exs, data


def test_load_udst_test_maj_conf_nt():
    exs, data = udst(tokenizer, split="test", example_dir="udst/DecompTime/maj_conf_nt/")
    return exs, data


def test_load_matres_train(mask_context=False):
    exs, data = matres_train_examples(tokenizer, mask_context=mask_context)
    return exs, data


def test_load_matres_dev():
    exs, data = matres_dev_examples(tokenizer)
    return exs, data


def test_load_matres_test():
    exs, data = matres_test_examples(tokenizer)
    return exs, data


def test_load_distant_train(source=None, mask_events=False, num_examples=None):
    exs, data = distant_train_examples(
        tokenizer, source=source, mask=True, mask_events=mask_events, num_examples=num_examples)
    return exs, data


def test_load_distant_test(mask=False):
    exs, data = distant_test_examples(tokenizer, mask=mask)
    return exs, data


def test_load_distant_parsed(num_examples=1000, ext='', mask=False, mask_events=False):
    exs, data = distant_parsed_examples(
        tokenizer, ext=ext, num_examples=num_examples, mask=mask, mask_events=mask_events)
    return exs, data


def test_load_beforeafter_yelp(num_examples=1000, mask=False):
    exs, data = beforeafter_examples(tokenizer, ext="_yelp", num_examples=num_examples, mask=mask)
    return exs, data


def test_load_beforeafter_gigaword(num_examples=1000):
    exs, data = beforeafter_examples(tokenizer, num_examples=num_examples)
    return exs, data


def test_beforeafter_mask():
    exs = get_beforeafter_examples("beforeafter/examples/", num_examples=1000)
    fes = convert_examples_to_features(
        examples=exs,
        tokenizer=tokenizer,
        max_seq_length=100,
        doc_stride=128,
        mask='beforeafter')
    return exs, fes

'''
def test_ne_replacement():
    replacer = NEReplacer()
    exs, data = matres_dev_examples(tokenizer)
    old_ex = exs[0]
    new_ex = replacer.replace(old_ex)
    return old_ex, new_ex
'''

