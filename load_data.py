import copy
import glob
import json
import pickle
import sys

from constants import *
from modeling import make_tensor_dataset
from utils import IndexedExamplePartial
from utils import convert_examples_to_features, apply_random_mask

from transformers import RobertaTokenizer

from udst import parse_udst
from timebank.examples import MatresLoader

sys.path.insert(1, 'Temporal-event-ordering/event_model')

def get_beforeafter_examples(EXAMPLE_DIR="beforeafter_examples/", num_examples=None, ratio=False, during=False):
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


def beforeafter_examples(tokenizer, lm='roberta', ext='', num_examples=None, mask=False, during=True):
    g_train_examples = get_beforeafter_examples(
        EXAMPLE_DIR="beforeafter/examples" + ext + "/", num_examples=num_examples, during=during)
    if mask:
        mask = 'beforeafter'
        print(mask)
    g_train_features = convert_examples_to_features(
        examples=g_train_examples,
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
        doc_stride=DOC_STRIDE,
        mask='beforeafter')
    g_train_data = make_tensor_dataset(g_train_features, model=lm)

    return g_train_examples, g_train_data


def matres_examples():
    loader = MatresLoader()
    train_examples, dev_examples = loader.read_train_dev_examples(
        doc_dir="timebank/TBAQ-cleaned/", rel_dir="timebank/MATRES/")

    return train_examples, dev_examples


def matres_train_examples(tokenizer, lm='roberta', train=False, mask_events=False, mask_context=False):
    train_examples, _ = matres_examples()

    train_features = convert_examples_to_features(
        examples=train_examples,
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
        doc_stride=DOC_STRIDE,
        mask_events=mask_events,
        mask_context=mask_context)
    train_data = make_tensor_dataset(train_features, model=lm)
    return train_examples, train_data


def matres_dev_examples(tokenizer, lm='roberta', mask_events=False, mask_context=False):
    _, dev_examples = matres_examples()

    dev_features = convert_examples_to_features(
        examples=dev_examples,
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
        doc_stride=DOC_STRIDE,
        mask_events=mask_events,
        mask_context=mask_context,
        id_prefix="md")
    dev_data = make_tensor_dataset(dev_features, model=lm)
    return dev_examples, dev_data

def matres_test_examples(tokenizer, lm='roberta', mask_events=False, mask_context=False):
    loader = MatresLoader()
    examples = loader.read_test_examples(doc_dir="timebank/te3-platinum/", rel_dir="timebank/MATRES/")
 
    features = convert_examples_to_features(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LENGTH,
        doc_stride=DOC_STRIDE,
        mask_events=mask_events,
        mask_context=mask_context,
        id_prefix="mt")
    data = make_tensor_dataset(features, model=lm)
    return examples, data


def filter_distant_source(exs, num_examples=None, source=None):
     if source is None:
         return exs
     else:
         exs = [e for e in exs if hasattr(e, 'doc_name') and e.doc_name[:3].lower() == source]
         if num_examples:
             num_examples = int(num_examples)
             exs = exs[:num_examples]
         return exs


def distant_train_examples(tokenizer, lm='roberta', source=None, ext='', num_examples=None, mask=False, random_mask=False, mask_events=False):
    f = open('Temporal-event-ordering/event_model/train_data.pkl', 'rb')
    train_examples = pickle.load(f)
    if source == "even":
      train_examples = filter_distant_source(train_examples, num_examples=num_examples/6, source="afp")
    else:
      train_examples = filter_distant_source(train_examples, source)
    if num_examples and num_examples > len(train_examples):
        more_examples = _distant_parsed_examples(
            tokenizer, source=source, ext='', num_examples=num_examples-len(train_examples))
        train_examples += more_examples
        train_examples = train_examples[:num_examples]
    if random_mask:
        train_examples = apply_random_mask(train_examples, tokenizer)
    if mask:
        mask = 'distant'
    data = convert_examples_to_features(examples=train_examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=MAX_SEQ_LENGTH,
                                                doc_stride=DOC_STRIDE,
                                                mask=mask,
                                                mask_events=mask_events)
    data = make_tensor_dataset(data, model=lm)
    return train_examples, data


def distant_test_examples(tokenizer, lm='roberta', train=False, mask=False, mask_events=False):
    f = open('Temporal-event-ordering/event_model/test_data.pkl', 'rb')
    test_examples = pickle.load(f)
    if mask:
        mask = 'distant'
    data = convert_examples_to_features(examples=test_examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=MAX_SEQ_LENGTH,
                                                doc_stride=DOC_STRIDE,
                                                mask=mask,
                                                mask_events=mask_events)
    data = make_tensor_dataset(data, model=lm)
    return test_examples, data


def _distant_parsed_examples(tokenizer, source=None, ext='', num_examples=None):
    assert source is None or source in set(['nyt', 'apw', 'afp', 'cna', 'wpb', 'xin', 'even'])
    if source == 'even':
        #afp_exs = _distant_parsed_examples(tokenizer, source='afp', num_examples=num_examples/6)
        apw_exs = _distant_parsed_examples(tokenizer, source='apw', num_examples=num_examples/5)
        cna_exs = _distant_parsed_examples(tokenizer, source='cna', num_examples=num_examples/5)
        nyt_exs = _distant_parsed_examples(tokenizer, source='nyt', num_examples=num_examples/5)
        wpb_exs = _distant_parsed_examples(tokenizer, source='wpb', num_examples=num_examples/5)
        xin_exs = _distant_parsed_examples(tokenizer, source='xin', num_examples=num_examples/5)
        exs = apw_exs + cna_exs + nyt_exs + wpb_exs + xin_exs
        if len(exs) < num_examples:
            print("Only", len(exs), "could be parsed with even split")
            #exs += _distant_parsed_examples(tokenizer, source='afp', num_examples=num_examples-len(exs))
            #if len(exs) < num_examples:
            nyt_exs = _distant_parsed_examples(tokenizer, source='nyt', num_examples=num_examples-len(exs))
        return exs

    allowed_labels = set(
        ['AFTER', 'BEFORE', 'IS_INCLUDED', 'INCLUDES', 'SIMULTANEOUS'])
    exs = []
    exs_dir = "Temporal-event-ordering/download_data/extracted" + ext + "/*"
    for filename in sorted(glob.glob(exs_dir)):
        print(filename)
        with open(filename) as file:
            line = file.readline()
            while line and (not num_examples or len(exs) < num_examples):
                rel = line.split('  ')
                event1 = rel[0]
                event2 = rel[2]
                rel = rel[1]
                positions = file.readline().split('  ')
                e1_idx = positions[0]
                e2_idx = positions[1]
                sent1 = file.readline().split()
                tags1 = file.readline().split(',')
                sent2 = file.readline().split()
                tags2 = file.readline().split(',')
                line = file.readline()
                doc_name = None
                if not line.isspace():
                    doc_name = line
                    line = file.readline()  # empty line between exs
                assert line.isspace(), line
                line = file.readline()
                if source:
                    if doc_name is None or doc_name[:3].lower() != source:
                         continue
                ex = IndexedExamplePartial(
                    rel, sent1, sent2, tags1, tags2, e1_idx, e2_idx, doc_name=doc_name)
                exs.append(ex)
    return exs


def distant_parsed_examples(tokenizer, lm='roberta', ext='', num_examples=None, mask=False, random_mask=False, mask_events=False):
    exs = _distant_parsed_examples(
        tokenizer, ext=ext, num_examples=num_examples)
    if random_mask:
        exs = apply_random_mask(exs, tokenizer)
    if mask:
        mask = 'distant'
    print(len(exs), mask)
    data = convert_examples_to_features(examples=exs,
                                                tokenizer=tokenizer,
                                                max_seq_length=MAX_SEQ_LENGTH,
                                                doc_stride=DOC_STRIDE,
                                                mask=mask,
                                                mask_events=mask_events)
    data = make_tensor_dataset(data, model=lm)
    return exs, data


def udst(tokenizer, lm='roberta', split="train", example_dir="udst/DecompTime/out/", mask_events=False, mask_context=False):
    exs = parse_udst.get_examples(
        example_dir=example_dir, split=split)
    feats = convert_examples_to_features(examples=exs,
                                                 tokenizer=tokenizer,
                                                 max_seq_length=MAX_SEQ_LENGTH,
                                                 doc_stride=DOC_STRIDE,
                                                 mask=False,
                                                 mask_events=mask_events,
                                                 mask_context=mask_context)
    data = make_tensor_dataset(feats, model=lm)
    return exs, data


def udst_majority(tokenizer, lm='roberta', example_dir="udst/DecompTime/out/", split="dev", mask_events=False, ties=True):
    exs = parse_udst.get_majority_examples(
        example_dir=example_dir, split=split, ties=ties)
    feats = convert_examples_to_features(examples=exs,
                                                 tokenizer=tokenizer,
                                                 max_seq_length=MAX_SEQ_LENGTH,
                                                 doc_stride=DOC_STRIDE,
                                                 mask=False,
                                                 mask_events=mask_events)
    data = make_tensor_dataset(feats, model=lm)
    return exs, data

