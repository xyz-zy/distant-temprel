import argparse
import json, pickle
import os



def save_data(data, save_path, args):
    if args.output_type == 'json':
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=4)
    elif args.output_type == 'pickle':
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)


def get_tok_charstart(text, toks):
    starts = []
    cur = 0
    for tok in toks:
        cur = text.find(tok, cur)
        #assert cur >= 0
        if not cur >= 0:
            print(text)
            print(toks)
            print(tok)
            print(starts)
            exit()
        starts.append(cur)
        cur += len(tok)
    return starts


def v2_5_ud_annotations_to_data(path, save_path, args):
    data = []
    # doc container
    ud_doc_id = None
    sents_text = None
    sents_tok = None
    sents_tok_charstart = None
    # sent container
    single_sent_text = None
    single_sent_tok = []
    with open(path, 'r') as f:
        for l in f:
            l = l.strip()
            if l.startswith('# newdoc id'):
                if not sents_text is None:
                    data.append({'ud_doc_id': ud_doc_id,
                                 'sents_text': sents_text,
                                 'sents_tok': sents_tok,
                                 'sents_tok_charstart': sents_tok_charstart})
                ud_doc_id = l[len('# newdoc id = '):]
                sents_text = []
                sents_tok = []
                sents_tok_charstart = []
            elif l.startswith('# sent_id'):
                continue
            elif l.startswith('# text'):
                single_sent_text = l[9:].strip()
            elif len(l) == 0:
                assert not single_sent_text is None
                sents_text.append(single_sent_text)
                sents_tok.append(single_sent_tok)
                sents_tok_charstart.append(get_tok_charstart(single_sent_text, single_sent_tok))
                single_sent_text = None
                single_sent_tok = []
            else:
                attrs = l.split('\t')
                if attrs[0].isnumeric():
                    single_sent_tok.append(attrs[1])
                else:
                    # like in line 1734 in UD_English/v2.5/en_ewt-ud-train.conllu
                    continue
        assert single_sent_text is None # last line is empty so that the last sentence is stored
        data.append({'ud_doc_id': ud_doc_id,
                     'sents_text': sents_text,
                     'sents_tok': sents_tok,
                     'sents_tok_charstart': sents_tok_charstart})
    if save_path:
        save_data(data, save_path, args)
    return data 


def v1_2_ud_annotations_to_data(path, save_path, args):
    data = {'sents_tok': []}
    # sent container
    single_sent_tok = []
    with open(path, 'r') as f:
        for l in f:
            l = l.strip()
            if len(l) == 0:
                data['sents_tok'].append(single_sent_tok)
                single_sent_tok = []
            else:
                attrs = l.split('\t')
                if attrs[0].isnumeric():
                    single_sent_tok.append(attrs[1])
                else:
                    # like in line 1734 in UD_English/v2.5/en_ewt-ud-train.conllu
                    continue
        assert len(single_sent_tok) == 0 # last line is empty so that the last sentence is stored
    if save_path:
        save_data(data, save_path, args)
    return data


def split_12_by_25(data12, data25, save_path, args):
    data = []
    i_12 = 0
    for d in data25:
        data.append({'ud_doc_id': d['ud_doc_id'],
                     'sents_text': [],
                     'sents_tok': [],
                     'sents_tok_charstart': []})
        for toks in d['sents_tok']:
            single_sent_tok = data12['sents_tok'][i_12]
            single_sent_text = " ".join(single_sent_tok)
            data[-1]['sents_text'].append(single_sent_text)
            data[-1]['sents_tok'].append(single_sent_tok)
            data[-1]['sents_tok_charstart'].append(get_tok_charstart(single_sent_text, single_sent_tok))
            i_12 += 1
    assert i_12 == len(data12['sents_tok'])
    if save_path:
        save_data(data, save_path, args)
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser("MATRES annotation to data")
    parser.add_argument('--input', help='MATRES annotation path')
    parser.add_argument('--output', help='MATRES data output path')
    parser.add_argument('--output_type', default='pickle', help='action to do', choices=['json', 'pickle'])
    parser.add_argument('--ud_version', default='pickle', help='action to do', choices=['2.5', '1.2'])
    parser.add_argument('--split_by_25', default=False, action='store_true', help='split into documents by v2.5')
    parser.add_argument('--input_25', help='MATRES annotation path')
    args = parser.parse_args()

    if args.ud_version == '2.5':
        data = v2_5_ud_annotations_to_data(args.input, args.output, args)
    elif args.ud_version == '1.2':
        if args.split_by_25:
            data12 = v1_2_ud_annotations_to_data(args.input, None, args)
            data25 = v2_5_ud_annotations_to_data(args.input_25, None, args)
            data = split_12_by_25(data12, data25, args.output, args)
        else:
            data = v1_2_ud_annotations_to_data(args.input, args.output, args)
