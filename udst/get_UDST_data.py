import re
import argparse
import json, pickle
import sys
import os
import glob
from collections import defaultdict, deque, Counter
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from graphviz import Digraph
import html
from tqdm import tqdm, trange
import math
import random


def check_short_cut(start, end, reverse_events_edges, eiids):
    assert start in reverse_events_edges[end]
    queue = deque()
    for neighbor in reverse_events_edges[end]:
        if neighbor == start:
            continue
        queue.append(neighbor)
    seen = {eiid: False for eiid in eiids}
    while len(queue) > 0:
        n = queue.popleft()
        if n == start:
            return True
        else:
            for neighbor in reverse_events_edges[n]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    queue.append(neighbor)
    return False


def get_equivalence_group(equal_egdes):
    graph = defaultdict(list)
    for e1, e2 in equal_egdes:
        graph[e1].append(e2)
        graph[e2].append(e1)
    seen = set()
    groups = []
    for node in graph:
        if not node in seen:
            cc = []
            queue = deque([node])
            seen.add(node)
            while len(queue) > 0:
                node = queue.popleft()
                cc.append(node)
                for neighbor in graph[node]:
                    if not neighbor in seen:
                        seen.add(neighbor)
                        queue.append(neighbor)
            groups.append(cc)
    return groups


def get_data_dict(raw, events, events_edges, args):
    # concate passage and tokenization
    sents_char_offset = []
    sents_tok_offset = []
    concat_passages = ""
    concat_passages_toks = []
    for s_idx in range(len(raw['sents_text'])):
        toks = raw['sents_tok'][s_idx]
        sents_char_offset.append(len(concat_passages))
        sents_tok_offset.append(len(concat_passages_toks))
        concat_passages += raw['sents_text'][s_idx] + (' ' if s_idx < len(raw)-1 else '')
        concat_passages_toks += raw['sents_tok'][s_idx]
    for e in events:
        s_id = e['sent_id']
        e['char_start'] = sents_char_offset[s_id] + raw['sents_tok_charstart'][s_id][e['tok_start']]
        assert concat_passages[e['char_start']] == e['event'][0]
        e['tok_span'] = [sents_tok_offset[s_id] + p for p in e['tok_span']]
        assert " ".join(concat_passages_toks[p] for p in e['tok_span']) == e['event'] or e['event'][-3:] == '...'
        e['tok_start'] = e['tok_span'][0]
        e['tok_end'] = e['tok_span'][-1] # inclusive
    # deal with equal edges. (e1, e2) => make all edges only connect to e1
    equal_egdes = events_edges.pop('EQUAL') if 'EQUAL' in events_edges else []
    data_dict = {'text': concat_passages,
                 'tokens': concat_passages_toks,
                 'sents_tok_offset': sents_tok_offset,
                 'sents_char_offset': sents_char_offset,
                 'eiid2events': {e['eiid']: e for e in events},
                 'events_edges': events_edges,
                 'vague_edges': [],
                 'equal_edges': equal_egdes
                 }
    return data_dict


def get_event_obj(pair, event_id, docid2sentoffset, UD_data):
    split_type = pair['Split']
    doc_id = int(pair['Document.ID']) - 1 # doc indexing starts from 1
    eid = pair['Event%d.ID' % event_id]
    src, flatten_sent_id = pair['Sentence%d.ID' % event_id].split(' ')
    src = src.split('.')[0]
    assert src == 'en-ud-'+split_type
    sent_id = int(flatten_sent_id) - 1 - docid2sentoffset[src][doc_id] # sent num indexing starts from 1
    event = html.unescape(pair['Pred%d.Text' % event_id]) # TODO: event too long may be shorten (A B C D -> A B...)
    lemma = html.unescape(pair['Pred%d.Lemma' % event_id])
    e_tok_pos = [int(p) for p in pair['Pred%d.Span' % event_id].split('_')] # 0-indexing, inclusive
    assert " ".join(UD_data[src][doc_id]['sents_tok'][sent_id][p] for p in e_tok_pos) == event or event[-3:] == '...'
    e_tok_start = e_tok_pos[0] # 0-indexing, inclusive
    e_tok_end = e_tok_pos[-1]
    e_head_pos = int(pair['Pred%d.Token' % event_id])
    obj = {'event': event,
           'eid': eid,
           'eiid': None,
           'tok_span': e_tok_pos, # may be discontinuous
           'tok_start': e_tok_start,
           'tok_end': e_tok_end,
           'char_start': None,
           'doc_id': doc_id,
           'sent_id': sent_id,
           'split': split_type}
    return eid, src, obj


def get_event_tempinfo(pair, event_id):
    e_duration = int(pair['Pred%d.Duration' % event_id])
    e_beg = int(pair['Pred%d.Beg' % event_id])
    e_end = int(pair['Pred%d.End' % event_id])
    e_duration_conf = float(pair['Pred%d.Duration.Confidence' % event_id])
    e_relation_conf = float(pair['Relation.Confidence'])
    return e_duration, e_beg, e_end, e_duration_conf, e_relation_conf


def get_relation(e_obj1, e_obj2):
    # decide the relation by beg
    beg1, end1 = e_obj1['beg'], e_obj1['end']
    beg2, end2 = e_obj2['beg'], e_obj2['end']
    if beg1 < beg2:
        return 'BEFORE'
    elif beg1 > beg2:
        return 'AFTER'
    else:
        return 'EQUAL'


def check_tie_annotation(rels):
    rel_avg_conf = defaultdict(lambda: [0, 0.])
    for rel, conf in rels:
        rel_avg_conf[rel][0] += 1
        rel_avg_conf[rel][1] += conf
    if all(rel_avg_conf[rel][0] == 1 for rel in rel_avg_conf):
        assert set(rel_avg_conf.keys()) == {'BEFORE', 'EQUAL', 'AFTER'}
        if all(rel_avg_conf[rel][1] == rel_avg_conf['BEFORE'][1] for rel in rel_avg_conf):
            return True
    return False


def get_majority_relation(rels, rel_score):
    ''' rels: [(REL, CONF), (REL, CONF), ...]
        first decide by count, then decide by avg conf
    '''
    # get avg conf for each rel
    rel_avg_conf = defaultdict(lambda: [0, 0.])
    for rel, conf in rels:
        rel_avg_conf[rel][0] += 1
        rel_avg_conf[rel][1] += conf
    if len(rel_avg_conf) > 0:
        if rel_score == 'avg_conf':
            rel_avg_conf = {rel: [tot / cnt, cnt] for rel, (cnt, tot) in rel_avg_conf.items()}
        elif rel_score == 'maj':
            rel_avg_conf = {rel: [cnt, tot / cnt] for rel, (cnt, tot) in rel_avg_conf.items()}
        # sort first by cnt, then by avg_conf
        all_rels = sorted(list(rel_avg_conf.keys()), key=lambda x: rel_avg_conf[x], reverse=True)
        return all_rels[0]
    else:
        return None


def get_consistent_annotations(inputs):
    fn = inputs[0]
    doc_id = inputs[1]
    events = inputs[2]
    events_rels = inputs[3]
    event_graph_solver = inputs[4]
    events_rels = event_graph_solver.solve_event_graph(events, events_rels)
    return fn, doc_id, events_rels


def time_annotations_to_data(path, UD_paths, save_dir, args):
    # load parsed UD data
    UD_data = {}
    for fn in glob.glob(UD_paths):
        with open(fn, 'r') as f:
            UD_data[fn.split('/')[-1].split('.')[0]] = json.load(f)
    # get the sent_num offset for each doc to transform the flatten sent_num to original one
    docid2sentoffset = {}
    for fn in UD_data:
        docid2sentoffset[fn] = []
        cur_sent_num = 0
        for doc in UD_data[fn]:
            docid2sentoffset[fn].append(cur_sent_num)
            cur_sent_num += len(doc['sents_text'])
    # get events from UDS-T
    docid2events = {}
    docid2events_rels = {}
    for fn in UD_data:
        docid2events[fn] = [[] for _ in UD_data[fn]]
        docid2events_rels[fn] = [defaultdict(list) for _ in UD_data[fn]]
    attr_keys = None
    eid2eobj = {}
    with open(path, 'r') as f:
        for l in tqdm(f):
            l = l.strip()
            if l.startswith('Split'):
                attr_keys = l.split('\t')
            else:
                pair = {k: v for k, v in zip(attr_keys, l.split('\t'))}
                split_type = pair['Split']
                doc_id = int(pair['Document.ID']) - 1 # doc indexing starts from 1
                # event 1
                eid1, src1, e_obj1 = get_event_obj(pair, 1, docid2sentoffset, UD_data)
                if eid1 in eid2eobj:
                    assert {k:v for k, v in e_obj1.items() if not k == 'eiid'} == {k:v for k, v in eid2eobj[eid1].items() if not k == 'eiid'}, repr(e_obj1) + '\n' + repr(eid2eobj[eid1])
                    e_obj1 = eid2eobj[eid1]
                    eiid1 = e_obj1['eiid']
                else:
                    eiid1 = 'ei'+str(len(docid2events[src1][doc_id]))
                    e_obj1['eiid'] = eiid1
                    docid2events[src1][doc_id].append(e_obj1)
                    eid2eobj[eid1] = e_obj1
                # event2
                eid2, src2, e_obj2 = get_event_obj(pair, 2, docid2sentoffset, UD_data)
                if eid2 in eid2eobj:
                    assert {k:v for k, v in e_obj2.items() if not k == 'eiid'} == {k:v for k, v in eid2eobj[eid2].items() if not k == 'eiid'}
                    e_obj2 = eid2eobj[eid2]
                    eiid2 = e_obj2['eiid']
                else:
                    eiid2 = 'ei'+str(len(docid2events[src2][doc_id]))
                    e_obj2['eiid'] = eiid2
                    docid2events[src2][doc_id].append(e_obj2)
                    eid2eobj[eid2] = e_obj2
                # edge
                assert src1 == src2
                _, e_beg1, e_end1, _, rel_conf = get_event_tempinfo(pair, 1)
                _, e_beg2, e_end2, _, rel_conf = get_event_tempinfo(pair, 2)
                rel = get_relation({'beg': e_beg1, 'end': e_end1}, {'beg': e_beg2, 'end': e_end2})
                #docid2events_rels[src1][doc_id][(eiid1, eiid2)].append(rel)
                docid2events_rels[src1][doc_id][(eiid1, eiid2)].append((rel, rel_conf))
    # get the event graph for each doc
    data = {}
    for fn in UD_data:
        if args.src is not None and not args.src == fn:
            continue
        data[fn] = []
        for doc_id, (raw, events, events_rels) in enumerate(zip(tqdm(UD_data[fn]), docid2events[fn], docid2events_rels[fn])):
            # some event pair may have multiple annotation => take the majority
            events_edges = defaultdict(list)
            for eiid1, eiid2 in events_rels:
                if args.remove_ties and check_tie_annotation(events_rels[(eiid1, eiid2)]):
                    continue
                #rel = Counter(events_rels[(eiid1, eiid2)]).most_common(1)[0][0]
                if args.preserve_all_anns:
                    rels = [rel for rel, rel_conf in events_rels[(eiid1, eiid2)]]
                else:
                    rel = get_majority_relation(events_rels[(eiid1, eiid2)], args.rel_score)
                    rels = [rel]
                for rel in rels:
                    if rel == 'BEFORE':
                        events_edges[eiid1].append(eiid2)
                    elif rel == 'AFTER':
                        events_edges[eiid2].append(eiid1)
                    elif rel == 'EQUAL':
                        events_edges['EQUAL'].append((eiid1, eiid2))
                    elif rel is None:
                        print("None")
                        pass
                    else:
                        raise ValueError("Unknown Relation:", rel)
            if len(events) == 0:
                assert len(events_edges) == 0
                continue
            data_dict = get_data_dict(raw, events, events_edges, args)
            data_dict['doc_id'] = str(doc_id)
            data[fn].append(data_dict)

        if args.action == 'json':
            with open(os.path.join(save_dir, fn+'.json'), 'w') as f:
                json.dump(data[fn], f, indent=4)
        elif args.action == 'pickle':
            with open(os.path.join(save_dir, fn+'.pkl'), 'wb') as f:
                pickle.dump(data[fn], f)
    return data


def render_events_tree(d):
    g = Digraph()
    eiid2events = d['eiid2events']
    for start, ends in d['events_edges'].items():
        for end in ends:
            g.edge(eiid2events[start]['event'], eiid2events[end]['event'])
    g.render(filename='vis_data/test.gv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("UDS-T annotation to data")
    parser.add_argument('--input', help='UDS-T annotation path')
    parser.add_argument('--UD_inputs', help='UD inputs')
    parser.add_argument('--output_dir', help='UDS-T data output path')
    parser.add_argument('--action', default='pickle', help='action to do', choices=['json', 'pickle'])
    parser.add_argument('--src', default=None, help='src', choices=['en-ud-train', 'en-ud-dev', 'en-ud-test'])
    parser.add_argument('--num_workers', type=int, help='number of workers for multiprocessing', default=12)
    parser.add_argument('--preserve_all_anns', action='store_true', default=False,
                        help='whether to preserve all the annotations')
    parser.add_argument('--rel_score', help='how to score relations. Only used when `preserve_all_anns` is false',
                        choices=['avg_conf', 'maj'], default='maj')
    parser.add_argument('--remove_ties', action='store_true', default=False,
                        help='whether to remove instances whose annotations are tie in both voting and conf.')
    args = parser.parse_args()

    data = time_annotations_to_data(args.input, args.UD_inputs, args.output_dir, args)
    #render_events_tree(data['en-ud-train'][0])
