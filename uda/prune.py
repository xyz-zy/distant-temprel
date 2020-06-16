import random
import re
import sys

import allennlp
from allennlp.predictors import Predictor

from load_data import IndexedExamplePartial

class Pruner(object):
    def __init__(self):
        self.predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz")

    def _predict(self, sentence):
        return self.predictor.predict(sentence=sentence)
    
    def _get_span2tok(self, orig_sent, new_sent):
        span2tok = [None] * len(new_sent)
        orig_ptr = 0
        tok_ptr = 0
        while orig_ptr < len(orig_sent) and len(orig_sent[orig_ptr]) == 0:
            orig_ptr += 1
        for i in range(len(new_sent)):
            new_c = new_sent[i]
            if new_c.isspace():
                continue

            cur_tok = orig_sent[orig_ptr]
#             print(cur_tok, tok_ptr)
            old_c = cur_tok[tok_ptr]
#             print(old_c, new_c)
            assert new_c == old_c
            span2tok[i] = orig_ptr

            # increment to point to next char in orig_sent
            if tok_ptr == len(cur_tok) - 1:
                tok_ptr = 0
                orig_ptr += 1
                while orig_ptr < len(orig_sent) and len(orig_sent[orig_ptr]) == 0:
                    orig_ptr += 1
            else:
                tok_ptr += 1
        return span2tok
    
    def _get_candidates(self, tree, prune_candidates):
        nodeType = tree['nodeType']
        if nodeType == 'prep' and 'req' not in tree:
            prune_candidates.append(tree)
        if 'children' in tree:
            for c in tree['children']:
                self._get_candidates(c, prune_candidates)
                
    def _mark_tree(self, tree, span2tok, e_idx):
        if e_idx == None:
            return True

        sp_st = tree['spans'][0]['start']
        if span2tok[sp_st] == e_idx:
            tree['req'] = True
            return True
        if 'children' in tree:
            for c in tree['children']:
                if self._mark_tree(c, span2tok, e_idx):
                    tree['req'] = True
                    return True
                
    def _get_start(self, tree, st=100000):
        start = tree['spans'][0]['start']
        if start < st:
            st = start
        if 'children' in tree:
            for c in tree['children']:
                st = self._get_start(c, st)
        return st

    def _get_end(self, tree, nd=-1):
        end = tree['spans'][0]['end']
        if end > nd:
            nd = end
        if 'children' in tree:
            for c in tree['children']:
                nd = self._get_end(c, nd)
        return nd

    def _get_newtok2oldtok(self, span2tok):
        last_seen = None
        newtok2oldtok = []
        for idx in span2tok:
            if idx == None:
                newtok2oldtok.append(last_seen)
            last_seen = idx
        newtok2oldtok.append(last_seen)
        return newtok2oldtok
    
    def parse(self, orig_sent):
        parsed = self.predictor.predict(
          sentence=" ".join(orig_sent)
        )
        text = parsed['hierplane_tree']['text']
        tree = parsed['hierplane_tree']['root']

        return text, tree
    
    def get_prune_candidates(self, text, tree, orig_sent, e1_idx, e2_idx=None):
        
        span2tok = self._get_span2tok(orig_sent, text)
        assert self._mark_tree(tree, span2tok, e1_idx), str(span2tok) + "\n"  + str(e1_idx)
        assert self._mark_tree(tree, span2tok, e2_idx), str(span2tok) + "\n" + str(e2_idx)

        spans_to_delete = []

        prune_candidates = []
        self._get_candidates(tree, prune_candidates)

        return prune_candidates, span2tok
    
    def prune_subtree(self, tree, text, node_to_prune, span2tok):
        # Gets span of text to be pruned.
        start = self._get_start(node_to_prune)
        end = self._get_end(node_to_prune)
    #     print(start, end)

        pruned_text = text[0:start] + text[end:]
        pruned_span2tok = span2tok[0:start] + span2tok[end:]

        return pruned_text, pruned_span2tok
                
    def parse_and_prune(self, ex, orig_sent, e1_idx, e2_idx=None, choice='random', attributor=None):
        text, tree = self.parse(orig_sent)
        prune_candidates, span2tok = self.get_prune_candidates(text, tree, orig_sent,e1_idx, e2_idx)
        if len(prune_candidates) == 0:
            return orig_sent, span2tok

        if choice == 'random':
            # Picks a random subtree to prune.
            node_to_prune = random.choice(prune_candidates)
        elif choice == 'attr':
            assert attributor is not None
            node_to_prune = self.choose_highest_attr(ex, tree, text, prune_candidates, span2tok, attributor)

        pruned_text, pruned_span2tok = self.prune_subtree(tree, text, node_to_prune, span2tok)

        sent = pruned_text.split()
        newtok2oldtok = self._get_newtok2oldtok(pruned_span2tok)

        return sent, newtok2oldtok
    
    def choose_highest_attr(self, ex, tree, text, prune_candidates, span2tok, attributor):
        deleted_spans = []
        for pp in prune_candidates:
            pruned_text, pruned_span2tok = self.prune_subtree(tree, text, pp, span2tok)
            sent = pruned_text.split()
            newtok2oldtok = self._get_newtok2oldtok(pruned_span2tok)
            deleted_tokens = set(range(len(ex.sent1))) - set(newtok2oldtok)
            deleted_spans.append(deleted_tokens)
        attrs = attributor.get_span_attributions(ex, deleted_spans)
        attrs = [a/len(sp) if len(sp) > 0 else 0 for a, sp in zip(attrs, deleted_spans)]
#         print(attrs)
        best = max([(i,a) for i,a in enumerate(attrs)], key=lambda a:a[1])[0]
#         print(best)
        return prune_candidates[best]
        
        
    def get_pruned_example(self, ex, choice='random', attributor=None):
        single = ex.sent1 == ex.sent2
        has_tags = hasattr(ex, "tags1") and ex.tags1 != None

        if single:
            sent1, newtok2oldtok = self.parse_and_prune(ex, ex.sent1, ex.e1_idx, e2_idx=ex.e2_idx, choice=choice, attributor=attributor)
            deleted_tokens = set(range(len(ex.sent1))) - set(newtok2oldtok)
            if ex.e1_idx not in newtok2oldtok or ex.e2_idx not in newtok2oldtok:
                print("error", newtok2oldtok, file=sys.stderr)
                return ex
            e1_idx = newtok2oldtok.index(ex.e1_idx)
            e2_idx = newtok2oldtok.index(ex.e2_idx)
            if e1_idx >= len(sent1) or e2_idx >= len(sent1):
                print("error", e1_idx, e2_idx, sent1, file=sys.stderr)
                return ex
            if ex.sent1[ex.e1_idx] != sent1[e1_idx] or ex.sent1[ex.e2_idx] != sent1[e2_idx]:
                print("error", ex.sent1[ex.e1_idx], sent1[e1_idx], ex.sent1[ex.e1_idx], sent1[e2_idx], file=sys.stderr)
                return ex
            tags1 = None
            if has_tags:
                tags1 = [""] * len(sent1)
                for i, tag in enumerate(ex.tags1):
                    if (tag[0] == 't' or tag[0] == 'e') and i in newtok2oldtok:
                        tags1[newtok2oldtok.index(i)] = tag 

            ret_ex = IndexedExamplePartial(ex.label, sent1, sent1.copy(), tags1, tags1, e1_idx, e2_idx)
            ret_ex.deleted_tokens = deleted_tokens
            return ret_ex
        else:
            r = random.randint(0,1)
            if r == 0:
                sent1, newtok2oldtok = self.parse_and_prune(ex, ex.sent1, ex.e1_idx, choice=choice, attributor=attributor)
                deleted_tokens = set(range(len(ex.sent1))) - set(newtok2oldtok)
                if ex.e1_idx not in newtok2oldtok:
                    print("error", newtok2oldtok, file=sys.stderr)
                    return ex
                e1_idx = newtok2oldtok.index(ex.e1_idx)
                if e1_idx >= len(sent1):
                    print("error", e1_idx, sent1, file=sys.stderr)
                    return ex
                if ex.sent1[ex.e1_idx] != sent1[e1_idx]:
                    print("error", ex.sent1[ex.e1_idx], sent1[e1_idx], file=sys.stderr)
                    return ex

                tags1 = None
                tags2 = None
                if has_tags:
                    tags1 = [""] * len(sent1)
                    for i, tag in enumerate(ex.tags1):
                        if tag and (tag[0] == 't' or tag[0] == 'e') and i in newtok2oldtok:
                            tags1[newtok2oldtok.index(i)] = tag 
                    tags2 = ex.tags2.copy()

                ret_ex = IndexedExamplePartial(ex.label, sent1, ex.sent2.copy(), tags1, tags2, e1_idx, ex.e2_idx)
                ret_ex.deleted_tokens = deleted_tokens
                return ret_ex
            else:
                sent2, newtok2oldtok = self.parse_and_prune(ex, ex.sent2, ex.e2_idx, choice=choice, attributor=attributor)
                deleted_tokens = set(range(len(ex.sent2))) - set(newtok2oldtok)
                if ex.e2_idx not in newtok2oldtok:
                    print("error", newtok2oldtok, file=sys.stderr)
                    return ex
                e2_idx = newtok2oldtok.index(ex.e2_idx)
                if e2_idx >= len(sent2):
                    print("error", e2_idx, sent2, file=sys.stderr)
                    return ex
                if ex.sent2[ex.e2_idx] != sent2[e2_idx]:
                    print("error", ex.sent2[ex.e2_idx], sent2[e2_idx], file=sys.stderr)
                    return ex

                tags1 = None
                tags2 = None
                if has_tags:
                    tags2 = [""] * len(sent2)
                    for i, tag in enumerate(ex.tags2):
                        if (tag[0] == 't' or tag[0] == 'e') and i in newtok2oldtok:
                            tags2[newtok2oldtok.index(i)] = tag 
                    tags1 = ex.tags1.copy()

                ret_ex = IndexedExamplePartial(ex.label, ex.sent1.copy(), sent2, tags1, tags2, ex.e1_idx, e2_idx)
                ret_ex.deleted_tokens = deleted_tokens
                return ret_ex

