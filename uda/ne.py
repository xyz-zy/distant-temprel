import sys

from allennlp.predictors.predictor import Predictor
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter

from load_data import IndexedExamplePartial
from timebank.timeml import TimeMLExample


# PER:
#   dobj: him/her
#   poss: his/hers
#   nsub: he/she
per_dict = {'dobj': 'him', 'poss': 'his', 'nsubj': 'he', 'pobj': 'him'}
loc_dict = {'dobj': 'there', 'poss': 'its', 'nsubj': 'it', 'pobj': 'there'}
repl_dict = {'PER': per_dict, 'LOC': loc_dict}

dep_map = {'nsubj': 'nsubj', 'nn': 'nsubj', 'nsubjpass': 'nsubj',
           'dobj': 'dobj', 'pobj': 'pobj',
           'poss': 'poss', 'possessive': 'poss'}


class NEReplacer(object):
    def __init__(self, ner=None, coref=None, dep=None):

        self.ner = ner if ner else Predictor.from_path(
            "https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.12.18.tar.gz")

        self.coref = coref if coref else Predictor.from_path(
            "https://s3-us-west-2.amazonaws.com/allennlp/models/coref-model-2018.02.05.tar.gz")
        self.dep = dep if dep else Predictor.from_path(
            "https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz")

    def do_ner(self, text):
        return self.ner.predict(sentence=text)

    def do_coref(self, text):
        return self.coref.predict(document=text)

    def do_dep(self, text):
        return self.dep.predict(sentence=text)

    def get_span_replacements(self, doc, ner_tags, coref_clusters, pred_dep, pos):
        span_replacements = []
        for cluster in coref_clusters:
            #             print(cluster)
            #             for ent_span in cluster:
            #                 print(doc[ent_span[0]:ent_span[1]+1])

            span_ne_tags = None
            for ent_span in cluster:
                ent_span_tags = ner_tags[ent_span[0]:ent_span[1]+1]
                isNE = len([x for x in ent_span_tags if x != 'O']) != 0
                if isNE:
                    span_ne_tags = ent_span_tags

            if not span_ne_tags:
                continue

            # Gets the named entity type (ORG, PER, or LOC)
            only_ne_tags = [x for x in span_ne_tags if x != 'O']
            only_ne_types = [x.split("-")[1] for x in only_ne_tags]

            if len(set(only_ne_types)) != 1:
                #                 print(only_ne_types)
                continue

            ne_type = only_ne_types[0]
#             print('named entity type:', ne_type)

            cluster_replacements = []

#             print(cluster, span_ne_tags)
            for ent_span in cluster:
                doc_span = doc[ent_span[0]:ent_span[1]+1]
#                     print(doc[ent_span[0]:ent_span[1]+1])
    #             print(pos[ent_span[0]:ent_span[1]+1])

                span_pred_dep = pred_dep[ent_span[0]:ent_span[1]+1]
#                     print('span_pred_dep', span_pred_dep)

                valid = True
                for x in span_pred_dep:
                    if x not in dep_map:
                        #                         print('not in dep map', x, doc_span)
                        cluster_replacements = None
                        valid = False
                if not valid:
                    #                     print('break')
                    break

                # Get the dependency type (nsubj, dobj, or pos).
                dep_types = [dep_map[x] for x in span_pred_dep]

                # Finds the pronoun to replace with, according to the dependency type.
                if len(set(dep_types)) == 1 and ne_type in repl_dict:
                    dep_type = dep_types[0]
#                     print('dep type:', dep_type)
                    repl = repl_dict[ne_type][dep_type]
#                     print('replacement:', repl)
                    cluster_replacements.append(
                        tuple([ent_span[0], ent_span[1], repl]))
                elif len(set(dep_types)) > 1:
                    #                     print('multiple dep types', dep_types)
                    repl = None
                elif len(set(dep_types)) == 0:
                    #                     print('no dep types', dep_types)
                    repl = None
            print()
            if cluster_replacements:
                span_replacements += cluster_replacements

#         print(span_replacements)
        return span_replacements

    def _get_oldtok2newtok(self, oldtoks, new_toks):
        oldtok2newtok = []
        new_tok_idx = 0
        for tok in oldtoks:
            # Original token may be split into multiple tokens in 'new_toks'.
            new_tok = new_toks[new_tok_idx]
            assert tok.startswith(new_tok), tok + " " + new_tok
            oldtok2newtok.append(new_tok_idx)

            while len(tok) > len(new_tok):
                tok = tok[len(new_tok):]
                new_tok_idx += 1
                new_tok = new_toks[new_tok_idx]
                assert tok.startswith(new_tok), tok + " " + new_tok
            new_tok_idx += 1
        return oldtok2newtok, new_tok_idx

    def get_oldtok2newtok(self, sent1, sent2, doc):
        oldtok2newtok_1, sent1_end = self._get_oldtok2newtok(sent1, doc)

        if sent1 == sent2:
            oldtok2newtok_2 = oldtok2newtok_1
            new_sent1 = new_sent2 = doc
        else:
            new_sent1 = doc[:sent1_end].copy()
            new_sent2 = doc[sent1_end:].copy()
            oldtok2newtok_2, _ = self._get_oldtok2newtok(
                sent2, doc[sent1_end:])

        return new_sent1, oldtok2newtok_1, new_sent2, oldtok2newtok_2

    def replace(self, ex):
        iss = ex.sent1 == ex.sent2
        ex.text = ex.sent1 if iss else ex.sent1 + ex.sent2

        sent = " ".join(ex.text)
        c = self.do_coref(sent)
        n = self.do_ner(sent)
        d = self.do_dep(sent)

        ner_tags = n['tags']
        coref_clusters = c['clusters']
        doc = c['document']
        words = n['words']

        for t1, t2 in zip(doc, words):
            assert t1 == t2, t1 + " " + t2

        pos = d['pos']
        pred_dep = d['predicted_dependencies']

        span_replacements = self.get_span_replacements(
            doc, ner_tags, coref_clusters, pred_dep, pos)

        span_replacements.sort(key=lambda a: a[0])
        if len(span_replacements) == 0:
            return ex

        # Checks that the spans to replace are non-overlapping.
        end = 0
        for sr in span_replacements:
            assert sr[0] >= end
            end = sr[1]

        # Maps from ex.sent1 and ex.sent2 to doc/words resulting from allennlp parse.
        sent1, oldtok2newtok_1, sent2, oldtok2newtok_2 = self.get_oldtok2newtok(
            ex.sent1, ex.sent2, doc)
        segment_ids = [0] * len(sent1) + [1] * \
            len(sent2) if not iss else [0] * len(sent1)
        tags = [None] * len(doc)

        # Gets e1_idx and e2_idx for transformed example.
        e1_idx = oldtok2newtok_1[ex.e1_idx]
        e2_idx = oldtok2newtok_2[ex.e2_idx]
        assert sent1[e1_idx] == ex.sent1[ex.e1_idx]
        assert sent2[e2_idx] == ex.sent2[ex.e2_idx]

        tags[e1_idx] = 'e1'
        if iss:
            tags[e2_idx] = 'e2'
        else:
            tags[len(sent1) + e2_idx] = 'e2'

        # Constructs new sentence.
        start = 0
        new_doc = []
        new_segment_ids = []
        new_tags = []
        for sr in span_replacements:
            new_doc += words[start:sr[0]]
            new_doc.append(sr[2])

            new_segment_ids += segment_ids[start:sr[0] + 1]
            new_tags += tags[start:sr[0] + 1]
            assert len(new_doc) == len(new_segment_ids), str(
                len(new_doc)) + " " + str(len(new_segment_ids))
            assert len(new_doc) == len(new_tags), str(
                len(new_doc)) + " " + str(len(new_tags))

            start = sr[1] + 1

        new_doc += words[start:len(words)]
        new_segment_ids += segment_ids[start:len(words)]
        new_tags += tags[start:len(words)]

        assert len(new_doc) == len(new_segment_ids), str(
            len(new_doc)) + " " + str(len(new_segment_ids))
        assert len(new_doc) == len(new_tags), str(
            len(new_doc)) + " " + str(len(new_tags))

        sent1 = [x for x, y in zip(new_doc, new_segment_ids) if y == 0]
        sent1_tags = [x for x, y in zip(new_tags, new_segment_ids) if y == 0]
        e1_idx = sent1_tags.index('e1')

        if iss:
            sent2 = sent1
            sent2_tags = sent1_tags
            e2_idx = sent1_tags.index('e2')
        else:
            sent2 = [x for x, y in zip(new_doc, new_segment_ids) if y == 1]
            sent2_tags = [x for x, y in zip(
                new_tags, new_segment_ids) if y == 1]
            e2_idx = sent2_tags.index('e2')

        assert sent1[e1_idx] == ex.sent1[ex.e1_idx]
        assert sent2[e2_idx] == ex.sent2[ex.e2_idx]

        tags[e1_idx] = 'e1'
        if sent1 == sent2:
            tags[e2_idx] = 'e2'
        else:
            tags[len(sent1) + e2_idx] = 'e2'

        if isinstance(ex, TimeMLExample):
            doc_name = ex.filename
        elif isinstance(ex, IndexedExamplePartial):
            doc_name = ex.doc_name
        else:
            doc_name = None

        # Returns new example.
        return IndexedExamplePartial(ex.label, sent1, sent2, None, None, e1_idx, e2_idx, doc_name=doc_name)
