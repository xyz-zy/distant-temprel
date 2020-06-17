from .timeml import *
from allennlp.data.tokenizers.sentence_splitter import SpacySentenceSplitter

class ExampleLoader(object):

    def __init__(self):
        self.label_list = None
        self.sentence_splitter = SpacySentenceSplitter()

    def get_loss_weights():
        # Calculate loss weights as the inverse of label occurrence.
        loss_weights = {}
        for label in self.label_list:
          loss_weights[label] = 0

        for ex in train_examples:
          loss_weights[ex.str_label] += 1

        num_examples = len(train_examples)
        for key in loss_weights:
          loss_weights[key] = num_examples / loss_weights[key]
          
        weights_list = [float("%3.f" % loss_weights[key]) for key in loader.label_list]

        return weights_list

    def get_text_from_element(self, node):
        if node.nodeType == node.TEXT_NODE:
            if node.data.isspace():
                return ""
            else:
                return node.data.replace("\n", " ")
        else:
            text = ""
            for child in node.childNodes:
                text += " "+ self.get_text_from_element(child) + " "
            return text

    def process_node(self, node, events, times, full_text):
        if node.nodeName == "EVENT":
            eid = node.attributes['eid'].value
            cls = node.attributes['class'].value

            event = Event(eid=eid, cls=cls, sentence=None, pos_in_sentence=None)
            event.idx_in_doc = len(full_text)
            events[eid] = event
            return event

        if node.nodeName == "TIMEX3":
            tid = node.attributes['tid'].value
            type = node.attributes['type'].value
            time = TimeX3(tid=tid, sentence=None, pos_in_sentence=None)
            time.idx_in_doc = len(full_text)
            times[tid] = time
            return time

    def get_instances(self, instance_elts, event_instances, events, input_file):
        for instance in instance_elts:
            eiid = instance.attributes["eiid"].value
            eventID = instance.attributes["eventID"].value
            tense = instance.attributes["tense"].value
            aspect = instance.attributes["aspect"].value
            polarity = instance.attributes["polarity"].value
            pos = instance.attributes["pos"].value

            if eventID not in events:
                print(eventID, input_file)
                continue

            event = events[eventID]
            sentence = event.sentence
            pos_in_sentence = event.pos_in_sentence

            instance = EventInstance(eiid, event, tense, aspect, polarity, pos, sentence, pos_in_sentence)
            event_instances[eiid] = instance

    def parse_node(self, root, events, times, full_text):
#         print(full_text)
        for node in root.childNodes:
            if node.nodeType == node.TEXT_NODE and not node.data.isspace():
                text = re.sub(r"\n+", " ", node.data)
                text = re.sub(r"_", "", node.data)
                text = re.sub(r"&UR;", "", node.data)
                text = re.sub(r"&LR;", "", node.data)
                split_space = text.split()
                full_text += split_space
            elif node.nodeName == "TEXT":
                self.parse_node(node, events, times, full_text)
            else:
                el = self.process_node(node, events, times, full_text)
                text = self.get_text_from_element(node)
                if el:
                    el.text = text.strip()
                full_text += text.split()
                
    def get_full_text_to_sentences(self, full_text, sentences):
        split_sentences = [s.split() for s in sentences]

        def next_position(split_sentences, sent_num, sent_idx):
            cur_sent = split_sentences[sent_num]
            if sent_idx < len(cur_sent) - 1:
                sent_idx += 1
            else:
                sent_idx = 0
                sent_num += 1
                if sent_num < len(split_sentences):
                    cur_sent = split_sentences[sent_num]
            return sent_num, sent_idx
            
        split_sentences = [s.split() for s in sentences]

        full_text_to_sentences = []

        sent_num = 0
        sent_idx = 0
        for i, tok in enumerate(full_text):
            sent_tok = split_sentences[sent_num][sent_idx]
#             print(tok, sent_tok)
            assert tok.startswith(sent_tok), str(i) + " " + tok + " " + sent_tok + "\n" + str(split_sentences[sent_num])
            full_text_to_sentences.append(tuple([sent_num, sent_idx]))
            
            while len(tok) > len(sent_tok):
                tok = tok[len(sent_tok):]
                sent_num, sent_idx = next_position(split_sentences, sent_num, sent_idx)
                sent_tok = split_sentences[sent_num][sent_idx]
#                 print("WHILE", tok, sent_tok)
                assert tok.startswith(sent_tok), str(i) + " " + tok + " " + sent_tok + "\n" + str(split_sentences[sent_num])
#                 print(tok)
            
            sent_num, sent_idx = next_position(split_sentences, sent_num, sent_idx)
            
        return full_text_to_sentences
    
    def convert_doc_idx_to_sentences(self, sentences, full_text_to_sentences, its):
        for key, obj in its.items():
            idx = obj.idx_in_doc
            sentence, pos_in_sentence = full_text_to_sentences[idx]
#             print(idx, sentence, pos_in_sentence)
            text = sentences[sentence].split()[pos_in_sentence]
            assert text == obj.text.split()[0], text + " " + obj.text
            obj.sentence = sentence
            obj.pos_in_sentence = pos_in_sentence

    def read_file(self, input_file):
        """
        Parameters
        ----------
        input_file: str, path to input file

        Returns
        -------
        TimeMLFile containing sentences, events, eventInstances, times, and tlinks.
        """
        doc = dom.parse(input_file)
        root = doc.childNodes[0]

        events = {}
        times = {}
        full_text = []
        self.parse_node(root, events, times, full_text)
#         print(full_text)
        
        sentences = self.sentence_splitter.split_sentences(" ".join(full_text))
 
        full_text_to_sentences = self.get_full_text_to_sentences(full_text, sentences)
        
        self.convert_doc_idx_to_sentences(sentences, full_text_to_sentences, events)
        self.convert_doc_idx_to_sentences(sentences, full_text_to_sentences, times)

        event_instances = {}
        instanceElts = root.getElementsByTagName("MAKEINSTANCE")
        self.get_instances(instanceElts, event_instances, events, input_file)

        tlinks = []
        tlinkElts = root.getElementsByTagName("TLINK")
        for tlinkElt in tlinkElts:
            if tlinkElt.hasAttribute("relatedToEventInstance") and \
              tlinkElt.hasAttribute("eventInstanceID"):
                lid = tlinkElt.attributes["lid"].value
                relType = tlinkElt.attributes["relType"].value
                eiid = tlinkElt.attributes["eventInstanceID"].value
                relatedToEventInstance = tlinkElt.attributes["relatedToEventInstance"].value

                if eiid not in event_instances or relatedToEventInstance not in event_instances:
                    continue

                tlink = Tlink(lid, relType, event_instances[eiid], event_instances[relatedToEventInstance])
                tlinks.append(tlink)

            if tlinkElt.hasAttribute("eventInstanceID") and \
              tlinkElt.hasAttribute("relatedToTime"):
                lid = tlinkElt.attributes["lid"].value
                relType = tlinkElt.attributes["relType"].value
                eiid = tlinkElt.attributes["eventInstanceID"].value
                relatedToTime = tlinkElt.attributes["relatedToTime"].value

                if eiid not in event_instances or relatedToTime not in times:
                    continue
                tlink = Tlink(lid, relType, event_instances[eiid], times[relatedToTime])
                tlinks.append(tlink)

            if tlinkElt.hasAttribute("timeID") and \
              tlinkElt.hasAttribute("relatedToEventInstance"):
                lid = tlinkElt.attributes["lid"].value
                relType = tlinkElt.attributes["relType"].value
                tid = tlinkElt.attributes["timeID"].value
                eiid = tlinkElt.attributes["relatedToEventInstance"].value

                if tid not in times or eiid not in event_instances:
                    continue
                tlink = Tlink(lid, relType, times[tid], event_instances[eiid])
                tlinks.append(tlink)

            if tlinkElt.hasAttribute("timeID") and \
              tlinkElt.hasAttribute("relatedToTime"):
                lid = tlinkElt.attributes["lid"].value
                relType = tlinkElt.attributes["relType"].value
                tid = tlinkElt.attributes["timeID"].value
                relatedToTime = tlinkElt.attributes["relatedToTime"].value

                if tid not in times or relatedToTime not in times:
                    continue
                tlink = Tlink(lid, relType, times[tid], times[relatedToTime])
                tlinks.append(tlink)

        return TimeMLFile(sentences, events, event_instances, times, tlinks, input_file)

    def read_examples(self, input_file):
        file_data = self.read_file(input_file)

        examples = []

        for tlink in file_data.tlinks:
            #print(tlink.lid, tlink.relType, tlink.e1, tlink.e2)
            sent1 = tlink.e1.sentence
            sent2 = tlink.e2.sentence
            #print(sent1, sent2)

            example = None
            if sent1 >= len(file_data.sentences) or sent2 >= len(file_data.sentences):
                continue

            if sent1 == sent2:
                text = file_data.sentences[sent1]
                example = TimeMLExample(text, tlink.e1.pos_in_sentence, tlink.e2.pos_in_sentence, tlink.relType)
            elif sent1 < sent2:
                sents = file_data.sentences[sent1:sent2+1]
                text = " [SEP] ".join(sents)

                e1_pos = tlink.e1.pos_in_sentence
                e2_pos = sum([len(s.split()) + 1 for s in sents[:-1]]) + tlink.e2.pos_in_sentence

                example = TimeMLExample(text, e1_pos, e2_pos, tlink.relType)
            elif sent1 > sent2:
                sents = file_data.sentences[sent2:sent1+1]
                text = " [SEP] ".join(sents)

                e1_pos = sum([len(s.split()) + 1 for s in sents[:-1]]) + tlink.e1.pos_in_sentence
                e2_pos = tlink.e2.pos_in_sentence

                example = TimeMLExample(text, e1_pos, e2_pos, tlink.relType)

            if example:
                examples.append(example)
            #print(example)
        return examples


    def antithetics(self, all_examples):
        new_exs = []

        for ex in all_examples:
            new_ex = None
            if ex.str_label == "AFTER":
                new_ex = TimeMLExample(ex.text, ex.e2_pos, ex.e1_pos, "BEFORE")
                new_ex.int_label = self.label_list.index("BEFORE")
                new_exs.append(new_ex)

            if ex.str_label == "BEFORE":
                new_ex = TimeMLExample(ex.text, ex.e2_pos, ex.e1_pos, "AFTER")
                new_ex.int_label = self.label_list.index("AFTER")
                new_exs.append(new_ex)
                
            if ex.str_label == "DURING":
                new_ex = TimeMLExample(ex.text, ex.e2_pos, ex.e1_pos, "DURING")
                new_ex.int_label = self.label_list.index("DURING")
                new_exs.append(new_ex)

            if new_ex != None:
                new_ex.sentences = ex.sentences
                new_ex.e1_sentence_num = ex.e2_sentence_num
                new_ex.e1_sentence_pos = ex.e2_sentence_pos
                new_ex.e2_sentence_num = ex.e1_sentence_num
                new_ex.e2_sentence_pos = ex.e1_sentence_pos


        all_examples.extend(new_exs)
                

    def assign_num_labels(self, all_examples):
        if not self.label_list:
            labels = set()
            for ex in all_examples:
                labels.add(ex.str_label)
            labels = list(labels)
            labels.sort()
            print(labels)
            print(len(labels))
            self.label_list = labels

        for ex in all_examples:
            ex.int_label = self.label_list.index(ex.str_label)


    def read_examples_from_directory(self, dir_path):
        #os.chdir(dir_path)
        examples_list = []    
        for file in glob.glob(dir_path + "*.tml"):
            #file_path = dir_path + file
            examples = self.read_examples(file)
            examples_list.append(examples)

        all_examples = list(itertools.chain.from_iterable(examples_list))
        #antithetics(all_examples)
        print(len(all_examples))
        self.assign_num_labels(all_examples)
        return all_examples

    def read_example_files(self, dir_path):
        all_files = glob.glob(dir_path + "*.tml")
        train_files = all_files[:-4]
        dev_files = all_files[-4:]

        train_examples_list = []
        for file in train_files:
            examples = self.read_examples(file)
            train_examples_list.append(examples)
        train = list(itertools.chain.from_iterable(train_examples_list))

        dev_examples_list = []
        for file in dev_files:
            examples = self.read_examples(file)
            dev_examples_list.append(examples)
        dev = list(itertools.chain.from_iterable(dev_examples_list))
        self.assign_num_labels(train + dev)

        return train, dev

    def read_dense_examples(self, td_path, extra=False, window_size=None):
        class DenseExample(object):
            def __init__(self, file_name, e1, e2, label):
                self.file_name = file_name
                self.e1 = e1
                self.e2 = e2
                self.label = self.parse_label(label)
            def parse_label(self, label):
                labels = { "a":"AFTER", "b":"BEFORE", "i":"INCLUDES", "ii":"IS_INCLUDED",
                           "s":"SIMULTANEOUS", "v":"VAGUE" }
                return labels[label]


        DEV_DOCS = { "APW19980227.0487",
                     "CNN19980223.1130.0960", "NYT19980212.0019",
                     "PRI19980216.2000.0170", "ed980111.1130.0089" }

        TEST_DOCS = { "APW19980227.0489", "APW19980227.0494", "APW19980308.0201", "APW19980418.0210",
                      "CNN19980126.1600.1104", "CNN19980213.2130.0155",
                      "NYT19980402.0453", "PRI19980115.2000.0186",
                      "PRI19980306.2000.1675" }

        files_to_exs = {}

        f = open(td_path, "r")

        for line in f.readlines():
            split = line.split()
            ex = DenseExample(split[0], split[1], split[2], split[3])

            if ex.file_name not in files_to_exs:
                files_to_exs[ex.file_name] = [ex]
            else: 
                files_to_exs[ex.file_name].append(ex)

        files = set(files_to_exs.keys())
        train_files = files - DEV_DOCS - TEST_DOCS
        dev_files = DEV_DOCS

        train_examples = []
        for file_name in train_files:
            file = self.read_extra_file(EXTRA_FILE_DIR + "/" + file_name + ".tml") \
                    if extra \
                    else self.read_file(FILE_DIR + "/" + file_name + ".tml")

            for ex in files_to_exs[file_name]:
                e1 = file.get_element(ex.e1)
                e2 = file.get_element(ex.e2)

                if e1 == None or e2 == None:
                    #print("oops", file_name, ex.e1, ex.e2)
                    continue

                example = file.get_example(e1, e2, ex.label, window_size)

                if not example:
                    print("o no")
                else:
                    train_examples.append(example)

        self.assign_num_labels(train_examples)

        dev_examples = []
        for file_name in dev_files:
            file = self.read_extra_file(EXTRA_FILE_DIR + "/" + file_name + ".tml") \
                    if extra \
                    else self.read_file(FILE_DIR + "/" + file_name + ".tml")

            for ex in files_to_exs[file_name]:
                e1 = file.get_element(ex.e1)
                e2 = file.get_element(ex.e2)

                if e1 == None or e2 == None:
                    #print("oops", file_name, ex.e1, ex.e2)
                    continue

                example = file.get_example(e1, e2, ex.label, window_size)

                if not example:
                    print("o no")
                else:
                    dev_examples.append(example)

        self.assign_num_labels(dev_examples)
        return train_examples, dev_examples

    def read_dense_test_examples(self, td_path, extra=False, window_size=None):
        class DenseExample(object):
            def __init__(self, file_name, e1, e2, label):
                self.file_name = file_name
                self.e1 = e1
                self.e2 = e2
                self.label = self.parse_label(label)
            def parse_label(self, label):
                labels = { "a":"AFTER", "b":"BEFORE", "i":"INCLUDES", "ii":"IS_INCLUDED",
                           "s":"SIMULTANEOUS", "v":"VAGUE" }
                return labels[label]

        TEST_DOCS = { "APW19980227.0489", "APW19980227.0494", "APW19980308.0201", "APW19980418.0210",
                      "CNN19980126.1600.1104", "CNN19980213.2130.0155",
                      "NYT19980402.0453", "PRI19980115.2000.0186",
                      "PRI19980306.2000.1675" }

        files_to_exs = {}

        f = open(td_path, "r")

        for line in f.readlines():
            split = line.split()
            ex = DenseExample(split[0], split[1], split[2], split[3])

            if ex.file_name not in files_to_exs:
                files_to_exs[ex.file_name] = [ex]
            else: 
                files_to_exs[ex.file_name].append(ex)

        test_examples = []
        for file_name in TEST_DOCS:
            file = self.read_extra_file(EXTRA_FILE_DIR + "/" + file_name + ".tml") \
                    if extra \
                    else self.read_file(FILE_DIR + "/" + file_name + ".tml")

            for ex in files_to_exs[file_name]:
                e1 = file.get_element(ex.e1)
                e2 = file.get_element(ex.e2)

                if e1 == None or e2 == None:
                    #print("oops", file_name, ex.e1, ex.e2)
                    continue

                example = file.get_example(e1, e2, ex.label, window_size)

                if not example:
                    print("o no")
                else:
                    test_examples.append(example)

        self.assign_num_labels(test_examples)
        return test_examples

    def read_tempeval3_examples():
        return None, None

class TBDenseLoader(ExampleLoader):

    def __init__(self, td_path):
        super().__init__()
        self.td_path = td_path

    def read_train_dev_examples(self, extra=False, window_size=None):
        class DenseExample(object):
            def __init__(self, file_name, e1, e2, label):
                self.file_name = file_name
                self.e1 = e1
                self.e2 = e2
                self.label = self.parse_label(label)
            def parse_label(self, label):
                labels = { "a":"AFTER", "b":"BEFORE", "i":"INCLUDES", "ii":"IS_INCLUDED",
                           "s":"SIMULTANEOUS", "v":"VAGUE" }
                return labels[label]


        DEV_DOCS = { "APW19980227.0487",
                     "CNN19980223.1130.0960", "NYT19980212.0019",
                     "PRI19980216.2000.0170", "ed980111.1130.0089" }

        TEST_DOCS = { "APW19980227.0489", "APW19980227.0494", "APW19980308.0201", "APW19980418.0210",
                      "CNN19980126.1600.1104", "CNN19980213.2130.0155",
                      "NYT19980402.0453", "PRI19980115.2000.0186",
                      "PRI19980306.2000.1675" }

        files_to_exs = {}

        f = open(self.td_path, "r")

        for line in f.readlines():
            split = line.split()
            ex = DenseExample(split[0], split[1], split[2], split[3])

            if ex.file_name not in files_to_exs:
                files_to_exs[ex.file_name] = [ex]
            else: 
                files_to_exs[ex.file_name].append(ex)

        files = set(files_to_exs.keys())
        train_files = files - DEV_DOCS - TEST_DOCS
        dev_files = DEV_DOCS

        train_examples = []
        for file_name in train_files:
            file = self.read_extra_file(EXTRA_FILE_DIR + "/" + file_name + ".tml") \
                    if extra \
                    else self.read_file(FILE_DIR + "/" + file_name + ".tml")

            for ex in files_to_exs[file_name]:
                e1 = file.get_element(ex.e1)
                e2 = file.get_element(ex.e2)

                if e1 == None or e2 == None:
                    #print("oops", file_name, ex.e1, ex.e2)
                    continue

                example = file.get_example(e1, e2, ex.label, window_size)

                if not example:
                    print("o no")
                else:
                    train_examples.append(example)

        self.assign_num_labels(train_examples)

        dev_examples = []
        for file_name in dev_files:
            file = self.read_extra_file(EXTRA_FILE_DIR + "/" + file_name + ".tml") \
                    if extra \
                    else self.read_file(FILE_DIR + "/" + file_name + ".tml")

            for ex in files_to_exs[file_name]:
                e1 = file.get_element(ex.e1)
                e2 = file.get_element(ex.e2)

                if e1 == None or e2 == None:
                    #print("oops", file_name, ex.e1, ex.e2)
                    continue

                example = file.get_example(e1, e2, ex.label, window_size)

                if not example:
                    print("o no")
                else:
                    dev_examples.append(example)

        self.assign_num_labels(dev_examples)
        return train_examples, dev_examples

    def read_test_examples(self, extra=False, window_size=None):
        class DenseExample(object):
            def __init__(self, file_name, e1, e2, label):
                self.file_name = file_name
                self.e1 = e1
                self.e2 = e2
                self.label = self.parse_label(label)
            def parse_label(self, label):
                labels = { "a":"AFTER", "b":"BEFORE", "i":"INCLUDES", "ii":"IS_INCLUDED",
                           "s":"SIMULTANEOUS", "v":"VAGUE" }
                return labels[label]

        TEST_DOCS = { "APW19980227.0489", "APW19980227.0494", "APW19980308.0201", "APW19980418.0210",
                      "CNN19980126.1600.1104", "CNN19980213.2130.0155",
                      "NYT19980402.0453", "PRI19980115.2000.0186",
                      "PRI19980306.2000.1675" }

        files_to_exs = {}

        f = open(self.td_path, "r")

        for line in f.readlines():
            split = line.split()
            ex = DenseExample(split[0], split[1], split[2], split[3])

            if ex.file_name not in files_to_exs:
                files_to_exs[ex.file_name] = [ex]
            else: 
                files_to_exs[ex.file_name].append(ex)

        test_examples = []
        for file_name in TEST_DOCS:
            file = self.read_extra_file(EXTRA_FILE_DIR + "/" + file_name + ".tml") \
                    if extra \
                    else self.read_file(FILE_DIR + "/" + file_name + ".tml")

            for ex in files_to_exs[file_name]:
                e1 = file.get_element(ex.e1)
                e2 = file.get_element(ex.e2)

                if e1 == None or e2 == None:
                    #print("oops", file_name, ex.e1, ex.e2)
                    continue

                example = file.get_example(e1, e2, ex.label, window_size)

                if not example:
                    print("o no")
                else:
                    test_examples.append(example)

        self.assign_num_labels(test_examples)
        return test_examples


class MatresLoader(ExampleLoader):

    def __init__(self):
        super().__init__()
        self.train_files = []
        self.dev_files = []

    def read_subset_examples(self, doc_dir, rel_filename, window_size=None):
        rels_to_files = {}
        with open(rel_filename) as rel_file:
            for line in rel_file:
                rel = line.split()
                # print(rel[0])
                file_name = rel[0]
                if file_name in rels_to_files:
                    rels_to_files[file_name].append(rel)
                else:
                    rels_to_files[file_name] = [rel]


        num_files = len(rels_to_files)
        # print(num_files)
        split = int(num_files * 0.8)

        files = list(rels_to_files.keys())

        # Reads train files.
        train_examples = []
        for file in files[:split]:
            rels = rels_to_files[file]
            file_data = self.read_file(doc_dir + file + ".tml")
            self.train_files.append(file_data)
            rels = rels_to_files[file]
            for rel in rels:
                eiid1 = file_data.get_element("ei" + rel[3])
                eiid2 = file_data.get_element("ei" + rel[4])

                if eiid1 == None or eiid2 == None:
                    #print("oops", file_name, ex.e1, ex.e2)
                    continue

                example = file_data.get_example(eiid1, eiid2, rel[5], window_size)

                if not example:
                    print("o no")
                else:
                    train_examples.append(example)

        # Reads dev files.
        dev_examples = []
        for file in files[split:]:
            rels = rels_to_files[file]
            file_data = self.read_file(doc_dir + file + ".tml")
            self.dev_files.append(file_data)
            rels = rels_to_files[file]
            for rel in rels:
                eiid1 = file_data.get_element("ei" + rel[3])
                eiid2 = file_data.get_element("ei" + rel[4])

                if eiid1 == None or eiid2 == None:
                    #print("oops", file_name, ex.e1, ex.e2)
                    continue

                example = file_data.get_example(eiid1, eiid2, rel[5], window_size)

                if not example:
                    print("o no")
                else:
                    dev_examples.append(example)

        return train_examples, dev_examples

    def read_train_dev_examples(self, doc_dir="TBAQ-cleaned/", rel_dir="MATRES/", window_size=None):
        aquaint_rels = rel_dir + "aquaint.txt"
        timebank_rels = rel_dir + "timebank.txt"

        a_train, a_dev = self.read_subset_examples(doc_dir + "AQUAINT/", aquaint_rels)
        t_train, t_dev = self.read_subset_examples(doc_dir + "TimeBank/", timebank_rels)

        train_examples = a_train + t_train
        dev_examples = a_dev + t_dev

        self.assign_num_labels(train_examples)
        self.assign_num_labels(dev_examples)
        return train_examples, dev_examples

    def read_test_examples(self, doc_dir="te3-platinum/", rel_dir="MATRES/", window_size=None):

        platinum_rels = rel_dir + "platinum.txt"
        test_examples_1, test_examples2 = self.read_subset_examples(doc_dir, platinum_rels)
        
        examples = test_examples_1 + test_examples2
        self.assign_num_labels(examples)
        return examples
