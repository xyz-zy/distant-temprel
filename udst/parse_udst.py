import json

data_dir = "DecompTime/out/"
splits = {"train" : "en-ud-train.json",
          "dev"   : "en-ud-dev.json",
          "test"  : "en-ud-test.json"}

class UdsExample:
  def __init__(self, e1, e2, label):
     self.e1 = e1
     self.e2 = e2
     self.sent1 = e1.sentence
     self.sent2 = e2.sentence
     self.e1_idx = e1.sent_tok_start
     self.e2_idx = e2.sent_tok_start
     self.label = label


class Event:
  def __init__(self, json_obj):
    #self.doc = doc
    self.text = json_obj["event"]
    self.eid = json_obj["eid"]
    self.eiid = json_obj["eiid"]
    self.tok_start = json_obj["tok_start"]
    self.tok_end = json_obj["tok_end"]
    self.sent_id = json_obj["sent_id"]
    self.split = json_obj["split"]


class Doc:
  def __init__(self, json_obj):
    self.tokens = json_obj["tokens"]
    self.sents_tok_offset = json_obj["sents_tok_offset"]
    self.eiid2events = {eiid:Event(val) for eiid, val in json_obj["eiid2events"].items()}
    self.events_edges = json_obj["events_edges"]
    self.equal_edges = json_obj["equal_edges"]
    self._to_sentences()
    self._extract_examples()
  
  def _to_sentences(self):
    self.sentences = []
    for i in range(len(self.sents_tok_offset)):
      beg_tok = self.sents_tok_offset[i]
      end_tok = self.sents_tok_offset[i+1] if i < len(self.sents_tok_offset)-1 else len(self.tokens)
      sentence = self.tokens[beg_tok:end_tok]
      self.sentences.append(sentence)

      for eiid, event in self.eiid2events.items():
        if event.tok_start >= beg_tok and event.tok_start < end_tok:
          event.sent_tok_start = event.tok_start - beg_tok
          event.sent_tok_end = event.tok_end - beg_tok
          event.tok_length = event.tok_end - event.tok_start + 1
          event.sentence = sentence

  def extract_majority_examples(self, ties=True):
    event_map = {}
    for from_iid, to_iids in self.events_edges.items():
      efrom = self.eiid2events[from_iid]
      for to_iid in to_iids:
        eto = self.eiid2events[to_iid]
        if eto.sent_id < efrom.sent_id:
          e1 = eto
          e2 = efrom
          label = "AFTER"
        elif eto.sent_id == efrom.sent_id and eto.sent_tok_start < efrom.sent_tok_start:
          e1 = eto
          e2 = efrom
          label = "AFTER"
        else:
          e1 = efrom
          e2 = eto
          label = "BEFORE"
        if e1 not in event_map:
          event_map[e1] = {}
        if e2 not in event_map[e1]:
          event_map[e1][e2] = []
        event_map[e1][e2].append(label)
    for edge in self.equal_edges:
       efrom = self.eiid2events[edge[0]]
       eto = self.eiid2events[edge[1]]
       if eto.sent_id < efrom.sent_id:
         e1 = eto
         e2 = efrom
       elif eto.sent_id == efrom.sent_id and eto.sent_tok_start < efrom.sent_tok_start:
         e1 = eto
         e2 = efrom
       else:
         e1 = efrom
         e2 = eto
       if e1 not in event_map:
         event_map[e1] = {}
       if e2 not in event_map[e1]:
         event_map[e1][e2] = []
       event_map[e1][e2].append("EQUAL")

    majority_examples = [] 
    for e1, e2s in event_map.items():
      for e2, labels in e2s.items():
        label_counts = [tuple([l, labels.count(l)]) for l in set(labels)]
        # Sorts labels alphabetically, so ties are broken with "EQUAL".
        label_counts.sort(reverse=True, key=lambda x: x[0])
        label_counts.sort(reverse=True, key=lambda x: x[1])
        label = label_counts[0][0]
        if not ties and len(label_counts) > 1 and label_counts[0][1] == label_counts[1][1]:
          continue
        ex = UdsExample(e1, e2, label)
        ex.annotator_labels = label_counts
        majority_examples.append(ex)
    return majority_examples
 

  def _extract_examples(self):
    self.examples = []
    for e1_iid, e2_iids in self.events_edges.items():
      e1 = self.eiid2events[e1_iid]
      e2_sets = set(e2_iids)
      e2_counts = {e2_iid:e2_iids.count(e2_iid) for e2_iid in e2_iids}
      
      for e2_iid, count in e2_counts.items():
        e2 = self.eiid2events[e2_iid]
        #if e1.tok_length > 1 or e2.tok_length > 1:
        #  continue
        if e1.sent_id == e2.sent_id:
          if e1.sent_tok_start < e2.sent_tok_start:
            self.examples.append(UdsExample(e1, e2, "BEFORE"))
          else:
            self.examples.append(UdsExample(e2, e1, "AFTER"))
        elif e1.sent_id == e2.sent_id + 1:
           self.examples.append(UdsExample(e2, e1, "AFTER"))
        elif e2.sent_id == e1.sent_id + 1:
           self.examples.append(UdsExample(e1, e2, "BEFORE"))
    for edge in self.equal_edges:
      e1_iid = edge[0]
      e2_iid = edge[1]
      e1 = self.eiid2events[e1_iid]
      e2 = self.eiid2events[e2_iid]
      if (e1.sent_id == e2.sent_id and e1.sent_tok_start < e2.sent_tok_start)or e2.sent_id > e1.sent_id:
        self.examples.append(UdsExample(e1, e2, "EQUAL"))
      else:
        self.examples.append(UdsExample(e2, e1, "EQUAL"))


def get_examples(example_dir=data_dir, split="dev"):
  filename = example_dir + splits[split]

  parsed_docs = json.load(open(filename))
  docs = []
  examples = []

  for doc in parsed_docs:
    doc_obj = Doc(doc)
    docs.append(doc_obj)
    examples.extend(doc_obj.examples)
  return examples


def get_majority_examples(example_dir=data_dir, split="dev", ties=True):
  filename = example_dir + splits[split]
  parsed_docs = json.load(open(filename))
  docs = []
  examples = []

  for doc in parsed_docs:
    doc_obj = Doc(doc)
    docs.append(doc_obj)
    examples.extend(doc_obj.extract_majority_examples(ties=ties))
  return examples

