import glob
import json

from bs4 import BeautifulSoup
from collections import deque
from nltk import ParentedTree
from tabulate import tabulate

SOURCE = "afp_eng_199405"
LABEL = "after"
FILE = SOURCE + "_" + LABEL + ".txt"
TMP_PATH = "./" + LABEL + "_tmp/"
failed_event_parse = 0


class Example:
    def __init__(self, tokens, e1, e1_pos, e2, e2_pos, label):
        self.valid = e1 is not None and e1_pos is not None and e2 is not None and e2_pos is not None
        self.tokens = tokens
        self.e1 = e1
        self.e1_pos = e1_pos
        self.e2 = e2
        self.e2_pos = e2_pos
        self.label = label

    def to_json(self):
        out_obj = {"tokens": self.tokens, "e1_text" : self.e1, "e1_pos": self.e1_pos,
                   "e2_text" : self.e2, "e2_pos" : self.e2_pos, "label" : self.label}
        return json.dumps(out_obj)

    def from_json(json_obj):
        return Example(json_obj["tokens"], json_obj["e1_text"], json_obj["e1_pos"],
                       json_obj["e2_text"], json_obj["e2_pos"], json_obj["label"])

    def __repr__(self):
        e1 = self.e1 if self.e1 else "None"
        e1_pos = str(self.e1_pos) if self.e1_pos is not None else "None"
        e2 = self.e2 if self.e2 else "None"
        e2_pos = str(self.e2_pos) if self.e2_pos is not None else "None"

        return str(self.tokens) + "\n (" + e1 + ", " + e1_pos + \
               ") (" + e2+ ", " + e2_pos + ")"

    def __bool__(self):
        # print(self.valid)
        return self.valid
    

def get_verb(vp_node):
    n = vp_node
    parent = None

    while isinstance(n, ParentedTree):
        parent = n
        n = None
        for child in parent:
            if not isinstance(child, ParentedTree) or child.label().startswith("VB"):
                n = child
                break
    # Takes main verb if present.
    if n in set(['was', 'had', 'were', 'being', 'is', 'are', 'has', 'be']):
        for child in vp_node:
            if child.label().startswith("VP"):
                 parent, n = get_verb(child)
                 break
    return parent, n


def get_parent_vp(rel_node):
    n = rel_node
    while n.parent() and n.label() != "VP":
        n = n.parent()  
    return n


def get_child_vp(rel_node):
    queue = deque()
    queue.append(rel_node)
    while len(queue) > 0:
        n = queue.popleft()
        if not isinstance(n, ParentedTree):
            continue
        if n.label() == "VP":
            return n
        else:
            for child in n:
                queue.append(child)
    return None


def get_rel_parent(tree):
    n = tree
    while n.parent() and (n.label() != "PP" and n.label() != "SBAR"):
        n = n.parent()
    return n


def get_token_position(root, token_parent, pos=0):
    # print(root.label())
    # print(token_parent.label())
    for child in root:
        if child == token_parent:
            return pos, True
        if isinstance(child, ParentedTree):
            pos, found = get_token_position(child, token_parent, pos=pos)
            if found:
                return pos, True
        else:
            pos += 1
    return pos, False


def match_pattern(root, tree, label):
    rel_parent = get_rel_parent(tree)
    # print(rel_parent)
    parent_vp = get_parent_vp(rel_parent)
    #print(parent_vp)
    e1_parent, e1 = get_verb(parent_vp)
    child_vp = get_child_vp(rel_parent)
    #print(child_vp)
    e2_parent, e2 = get_verb(child_vp)
    #print(e2)

    e1_pos = e2_pos = None
    if e1 and e2:
        e1_pos, e1_found = get_token_position(root, e1_parent)
        e2_pos, e2_found = get_token_position(root, e2_parent)

    return Example(root.leaves(), e1, e1_pos, e2, e2_pos, label)


def get_relation(tree, label):
    global failed_event_parse

    label_st = tree.subtrees(filter=lambda t: label in t.leaves() and t.height() == 2)
    #print("label_st", label_st)
    example = None
    for t in label_st:
        example = match_pattern(tree, t, label)
        break


    if example:
        # print(example)
        # print()
        pass
    else:
        failed_event_parse += 1
        print("ERROR IN PARSING")
        print(tree.leaves())

    return example


def main():
    files = glob.glob(TMP_PATH + "*.info.xml")

    OUT = open("out.out", "w")

    EXAMPLE_OUT = open("examples/" + SOURCE + "_" + LABEL + ".json", "w")
    num_examples = 0

    total_files = len(files)
    no_tlinks = 0

    print("[", file=EXAMPLE_OUT)
    for file in files:
        print(file, file=OUT)
        soup = BeautifulSoup(open(file), "html.parser")

        sentence = soup.sentence

        print(sentence.string, file=OUT)

        # store tokens in a list.
        tokens = []
        for token in soup.tokens.find_all("t"):
            text = token.string.rsplit("\"", 3)[0].split("\"", 3)[-1]
            if text[0] == " ":
                text = text[1:]
            tokens.append(text)

        print(file=OUT)
        # parse events
        # <event id="e1" eiid="ei1" offset="2" string="said" tense="PAST"
        #  aspect="NONE" class="REPORTING" polarity="POS" modality="" happen=""
        #  lowerBoundDuration="" upperBoundDuration="" 
        # />
        eid_dict = {}
        eiid_dict = {}
        for event in soup.events.find_all("event"):
            text = event["string"]
            token_pos = event["offset"]
            eid_dict[event["id"]] = text
            eiid_dict[event["eiid"]] = text
            print(text, file=OUT)

        # parse timexes
        # <timex tid="t1" text="autumn" offset="19" length="1" type="DATE"
        #  value="XXXX-FA" temporalFunction="false"/>
        timex_dict = {}
        for timex in soup.timexes.find_all("timex"):
            text = timex["text"]
            timex_dict[timex["tid"].strip()] = text
            print(text, file=OUT)

        print(file=OUT)


        tlinks = soup.find_all("tlink")
        if len(tlinks) == 0:
            no_tlinks += 1
            print("NO TLINKS", file=OUT)
        else:
            headers = ["e1", "e2", "relation"]
            table = []
            e1s = []
            e2s = []
            rels = []
            for tlink in tlinks:
                e1 = tlink["event1"]
                e2 = tlink["event2"]

                if e1 in eid_dict:
                    e1 = eid_dict[e1]
                elif e1 in eiid_dict:
                    e1 = eiid_dict[e1]
                elif e1 in timex_dict:
                    e1 = timex_dict[e1]
                else:
                    print("ERROR: Can't find e1", file=OUT)
                    print(eiid_dict)

                if e2 in eid_dict:
                    e2 = eid_dict[e2]
                elif e2 in eiid_dict:
                    e2 = eiid_dict[e2]
                elif e2 in timex_dict:
                    e2 = timex_dict[e2]
                else:
                    print("ERROR: Can't find e2", file=OUT)

                # print(e1, "\t", e2, "\t", tlink["relation"])
                table.append([e1, e2, tlink["relation"]])
            print(tabulate(table, headers=headers), file=OUT)
        print(file=OUT)

        parse = soup.parse.string

        t = ParentedTree.fromstring(parse)
        example = get_relation(t, LABEL)
        if example:
            if num_examples > 0:
                print(",", file=EXAMPLE_OUT)
            print(example.to_json(), file=EXAMPLE_OUT, end="")
            num_examples += 1

        print(parse, file=OUT)
        print(file=OUT)

    print("\n]", file=EXAMPLE_OUT)

    print("total files: ", total_files)
    print("files without tlinks: ", no_tlinks)
    print("files with failed event parsing: ", failed_event_parse)
    print("files with successful event parsing: ", total_files-failed_event_parse)

if __name__== "__main__":
    main()
