import argparse
import glob
import os

from bs4 import BeautifulSoup


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_path', help='will use glob, needs .info.xml output from caevo, hint:use quotes')
    parser.add_argument('--out_dir', help='will create if nonexistent')
    args = parser.parse_args()
    print(args)
    if not args.xml_path or not args.out_dir:
        print("Please provide arguments.")
        exit()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    main(xml_path, out_dir)

def trees_to_out(basename, trees, connective):
    if len(trees) > 0:
        with open(basename + "_" + connective + ".txt", "w+") as outfile:
            for tree in trees:
                print(tree, file=outfile)
                print(file=outfile)


def main(xml_path, out_dir):
    for filename in glob.iglob(xml_path):
        basename = os.path.basename(filename)
        if not basename.endswith('.info.xml'):
            continue
        #print(basename)
        basename = basename[-len('.info.xml'):] 
        basename = basename.replace('.', '_')
        soup = BeautifulSoup(open(filename), "html.parser")
        sentence_annotations = list(soup.findAll('entry'))
        after_trees = []
        before_trees = []
        for entry in sentence_annotations:
            sentence = entry.sentence.text
            if "after" in sentence:
                after_trees.append(entry.parse.text)
            if "before" in sentence:
                before_trees.append(entry.parse.text)

        trees_to_out(out_dir + basename, after_trees, "after")
        trees_to_out(out_dir + basename, before_trees, "before")

