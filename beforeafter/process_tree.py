import argparse
import glob
import os

from distant import Example, get_relation
from nltk import Tree, ParentedTree

def process_tree(tree_str, label):
   # print(tree_str)
    example = None
    try:
        #print("getting tree")
        tree = ParentedTree.fromstring(tree_str.__str__())
        #print("before get_relation")
        example = get_relation(tree, label)
        #print(example.e1)
    except ValueError as err:
        #print(err)
        pass
    return example


def get_next_tree(file):
    tree = ""
    line = file.readline()

    while not line.isspace() and len(line) > 0:
        tree += line
        line = file.readline()

    return tree

def get_label(filename):
    if "after" in filename:
        return "after"
    elif "before" in filename:
        return "before"
    elif "during" in filename:
        return "during"
    return None

def process_trees(filename, out_dir):
    num_examples = 0
    label = get_label(filename)

    basename = os.path.basename(filename)
    basename = basename.split(".")[0]

    output_filename = out_dir + basename + ".json"
    if os.path.exists(output_filename):
        print("already parsed, at", output_filename)
        return
    output_file = None
    has_tree = False

    with open(filename) as file:

        tree = get_next_tree(file)

        while len(tree) > 0:
            # print(tree)
            example = process_tree(tree, label)

            if example:
                if not has_tree:
                    output_file = open(output_filename, "w")
                    has_tree = True
                    print("[", file=output_file)
                    print(example.to_json(), file=output_file, end="")
                if num_examples > 0:
                    print(",\n" + example.to_json(), file=output_file, end="")

                num_examples += 1
            else:
                print("error")

            tree = get_next_tree(file)
    if has_tree:
        print("\n]", file=output_file)


def main(path, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    for filename in glob.iglob(path):
        print(filename)
        process_trees(filename, out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tree_path', help="will be glob'd")
    parser.add_argument('--out_dir')
    args = parser.parse_args()
    main(args.tree_path, args.out_dir)


