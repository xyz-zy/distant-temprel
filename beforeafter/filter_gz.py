'''
filter_gz.py : script to filter gigaword files to only before/after sentences
'''
import argparse
import glob
import gzip
import os

from bs4 import BeautifulSoup

BATCH_SIZE = 2000
def write(sentences, out_filename):
    if len(sentences) == 0:
        return
    it = 0 
    beg = 0 
    end = min(len(sentences), BATCH_SIZE)

    while beg < len(sentences):
        with open(out_filename + "_" + str(it) + ".txt", "w") as out_file:
            for i in range(beg, end):
                out_file.write(sentences[i] + "\n")

        beg = end 
        end = min(len(sentences), beg + BATCH_SIZE)
        it += 1


def filter_file(gz, out_dir):
    basename = os.path.basename(gz)
    with gzip.open(gz, 'rt') as un_gz:
        soup = BeautifulSoup(un_gz, 'html.parser')
        sentences = [s.text.strip().replace('\n', ' ') for s in soup.find_all('p')]
        before_sentences = []
        after_sentences = []
        for s in sentences:
            if "before" in s:
                before_sentences.append(s)
            if " after " in s:
                after_sentences.append(s)
        write(before_sentences, out_dir + "/" + basename[:-3] + "_before")
        write(after_sentences, out_dir + "/" + basename[:-3] + "_after")
        print("File: ", gz)
        print("Before Sentences:", len(before_sentences))
        print("After Sentences:", len(after_sentences))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', help="path to .gz gigaword files, will be glob'd")
    parser.add_argument('--out_dir', help='output_dir for filtered files')
    args = parser.parse_args()

    if not args.files or not args.out_dir:
        print("Please provide arguments.")
        exit()

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    for gz in glob.iglob(args.files):
        if not gz.endswith('.gz'):
            print(gz, "not a .gz file, skipping")
        else:
            filter_file(gz, args.out_dir)
    
