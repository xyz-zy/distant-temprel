'''
predict.py: script for making predictions
'''
import argparse
import sys

import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm, trange
from transformers import *

from modeling import make_tensor_dataset
from modeling import BertForMatres, RobertaForMatres#, ElectraForMatres

from constants import CLASSES, MAX_SEQ_LENGTH, DOC_STRIDE
from load_data import *


UDST_DIR = "DecompTime/DecompTime/out/"

def get_dummy_data(tokenizer, lm):
  sent1 = "Today I went to the store.".split()
  sent2 = "I came home.".split()
  # load_data.py:IndexedExamplePartial
  #        - for single-sentence examples sent1&sent2 are the same
  #            --> pass in SAME LIST (reference/pointer) for both
  ex = IndexedExamplePartial(label="BEFORE", # {AFTER, BEFORE, EQUALS, VAGUE}
                             sent1=sent1,
                             sent2=sent2,
                             tags1=None, # none, unless you want to mask timexes
                             tags2=None,
                             e1_idx=2, # "went" = sent1[e1_idx]
                             e2_idx=1, # "came" = sent2[e2_idx]
                             doc_name=None) # specify doc_name if you need it later

  exs = [ex]

  # load_data.py:convert_distant_examples_to_features
  #     - should automatically generate the right model-specific
  #       input features according to tokenizer type
  feats = convert_distant_examples_to_features(examples=exs,
                                               tokenizer=tokenizer,
                                               max_seq_length=MAX_SEQ_LENGTH,
                                               doc_stride=DOC_STRIDE)
 
  data = make_tensor_dataset(feats, model=lm) 
  return exs, data

def get_data(tokenizer, lm, data):
  '''
  tokenizer: PreTrainedTokenizer
  lm: str, {bert, roberta}
  data: str
  '''
  if data == 'udst_dev_maj':
    exs, data = udst_majority(tokenizer,
                              lm=lm,
                              example_dir=UDST_DIR,
                              split="dev")
  elif data == 'udst_test_maj':
    exs, data = udst_majority(tokenizer, 
                              lm=lm,
                              example_dir=UDST_DIR,
                              split="test")
  elif data == 'udst_train':
    exs, data = udst(tokenizer,
                     lm=lm,
                     split="train",
                     example_dir=UDST_DIR)
  elif data == 'dummy_data':
    exs, data = get_dummy_data(tokenizer, lm)
  else:
    print("model not yet supported, try {udst_train,udst_dev_maj,udst_test_maj}")
    raise NotImplementedError

  return exs, data

def load_model_from_directory(lm, model_dir):
  if lm.startswith('bert'):
    model = BertForMatres.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    return model, tokenizer
  elif lm.startswith('roberta'):
    model = RobertaForMatres.from_pretrained(model_dir)
    tokenizer = RobertaTokenizer.from_pretrained(model_dir)
  else:
    print("model not yet supported, try 'bert' model")
    raise NotImplementedError
  return model, tokenizer


def predict(model, data, device):
  data_sampler = SequentialSampler(data)
  data_loader = DataLoader(data, sampler=data_sampler, batch_size=20)
  
  model.to(device)
  model.eval()

  for batch in tqdm(data_loader, desc="Evaluating"):
    batch = tuple(t.to(device) for t in batch)
    with torch.no_grad():
      _, out, hidden = model(*batch)
      val, guess_idxs = out.max(1)
      for guess in guess_idxs:
        print(CLASSES[guess.item()])


def main(lm, model_dir, data):
  '''
  lm : str, 
  model_dir: str, 
  '''
  model, tokenizer = load_model_from_directory(lm, model_dir)
  exs, data = get_data(tokenizer, lm, data) 
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
  predict(model, data, device)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument('--lm', help='transformer model type, from {bert,roberta,electra}')
  parser.add_argument('--model_dir', help='path to model directory')
  parser.add_argument('--data', help='udst_dev_maj,udst_train,udst_test_maj,dummy_data')

  args = parser.parse_args()

  main(args.lm, args.model_dir, args.data)

