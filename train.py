'''
train.py: script for training
'''

import argparse
import os
import sys
import torch
from torch.nn.functional import kl_div, softmax, log_softmax
from itertools import chain
from utils import count_labels

parser = argparse.ArgumentParser()
parser.add_argument('--data', nargs='+', help='beforeafter,beforeafter_nd,beforeafter_yelp,matres,distant,distant_dct,udst')
parser.add_argument('--output_dir', help='path to model output directory')
parser.add_argument('--model_dir', help='directory for pretrained model')
parser.add_argument('--distant_source', help='{afp,apw,nyt,cna,wpb}')
parser.add_argument('--num_examples', type=int, help='only supported for beforeafter and distant')
parser.add_argument('--num_examples_nd', type=int, help='number of non-distant examples')
parser.add_argument('--mask', action='store_true')
parser.add_argument('--lm', help='bert,roberta,electra')
parser.add_argument('--random_mask', type=float)
parser.add_argument('--mask_events', action='store_true')
parser.add_argument('--serialize', action='store_true')
parser.add_argument('--epochs', type=int)
parser.add_argument('--batch', type=int)
parser.add_argument('--lr', type=float)
parser.add_argument('--print_loss', action='store_true')
parser.add_argument('--mask_context', action='store_true')
parser.add_argument('--prune', action='store_true')
parser.add_argument('--prune_rep', action='store_true')
parser.add_argument('--prune_choice', help='random,attr')
parser.add_argument('--unsup', help='matres')
parser.add_argument('--unsup_conf', action='store_true')
parser.add_argument('--unsup_batch', type=int)
parser.add_argument('--unsup_num_examples', type=int)
parser.add_argument('--unsup_mask', action='store_true')
parser.add_argument('--uda_method', help='prune,ne')
parser.add_argument('--uda_weight', type=float)
parser.add_argument('--disable_tqdm', action='store_true')
args = parser.parse_args()
print(args)

from constants import *

TRAIN_BATCH_SIZE = args.batch if args.batch else TRAIN_BATCH_SIZE
print("train batch size", TRAIN_BATCH_SIZE)
LEARNING_RATE = args.lr if args.lr else LEARNING_RATE
print("learning_rate", LEARNING_RATE)

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, ConcatDataset)
                             
from transformers import *
from tqdm import tqdm, trange

# Import data utils from repository. 
from timebank.examples import ExampleLoader, MatresLoader

from modeling import BertForMatres, RobertaForMatres, ElectraForMatres
from modeling import make_tensor_dataset, get_tensors

from prune import Pruner
from interpret import Attributor
from ne import NEReplacer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
#device = torch.device('cuda:0')
print(device, n_gpu)

if args.model_dir:
  if args.lm == 'roberta':
    model = RobertaForMatres.from_pretrained(args.model_dir)
    tokenizer = RobertaTokenizer.from_pretrained(args.model_dir)
  elif args.lm == 'bert':
    model = BertForMatres.from_pretrained(args.model_dir)
    tokenizer = BertTokenizer.from_pretrained(args.model_dir, do_lower_case=DO_LOWER_CASE)
  elif args.lm == 'electra':
    model = ElectraForMatres.from_pretrained(args.model_dir)
    tokenizer = ElectraTokenizer.from_pretrained(args.model_dir)
  else:
    print("Please specifify valid model from {'bert', 'roberta', 'electra'}", file=sys.stderr)
    exit()
else:
  if args.lm =='roberta':
    model = RobertaForMatres.from_pretrained('roberta-base')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
  elif args.lm == 'bert':
    model = BertForMatres.from_pretrained('bert-base-uncased',
                                          cache_dir=os.path.join(str(file_utils.PYTORCH_PRETRAINED_BERT_CACHE),
                                                               'distributed_{}'.format(-1)))
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
  elif args.lm == 'bert-large':
    model = BertForMatres.from_pretrained('bert-large-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    args.lm = 'bert'
  elif args.lm == 'electra':
    model = ElectraForMatres.from_pretrained('google/electra-base-discriminator')
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-base-discriminator')
  elif args.lm == 'electra-large':
    model = ElectraForMatres.from_pretrained('google/electra-large-discriminator')
    tokenizer = ElectraTokenizer.from_pretrained('google/electra-large-discriminator')
    args.lm = 'electra'
  else:
    print("Please specifify valid model from {'bert', 'roberta', 'electra'}", file=sys.stderr)
    exit()

model.to(device)
from load_data import *

all_exs = []
all_data = []

if 'beforeafter' in args.data:
  print("using beforeafter examples")
  exs, data = beforeafter_examples(tokenizer, lm=args.lm, num_examples=args.num_examples, mask=args.mask)
  all_exs.append(exs)
  all_data.append(data)
if 'beforeafter_nd' in args.data:
  print("using beforeafter examples, no during")
  exs, data = beforeafter_examples(tokenizer, lm=args.lm, num_examples=args.num_examples, mask=args.mask, during=False)
  all_exs.append(exs)
  all_data.append(data)
if 'beforeafter_yelp' in args.data:
  print("using beforeafter yelp examples")
  exs, data = beforeafter_examples(tokenizer, lm=args.lm, ext="_yelp",  num_examples=args.num_examples, mask=args.mask)
  all_exs.append(exs)
  all_data.append(data)
if 'matres' in args.data:
  print("using matres training examples")
  exs, data = matres_train_examples(tokenizer, lm=args.lm, train=True, mask_events=args.mask_events, mask_context=args.mask_context)
  if args.num_examples_nd:
    exs = random.sample(exs, args.num_examples_nd)
    data = convert_distant_examples_to_features(exs,
                                                 tokenizer=tokenizer,
                                                 max_seq_length=MAX_SEQ_LENGTH,
                                                 doc_stride=DOC_STRIDE,
                                                 mask_events=args.mask_events,
                                                 mask_context=args.mask_context)
    data = make_tensor_dataset(data, model=args.lm)
  all_exs.append(exs)
  all_data.append(data)
if 'distant' in args.data:
  print("using distant training examples")
  examples, data = distant_train_examples(tokenizer, lm=args.lm, source=args.distant_source, mask=args.mask, mask_events=args.mask_events, num_examples=args.num_examples)
  print(len(examples), "distant examples")
  all_exs.append(examples)
  all_data.append(data)
if 'distant_dct' in args.data:
  print("using distant DCT training examples")
  examples, data = distant_train_examples(tokenizer, lm=args.lm, ext="_dct", mask=args.mask, mask_events=args.mask_events, num_examples=args.num_examples)
  print(len(examples), "distant dct examples")
  all_exs.append(examples)
  all_data.append(data)
if 'distant_yelp' in args.data:
  print("using distant yelp training examples")
  examples, data = distant_train_examples(tokenizer, lm=args.lm, ext="_yelp", mask=args.mask, mask_events=args.mask_events, num_examples=args.num_examples)
  print(len(examples), "distant yelp examples")
  all_exs.append(examples)
  all_data.append(data)
if 'distant_yelp_parsed' in args.data:
  print("using distant yelp training examples, parsed only")
  examples, data = distant_parsed_examples(tokenizer, lm=args.lm, ext="_yelp", mask=args.mask, mask_events=args.mask_events, num_examples=args.num_examples)
  print(len(examples), "distant yelp examples, parsed only")
  all_exs.append(examples)
  all_data.append(data)
if 'udst' in args.data:
  print("using UDS-T training examples")
  exs, data = udst(tokenizer, lm=args.lm, split="train", mask_events=args.mask_events, mask_context=args.mask_context)
  if args.num_examples_nd:
    exs = random.sample(exs, args.num_examples_nd)
    data = convert_distant_examples_to_features(exs,
                                                 tokenizer=tokenizer,
                                                 max_seq_length=MAX_SEQ_LENGTH,
                                                 doc_stride=DOC_STRIDE,
                                                 mask_events=args.mask_events,
                                                 mask_context=args.mask_context)
    data = make_tensor_dataset(data, model=args.lm)
  all_exs.append(exs)
  all_data.append(data)
for data_source in args.data:
  if data_source.endswith('.pkl'):
    inputs = pickle.load(open(data_source, 'rb'))
    all_exs.append(inputs["exs"])
    all_data.append(inputs["data"])

if args.unsup:
  pruner = Pruner()
  if args.unsup == 'matres':
    print("using matres training examples")
    u_exs, u_data = matres_train_examples(tokenizer, lm=args.lm, train=True, mask_events=args.mask_events, mask_context=args.mask_context)
  elif args.unsup == 'udst':
    u_exs, u_data = udst(tokenizer, lm=args.lm, split="train", mask_events=args.mask_events, mask_context=args.mask_context)
  elif args.unsup == 'distant':
    u_exs, u_data = distant_train_examples(tokenizer, lm=args.lm, train=True, mask=args.mask, mask_events=args.mask_events, num_examples=args.unsup_num_examples)
  elif args.unsup.endswith('.pkl'):
    inputs = pickle.load(open(args.unsup, 'rb'))
    u_exs = inputs['exs']
    u_data = inputs['old_data']
    u_new_data = inputs['new_data']
  #  u_new_iter = itertools.cycle(u_exs)
  else:
    print('invalid option for args.unsup')
    exit()
  if args.unsup_num_examples and args.unsup != 'distant':
    u_exs = u_exs[-args.unsup_num_examples:]
  print(len(u_exs), "unsup examples loaded")
  #u_iter = itertools.cycle(u_exs)
  u_idxs = list(range(len(u_exs)))
  random.shuffle(u_idxs)
  u_idxs_pos = 0

OUTPUT_DIR = args.output_dir if args.output_dir else "models/scratch/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

if len(all_exs) == 0:
  print("no dataset specified")
elif len(all_exs) == 1:
  print("one dataset specified")
  exs = all_exs[0]
  data = all_data[0]
  if args.prune:
    pruner = Pruner()
    new_exs = []
    if args.prune_choice and args.prune_choice == 'attr':
      attributor = Attributor(model, tokenizer, device)
    for ex in exs:
      if args.prune_choice and args.prune_choice == 'attr':
        exs.append(pruner.get_pruned_example(ex, choice='attr', attributor=attributor))
      else:
        exs.append(pruner.get_pruned_example(ex))
      if len(exs) % 1000 == 0:
        print(len(exs))
    pickle.dump(new_exs, open(OUTPUT_DIR+"pruned.emb", "wb"))
    exs = exs + new_exs
    data = convert_distant_examples_to_features(exs, tokenizer, MAX_SEQ_LENGTH, DOC_STRIDE, mask=args.mask)
    data = make_tensor_dataset(data,model=args.lm)
else:
  print("using multiple data sources")
  inputs = []
  for i in range(len(all_data[0].tensors)):
    inputs.append(torch.cat([d.tensors[i] for d in all_data]))

  exs = list(chain(*all_exs))
  data = TensorDataset(*inputs) 


data_sampler = RandomSampler(data)
dataloader = DataLoader(data, sampler=data_sampler, batch_size=TRAIN_BATCH_SIZE)

print(len(data), len(exs), "examples loaded")

num_train_optimization_steps = int(len(data) / TRAIN_BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS) * NUM_TRAIN_EPOCHS
print(num_train_optimization_steps, "optimization steps")
num_warmup_steps = WARMUP_PROPORTION * num_train_optimization_steps

model.to(device)

# Prepare optimizer
param_optimizer = list(model.named_parameters())

# hack to remove pooler, which is not used
# thus it produce None grad that break apex
param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

optimizer = AdamW(optimizer_grouped_parameters,
                  lr=LEARNING_RATE,
                  correct_bias=False)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_optimization_steps)  # PyTorch scheduler

if args.serialize:
  inputs = {"exs" : exs, "data" : data}
  pickle.dump(inputs, open(OUTPUT_DIR+"inputs.pkl", 'wb'))

logfile = open(OUTPUT_DIR + "/log.txt", "w+")
print(args, file=logfile)
print(len(data), len(exs), "examples loaded", file=logfile)
count_labels(exs, file=logfile)
count_labels(exs)
print("learning_rate", LEARNING_RATE, file=logfile)

global_step = 0
num_epochs = args.epochs if not args.epochs is None else int(NUM_TRAIN_EPOCHS)
if num_epochs == 0:
  exit()
model.train()
exs_cpy = exs


all_preds = 0
high_conf_preds = 0


UNSUP_BATCH_SIZE = args.unsup_batch if args.unsup_batch else int(TRAIN_BATCH_SIZE/2)

ne_replacer = NEReplacer()

def get_new_example(old_example, method="random"):
  if method == "random":
    method = random.choice(["prune", "ne"])
    if method == "prune":
        new_example = pruner.get_pruned_example(old_example)
        if new_example == old_example:
            try:
                new_example = ne_replacer.replace(old_example)
            except:
                new_example = old_example
    if method == "ne":
        try:
            new_example = ne_replacer.replace(old_example)
        except:
            new_example = old_example
        if new_example == old_example:
            new_example = pruner.get_pruned_example(old_example)
  elif method == "prune":
    return pruner.get_pruned_example(old_example)
  elif method == "ne":
    try:
      new_example = ne_replacer.replace(old_example)
    except:
      new_example = old_example
  return new_example

def get_uda_examples(idxs, method="random"):
  #print(idxs)
  examples = [u_exs[i] for i in idxs]
  if method == "random_mask":
    examples = u_exs[idxs]
    new_examples  = apply_random_mask(examples, tokenizer, threshold=args.random_mask)
    return examples, new_examples
  old_examples = []
  new_examples = []
  #while len(old_examples) < UNSUP_BATCH_SIZE:
  #  old_ex = next(u_iter)
  for old_ex in examples:
    new_ex = get_new_example(old_ex, method)
    if new_ex != old_ex:
      old_examples.append(old_ex)
      new_examples.append(new_ex)
  return old_examples, new_examples

def unsup_loss(model):
  global all_preds, high_conf_preds, u_idxs_pos, u_idxs
  stop = min(u_idxs_pos+UNSUP_BATCH_SIZE, len(u_exs))
  idxs = u_idxs[u_idxs_pos:stop].copy()
  if stop == len(u_exs):
    random.shuffle(u_idxs)
    diff = u_idxs_pos + UNSUP_BATCH_SIZE - len(u_exs)
    idxs += u_idxs[:diff]
    u_idxs_pos = diff
  else:
    u_idxs_pos += UNSUP_BATCH_SIZE
  method = args.uda_method if args.uda_method else "random"
  unsup_batch, trans_batch = get_uda_examples(idxs=idxs, method=method)
  unsup_batch = convert_distant_examples_to_features(unsup_batch, tokenizer, MAX_SEQ_LENGTH, DOC_STRIDE, mask=args.unsup_mask)
  if len(unsup_batch) == 0:
    return None
  trans_batch = convert_distant_examples_to_features(trans_batch, tokenizer, MAX_SEQ_LENGTH, DOC_STRIDE, mask=args.unsup_mask)
  if len(unsup_batch) != len(trans_batch):
    return None
      
  unsup_batch = get_tensors(unsup_batch)
  trans_batch = get_tensors(trans_batch)
  batch = tuple(torch.cat((u, t), dim=0).to(device) for u, t in zip(unsup_batch, trans_batch))
  _, out, _ = model(*batch)
  u_out = out[:len(unsup_batch)]
  t_out = out[-len(trans_batch):]
  u_preds = softmax(u_out, dim=1).detach()
  threshold = .7 if args.unsup_conf else 0.0
  high_conf_mask = torch.max(u_preds, dim=1).values >= threshold
  all_preds += len(u_preds)
  u_preds = u_preds[high_conf_mask]
  if len(u_preds) == 0:
    return None
  high_conf_preds += len(u_preds)
  print("filtered to ", high_conf_preds / all_preds, file=logfile)
  t_out = t_out[high_conf_mask]
  t_logits = log_softmax(t_out, dim=1)

  loss_kldiv = kl_div(t_logits, u_preds, reduction='batchmean')
  if args.uda_weight:
      loss_kldiv *= args.uda_weight
  return loss_kldiv

for ep in trange(num_epochs, desc="Epoch"):
    if args.random_mask:
      exs  = apply_random_mask(exs_cpy, tokenizer, threshold=args.random_mask)
      data = convert_distant_examples_to_features(exs, tokenizer, MAX_SEQ_LENGTH, DOC_STRIDE, mask=args.mask)
      data = make_tensor_dataset(data, model=args.lm)
      data_sampler = RandomSampler(data)
      dataloader = DataLoader(data, sampler=data_sampler, batch_size=TRAIN_BATCH_SIZE)
    if args.prune_rep:
      pruner = Pruner()
      del exs
      torch.cuda.empty_cache()
      exs = []
      if args.prune_choice and args.prune_choice == 'attr':
        attributor = Attributor(model, tokenizer, device)
      for ex in exs_cpy:
        if args.prune_choice and args.prune_choice == 'attr':
          exs.append(pruner.get_pruned_example(ex, choice='attr', attributor=attributor))
        else:
          exs.append(pruner.get_pruned_example(ex))
        if len(exs) % 500 == 0:
          print(len(exs))
      exs = exs + exs_cpy
      data = convert_distant_examples_to_features(exs, tokenizer, MAX_SEQ_LENGTH, DOC_STRIDE, mask=args.mask)
      data = make_tensor_dataset(data, model=args.lm)
      data_sampler = RandomSampler(data)
      dataloader = DataLoader(data, sampler=data_sampler, batch_size=TRAIN_BATCH_SIZE)
    #pickle.dump(new_exs, open(OUTPUT_DIR+"pruned.emb", "wb"))
      model.to(device)
    last_loss_kldiv = 0
    for step, batch in enumerate(tqdm(dataloader, desc="Iteration " + str(ep), disable=args.disable_tqdm)):
        bbatch = tuple(t.to(device) for t in batch) # multi-gpu does scattering it-self
        loss, _, _ = model(*bbatch)

        if args.unsup:
          loss_kldiv = unsup_loss(model)
          if loss_kldiv:
            last_loss_kldiv = loss_kldiv.item()
            loss += loss_kldiv
        
        loss.backward()
        if step % 100 == 0:
          print("Loss: %.3f at step %d" %(loss.item(), step), file=logfile)
          if args.print_loss:
            print("Loss: %.3f at step %d" %(loss.item(), step))
          if args.unsup and last_loss_kldiv:
            print("Unsup Loss: %.3f at step %d" %(last_loss_kldiv, step), file=logfile)
            if args.print_loss:
              print("Unsup Loss: %.3f at step %d" %(last_loss_kldiv, step))
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        global_step += 1
       
        if n_gpu == 1:
          del bbatch
          torch.cuda.empty_cache()
    
    # Save a trained model, configuration and tokenizer
    model_output_dir = OUTPUT_DIR + "/output_" + str(ep) + "/"
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

