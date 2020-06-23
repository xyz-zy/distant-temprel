'''
train.py: script for training
'''
import argparse
import os
import sys
import torch

from itertools import chain
from torch.nn.functional import kl_div, softmax, log_softmax
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset, ConcatDataset)
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup

from constants import *
from load_data import get_train_data
from modeling import load_model_and_tokenizer
#from modeling import  make_tensor_dataset, get_tensors
from modeling import BertForMatres, RobertaForMatres, ElectraForMatres
#from uda.uda import unsup_loss
from utils import count_labels


def train(args):
    global TRAIN_BATCH_SIZE, LEARNING_RATE

    TRAIN_BATCH_SIZE = args.batch if args.batch else TRAIN_BATCH_SIZE
    print("train batch size", TRAIN_BATCH_SIZE)
    LEARNING_RATE = args.lr if args.lr else LEARNING_RATE
    print("learning_rate", LEARNING_RATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(device, n_gpu)

    model, tokenizer = load_model_and_tokenizer(lm=args.lm,
                                                model_dir=args.model_dir)

    all_exs = []
    all_data = []

    if not args.num_examples:
        args.num_examples = [None] * len(args.data)
    else:
        args.num_examples += [None] * (len(args.data) - len(args.num_examples))
    print(args.data, args.num_examples)

    for data_source, num_exs in zip(args.data, args.num_examples):
        exs, data = get_train_data(data_source, tokenizer, lm=args.lm,
                       num_examples=num_exs, mask=args.mask, distant_source=args.distant_source)
        all_exs.append(exs)
        all_data.append(data)

    '''
    if args.unsup:
      if args.unsup.endswith(".pkl"):
        inputs = pickle.load(open(args.unsup, 'rb'))
        u_exs = inputs['exs']
        u_data = inputs['old_data']
        u_new_data = inputs['new_data']
      else:
        assert args.unsup in set(["matres", "udst"])
        u_exs, u_data = get_train_data(args.unsup, lm=arg.lm, num_examples=args.unsup_num_examples, mask=args.mask)  
      print(len(u_exs), "unsup examples loaded")
      UNSUP_BATCH_SIZE = args.unsup_batch if args.unsup_batch else int(TRAIN_BATCH_SIZE/2)
      uda_dataset = UdaDataset(u_exs, UNSUP_BATCH_SIZE)
    '''

    OUTPUT_DIR = args.output_dir if args.output_dir else "models/scratch/"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if len(all_exs) == 0:
        print("no dataset specified")
    elif len(all_exs) == 1:
        print("one dataset specified")
        exs = all_exs[0]
        data = all_data[0]
    else:
        print("using multiple data sources")
        inputs = []
        for i in range(len(all_data[0].tensors)):
            inputs.append(torch.cat([d.tensors[i] for d in all_data]))

        exs = list(chain(*all_exs))
        data = TensorDataset(*inputs)


    data_sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=data_sampler,
                            batch_size=TRAIN_BATCH_SIZE)

    print(len(data), len(exs), "examples loaded")

    num_train_optimization_steps = int(
        len(data) / TRAIN_BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS) * NUM_TRAIN_EPOCHS
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
        {'params': [p for n, p in param_optimizer if not any(
        nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
        nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
              lr=LEARNING_RATE,
              correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                            num_warmup_steps=num_warmup_steps,
                            num_training_steps=num_train_optimization_steps)  # PyTorch scheduler

    if args.serialize:
        inputs = {"exs": exs, "data": data}
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

    for ep in trange(num_epochs, desc="Epoch"):
        last_loss_kldiv = 0
        for step, batch in enumerate(tqdm(dataloader,
                          desc="Iteration " + str(ep),
                          disable=args.disable_tqdm)):
            bbatch = tuple(t.to(device) for t in batch)
            loss, _, _ = model(*bbatch)

            '''
            if args.unsup:
                loss_kldiv = unsup_loss(model)
                if loss_kldiv:
                last_loss_kldiv = loss_kldiv.item()
                loss += loss_kldiv
            '''

            loss.backward()
            if step % 100 == 0:
                print("Loss: %.3f at step %d" % (loss.item(), step), file=logfile)
                # if args.unsup and last_loss_kldiv:
                #  print("Unsup Loss: %.3f at step %d" %(last_loss_kldiv, step), file=logfile)
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step += 1

            # Save a trained model, configuration and tokenizer
            model_output_dir = OUTPUT_DIR + "/output_" + str(ep) + "/"
            if not os.path.exists(model_output_dir):
                os.makedirs(model_output_dir)
            model.save_pretrained(model_output_dir)
            tokenizer.save_pretrained(model_output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                        nargs='+',
                        #choices=['matres','udst','beforeafter',
                        #         'distant','beforeafter_yelp'],
                        help='training data sources, can specifiy multiple. choose from {matres, udst, beforeafter, distant} or specify path to .pkl containing serialized data')
    parser.add_argument('--output_dir',
                        help='directory path to save model checkpoints')
    parser.add_argument('--model_dir',
                        help='checkpoint directory to load model; if None, then will load pre-trained transformers model')
    parser.add_argument('--distant_source',
                        choices=['afp','apw','nyt','cna','wpb', 'even'],
                        help='filter DistantTimex data to single source, or even split')
    parser.add_argument('--num_examples', nargs='+', type=int,
                        help='number of examples to be used from each --data source')
    parser.add_argument('--mask', action='store_true',
                        help='mask BeforeAfter connectives and DistantTimex timexes when training')
    parser.add_argument('--lm',
                        choices=['bert', 'bert-large', 'roberta', 'electra', 'electra-large'],
                        help='language model architecture for training, must be same as --model_dir, if specified.')
    parser.add_argument('--serialize', action='store_true',
                        help='save training examples and TensorDataset to `inputs.pkl file in --output_dir')
    parser.add_argument('--epochs', type=int,
                        help='number of epochs for fine-tuning, defaults to constants.py')
    parser.add_argument('--batch', type=int,
                        help='train batch size, defaults to constants.py')
    parser.add_argument('--lr', type=float,
                    help='learning rate, defaults to constants.py')
    #parser.add_argument('--mask_events', action='store_true')
    #parser.add_argument('--mask_context', action='store_true')
    #parser.add_argument('--unsup', help='matres')
    #parser.add_argument('--unsup_conf', action='store_true')
    #parser.add_argument('--unsup_batch', type=int)
    #parser.add_argument('--unsup_num_examples', type=int)
    #parser.add_argument('--unsup_mask', action='store_true')
    #parser.add_argument('--uda_method', help='prune,ne')
    #parser.add_argument('--uda_weight', type=float)
    parser.add_argument('--disable_tqdm', action='store_true')
    args = parser.parse_args()
    print(args)

    if not args.lm and not args.data:
        raise RuntimeError("Please specifiy both --lm and --data; see -h or --help for details.")   

    train(args)

