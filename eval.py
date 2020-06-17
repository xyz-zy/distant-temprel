'''
eval.py: script for evaluating models
'''
import pandas as pd
import argparse
import glob
import os
import pickle
import re
import sys
import time
import torch

from tqdm import tqdm, trange
from transformers import *
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

# Import TRC model from repository
from modeling import make_tensor_dataset
from modeling import BertForMatres, RobertaForMatres, ElectraForMatres

from constants import *
from load_data import *
from metrics import *
from test import count_labels
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--data', nargs='+',
                    help='gigaword,matres_train,matres_dev,distant_train,distant_dev,udst_train,udst_dev,udst_dev_maj,udst_test_maj,udst_dev_maj_conf,udst_dev_ilp,udst_test_maj_conf_nt')
parser.add_argument('--epoch')
parser.add_argument('--untrained', action='store_true')
parser.add_argument('--model_dir', nargs='+')
parser.add_argument('--num_examples', help='only supported for distant_train')
parser.add_argument('--mask', action='store_true')
parser.add_argument('--lm', help='bert,roberta,electra')
parser.add_argument('--output_results', action='store_true',
                    help='toggle store results to csv')
parser.add_argument('--save_emb', action='store_true')
parser.add_argument('--mask_events', action='store_true')
parser.add_argument('--mask_context', action='store_true')
parser.add_argument('--pkl', help='path to pickle file with examples')
parser.add_argument('--ens_out_dir', help='directory for output ensemble fiels')
args = parser.parse_args()
print(args)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
print(device, n_gpu)

if not args.model_dir:
  print("please specify --model_dir")
  exit()


def get_data(tokenizer, data_source):
    if data_source == 'gigaword':
        print("using beforeafter gigaword examples")
        examples, data = gigaword_examples(tokenizer, lm=args.lm)
    elif data_source == 'matres_train':
        print("using matres training examples")
        examples, data = matres_train_examples(
            tokenizer, lm=args.lm, mask_events=args.mask_events, mask_context=args.mask_context)
    elif data_source == 'matres_dev':
        print("using matres dev examples")
        examples, data = matres_dev_examples(
            tokenizer, lm=args.lm, mask_events=args.mask_events, mask_context=args.mask_context)
    elif data_source == 'matres_test':
        print("using matres test examples")
        examples, data = matres_test_examples(
            tokenizer, lm=args.lm, mask_events=args.mask_events, mask_context=args.mask_context)
    elif data_source == 'distant_train':
        print("using distant training examples")
        examples, data = distant_train_examples(
            tokenizer, lm=args.lm, train=False, mask=args.mask, mask_events=args.mask_events)
        if args.num_examples and int(args.num_examples) > len(examples):
            more_exs, more_data = distant_parsed_examples(tokenizer, lm=args.lm, train=False, num_examples=int(
                args.num_examples)-len(examples), mask_events=args.mask_events)
            examples.extend(more_exs)
            data = convert_distant_examples_to_features(examples=examples,
                                                        tokenizer=tokenizer,
                                                        max_seq_length=MAX_SEQ_LENGTH,
                                                        doc_stride=DOC_STRIDE,
                                                        mask=args.mask,
                                                        mask_events=args.mask_events)
            data = make_tensor_dataset(data, model=args.lm)
    elif data_source == 'distant_dev':
        print("using distant test examples")
        examples, data = distant_test_examples(
            tokenizer, lm=args.lm, mask=args.mask, mask_events=args.mask_events)
    elif data_source == 'udst_train':
        print("using UDS-T train examples")
        examples, data = udst(tokenizer, lm=args.lm, split="train",
                              mask_events=args.mask_events)
    elif data_source == 'udst_dev':
        print("using UDS-T dev examples")
        examples, data = udst(tokenizer, lm=args.lm, split="dev",
                              mask_events=args.mask_events)
    elif data_source == 'udst_dev_maj':
        print("using UDS-T dev examples, majority vote")
        examples, data = udst_majority(
            tokenizer, lm=args.lm, split="dev", mask_events=args.mask_events)
        count_labels(examples)
    elif data_source == 'udst_dev_maj_nt':
        print("using UDS-T dev examples, majority vote, no ties")
        examples, data = udst_majority(
            tokenizer, lm=args.lm, split="dev", mask_events=args.mask_events, ties=False)
        count_labels(examples)
    elif data_source == 'udst_dev_maj_conf':
        print("using UDS-T dev examples, majority vote, conf-broken")
        examples, data = udst(
            tokenizer, lm=args.lm, split="dev", example_dir="udst/DecompTime/maj_conf/", mask_events=args.mask_events)
        count_labels(examples)
    elif data_source == 'udst_dev_maj_conf_nt':
        print("using UDS-T dev examples, majority vote, conf-broken, no ties")
        examples, data = udst(
            tokenizer, lm=args.lm, split="dev", example_dir="udst/DecompTime/maj_conf_nt/", mask_events=args.mask_events)
        count_labels(examples)
    elif data_source == 'udst_dev_ilp':
        print("using UDS-T dev examples, ilp")
        examples, data = udst(
            tokenizer, lm=args.lm, split="dev", example_dir="udst/DecompTime/ilp/", mask_events=args.mask_events)
        count_labels(examples)
    elif data_source == 'udst_test_maj':
        print("using UDS-T test examples, majority vote")
        examples, data = udst_majority(
            tokenizer, lm=args.lm, split="test", mask_events=args.mask_events)
    elif data_source == 'udst_test_maj_conf_nt':
        print("using UDS-T dev examples, majority vote, no ties")
        examples, data = udst_majority(
            tokenizer, lm=args.lm, split="test", example_dir="udst/DecompTime/maj_conf_nt/", mask_events=args.mask_events)
        count_labels(examples)
    elif data_source == 'pkl':
        assert args.pkl is not None
        examples = pickle.load(open(args.pkl, "rb"))
        print(len(examples))
        data = convert_distant_examples_to_features(examples=examples,
                                                    tokenizer=tokenizer,
                                                    max_seq_length=MAX_SEQ_LENGTH,
                                                    doc_stride=DOC_STRIDE,
                                                    mask=args.mask,
                                                    mask_events=args.mask_events)
        data = make_tensor_dataset(data, model=args.lm)
        count_labels(examples)
    else:
        print("please specify valid dataset")
        exit()
    return examples, data


def eval(model_dir, epoch_num, data_source):
    print(model_dir, epoch_num, file=logfile)
    if args.lm == 'roberta':
        model = RobertaForMatres.from_pretrained(model_dir)
        tokenizer = RobertaTokenizer.from_pretrained(model_dir)
    elif args.lm == 'bert':
        model = BertForMatres.from_pretrained(model_dir)
        tokenizer = BertTokenizer.from_pretrained(model_dir)
    elif args.lm.startswith('electra'):
        model = ElectraForMatres.from_pretrained(model_dir)
        tokenizer = ElectraTokenizer.from_pretrained(model_dir)
    else:
        print("Please specifiy valid pretrained model", file=sys.stderr)
        exit()

    examples, data = get_data(tokenizer, data_source)
    data_sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=data_sampler,
                            batch_size=EVAL_BATCH_SIZE)

    model.to(device)
    model.eval()
    all_results = []
    all_labels = []
    print("Start evaluating", file=logfile)

    if args.save_emb:
        all_e1_hidden = []
        all_e2_hidden = []

    for batch in tqdm(dataloader, desc="Evaluating", disable=LOCAL_RANK not in [-1, 0]):
        if args.lm == 'roberta':
            input_ids, attention_masks, label_ids, e1_pos, e2_pos = batch
        else:
            input_ids, token_type_ids, attention_masks, label_ids, e1_pos, e2_pos = batch
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            _, out, hidden = model(*batch)
            val, guess_idxs = out.max(1)
            for idx, guess in enumerate(guess_idxs):
                all_results.append(guess.item())
                all_labels.append(label_ids[idx].item())
            if args.save_emb:
                for e1, e2 in zip(hidden[0], hidden[1]):
                    all_e1_hidden.append(e1.cpu())
                    all_e2_hidden.append(e2.cpu())

    if args.save_emb:
        print(len(all_e1_hidden))
        output = []
        for ex, e1_hidden, e2_hidden, label, guess in zip(examples, all_e1_hidden, all_e2_hidden, all_labels, all_results):
            output.append(ExampleWithEmbeddings(
                ex, e1_hidden, e2_hidden, label, guess))
        pickle.dump(output, open(model_dir + data_source + ".emb", 'wb'))
        print(len(output))
    if args.output_results:
        if "matres" in data_source:
            df = [[ex.filename, ex.sent1 if ex.sent1 == ex.sent2 else ex.sent1+ex.sent2, ex.sent1[ex.e1_idx], ex.e1_eid,
                   ex.sent2[ex.e2_idx], ex.e2_eid, ex.label, CLASSES[result]] for ex, result in zip(examples, all_results)]
            df = pd.DataFrame(df, columns=[
                              "filename", "text", "e1", "e1_eid", "e2", "e2_eid", "label", "guess"])
            df.to_csv(model_dir + data_source + ".csv")
        else:
            df = [[ex.sent1 if ex.sent1 == ex.sent2 else ex.sent1+ex.sent2, ex.sent1[ex.e1_idx],
                   ex.sent2[ex.e2_idx], ex.label, CLASSES[result]] for ex, result in zip(examples, all_results)]
            df = pd.DataFrame(
                df, columns=["text", "e1", "e2", "label", "guess"])
            df.to_csv(model_dir + data_source + ".csv")

    if "matres" in data_source:
        said_default_guesses = []
        no_said_guesses = []
        no_said_labels = []
        said_correct = 0
        said_incorrect = 0
        same_sent = 0
        diff_sent = 0
        same_sent_correct = 0
        diff_sent_correct = 0
        for i, (guess, label) in enumerate(zip(all_results, all_labels)):
            ex = examples[i]
            if ex.e1_sentence_num == ex.e2_sentence_num:
                same_sent += 1
                if guess == label:
                    same_sent_correct += 1
            else:
                diff_sent += 1
                if guess == label:
                    diff_sent_correct += 1
            if ex.sent1[ex.e1_idx] == 'said' or ex.sent2[ex.e2_idx] == 'said':
                if guess == label:
                    said_correct += 1
                else:
                    said_incorrect += 1
                if ex.sent1[ex.e1_idx] == 'said':
                    said_default_guesses.append(CLASSES.index('AFTER'))
                else:
                    said_default_guesses.append(CLASSES.index('BEFORE'))
            else:
                said_default_guesses.append(guess)
                no_said_guesses.append(guess)
                no_said_labels.append(label)
        print("said_default_guess", file=logfile)
        get_metrics(all_labels, said_default_guesses, logfile)
        fig, ax = plot_confusion_matrix(
            all_labels, said_default_guesses, "said_default_guess", logfile)
        print("no_said", file=logfile)
        get_metrics(no_said_labels, no_said_guesses, logfile)
        fig, ax = plot_confusion_matrix(
            no_said_labels, no_said_guesses, "no_said", logfile)
        plt.close()
        print("Number of same sentence examples, ", same_sent, file=logfile)
        print("Number of diff sentence examples, ", diff_sent, file=logfile)
        print("Same sentence acc:\t", same_sent_correct / same_sent, file=logfile)
        print("Diff sentence acc:\t", diff_sent_correct / diff_sent, file=logfile)
        print("Said exs acc:\t", said_correct /
              (said_correct + said_incorrect), file=logfile)
        print("said correct:\t", said_correct, file=logfile)
        print("said incorrect:\t", said_incorrect, file=logfile)
    print(args, file=logfile)
    metrics = get_metrics(all_labels, all_results, logfile)
    fig_title = "Untrained" if args.untrained else str(epoch_num) + " Epochs"
    fig, ax = plot_confusion_matrix(
        all_labels, all_results, fig_title, logfile)
    fig_save_path = model_base_dir if args.untrained else model_dir
    fig_save_path = fig_save_path + data_source + ".png"
    fig.savefig(fig_save_path)
    return metrics, {"preds": all_results, "labels": all_labels}


metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
macro_metric_names = ['Precision', 'Recall', 'F1']
all_metrics = {}

logfile = None
outfile = None


def eval_on_data(model_base_dir, data_source):
    global logfile, outfile
    if data_source == "distant_dev":
        if args.mask:
            out_basename = data_source + "_mask"
        else:
            out_basename = data_source + "_nomask"
    else:
        out_basename = data_source
    if args.mask_events:
        out_basename += "_me"
    all_results = []
    if args.epoch:
        model_dir = model_base_dir + "output_" + args.epoch + "/"
        logfile = open(model_dir + out_basename + ".log", "w+")
        outfile = open(model_dir + out_basename + ".txt", "w+")
        all_metrics[args.epoch], results = eval(
            model_dir, int(args.epoch), data_source)
        all_results.append(results)
    else:
        print("evaluating over all epochs")
        logfile = open(model_base_dir + out_basename + ".log", "w+")
        outfile = open(model_base_dir + out_basename + ".txt", "w+")
        macro_outfile = open(
            model_base_dir + out_basename + "_macro.txt", "w+")
        for model_dir in glob.iglob(model_base_dir + "output_*"):
            print(model_dir)
            epoch = re.findall("\d+", model_dir[len(model_base_dir):])[0]
            print("found epoch", epoch)
            all_metrics[epoch], results = eval(model_dir+"/", epoch, data_source)
            all_results.append(results)
    epochs = sorted(list(all_metrics.keys()))
    for epoch in epochs:
        print("Epoch %s" % str(epoch), file=outfile)
        for metric_name in metric_names:
            print(metric_name + "\t%.4f" %
                  all_metrics[epoch][metric_name], file=outfile)
        print("", file=outfile)

        print("Epoch %s" % str(epoch), file=macro_outfile)
        for cl in MACRO_CLASSES:
            print(cl, file=macro_outfile)
            for metric_name in macro_metric_names:
                print(metric_name + "\t%.4f" %
                      all_metrics[epoch][cl][metric_name], file=macro_outfile)
        print("Macro", file=macro_outfile)
        for metric_name in macro_metric_names:
            print("Macro " + metric_name + "\t%.4f" %
                  all_metrics[epoch]['Macro'][metric_name], file=macro_outfile)
        print("", file=macro_outfile)

    logfile.close()
    outfile.close()
    macro_outfile.close()
    return all_results

for data_source in args.data:
    all_model_results = []
    for model_dir in args.model_dir:
        all_epoch_results = eval_on_data(model_dir, data_source)
        all_model_results.append(all_epoch_results)
    if len(args.model_dir) > 1:
        if args.ens_out_dir:
            ensemble_file = open(args.ens_out_dir + data_source + ".ens", "w+")
        else:
            ts = int(time.time())
            ensemble_file = open(str(ts) + ".log", "w+")
        print(args, file=ensemble_file)

        min_epochs = min([len(model_results) for model_results in all_model_results])
        for i in range(min_epochs):
            print("Epoch " + str(i), file=ensemble_file)
            all_preds = [model_results[i]['preds'] for model_results in all_model_results]
            print(len(all_preds))
            print(len(all_preds[0]))
            all_preds = list(map(list, zip(*all_preds)))
            print(all_preds[:4])
            all_preds = [max(set(lst), key=lst.count) for lst in all_preds]
            print(all_preds[:4])
            labels = all_model_results[0][i]['labels']

            get_metrics(labels, all_preds, ensemble_file)

