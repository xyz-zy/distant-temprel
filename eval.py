'''
eval.py: script for evaluating models
'''
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

from modeling import load_model_and_tokenizer, make_tensor_dataset
from modeling import BertForMatres, RobertaForMatres, ElectraForMatres

from constants import *
from load_data import *
from metrics import *
from test import count_labels
from utils import *


def get_data(tokenizer, data_source):
    if data_source.endswith("_train") or data_source.endswith(".pkl"):
        examples, data = get_train_data(data_source[:len("_train")],
                                        tokenizer,
                                        lm=args.lm,
                                        mask=args.mask)
    elif data_source == 'beforeafter':
        print("using beforeafter gigaword examples")
        examples, data = gigaword_examples(tokenizer, lm=args.lm)
    elif data_source == 'matres_dev':
        print("using matres dev examples")
        examples, data = matres_dev_examples(tokenizer,
                                             lm=args.lm)#,
                                             #mask_events=args.mask_events,
                                             #mask_context=args.mask_context)
    elif data_source == 'matres_test':
        print("using matres test examples")
        examples, data = matres_test_examples(tokenizer,
                                              lm=args.lm)#,
                                              #mask_events=args.mask_events,
                                              #mask_context=args.mask_context)
    elif data_source == 'distant_dev' or data_source == "distant_test":
        print("using distant test examples")
        examples, data = distant_test_examples(tokenizer,
                                               lm=args.lm,
                                               mask=args.mask)#,
                                               #mask_events=args.mask_events)
    elif data_source == 'udst_dev':
        print("using UDS-T dev examples")
        examples, data = udst(tokenizer, lm=args.lm, split="dev")#,
                              #mask_events=args.mask_events)
    elif data_source == 'udst_dev_maj':
        print("using UDS-T dev examples, majority vote")
        examples, data = udst_majority(
            tokenizer, lm=args.lm, split="dev")#), mask_events=args.mask_events)
        count_labels(examples)
    elif data_source == 'udst_dev_maj_conf_nt':
        print("using UDS-T dev examples, majority vote, conf-broken, no ties")
        examples, data = udst(tokenizer, lm=args.lm, split="dev",
                              example_dir="udst/maj_conf_nt/")#,
                              #mask_events=args.mask_events)
        count_labels(examples)
    elif data_source == 'udst_dev_ilp':
        print("using UDS-T dev examples, ilp")
        examples, data = udst(tokenizer, lm=args.lm, split="dev",
                              example_dir="udst/ilp/")#,
                              #mask_events=args.mask_events)
        count_labels(examples)
    elif data_source == 'udst_test_maj':
        print("using UDS-T test examples, majority vote")
        examples, data = udst_majority(tokenizer,
                                       lm=args.lm,
                                       split="test")#,
                                       #mask_events=args.mask_events)
    elif data_source == 'udst_test_maj_conf_nt':
        print("using UDS-T dev examples, majority vote, no ties")
        examples, data = udst_majority(tokenizer,
                                       lm=args.lm,
                                       split="test",
                                       example_dir="udst/maj_conf_nt/")#,
                                       #mask_events=args.mask_events)
        count_labels(examples)
    else:
        print("please specify valid dataset")
        exit()
    return examples, data


def eval(model_dir, epoch_num, data_source):
    print(model_dir, epoch_num, file=logfile)

    model, tokenizer = load_model_and_tokenizer(
        args.lm, model_dir=model_dir)

    examples, data = get_data(tokenizer, data_source)
    data_sampler = SequentialSampler(data)
    dataloader = DataLoader(data, sampler=data_sampler,
                            batch_size=EVAL_BATCH_SIZE)

    model.to(device)
    model.eval()
    all_results = []
    all_labels = []
    print("Start evaluating", file=logfile)

    #if args.save_emb:
    #    all_e1_hidden = []
    #    all_e2_hidden = []

    for batch in tqdm(dataloader, desc="Evaluating", disable=args.disable_tqdm):
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
            #if args.save_emb:
            #    for e1, e2 in zip(hidden[0], hidden[1]):
            #        all_e1_hidden.append(e1.cpu())
            #        all_e2_hidden.append(e2.cpu())

    #if args.save_emb:
    #    print(len(all_e1_hidden))
    #    output = []
    #    for ex, e1_hidden, e2_hidden, label, guess in zip(examples, all_e1_hidden, all_e2_hidden, all_labels, all_results):
    #        output.append(ExampleWithEmbeddings(
    #            ex, e1_hidden, e2_hidden, label, guess))
    #    pickle.dump(output, open(model_dir + data_source + ".emb", 'wb'))
    #    print(len(output))

    if args.output_results:
        output_results(data_source, examples, all_results,
                       model_dir+data_source+".csv")

    if "matres" in data_source:
        get_fine_metrics(examples, all_labels, all_results)

    print(args, file=logfile)
    metrics = get_metrics(all_labels, all_results, logfile)

    # Plots confusion matrix and outputs to file
    #fig_title = str(epoch_num) + " Epochs"
    #fig, ax = plot_confusion_matrix(
    #    all_labels, all_results, fig_title, logfile)
    #fig_save_path = model_dir
    #fig_save_path = fig_save_path + data_source + ".png"
    #fig.savefig(fig_save_path)

    return metrics, {"preds": all_results, "labels": all_labels}


metric_names = ['Accuracy', 'Precision', 'Recall', 'F1']
macro_metric_names = ['Precision', 'Recall', 'F1']
all_metrics = {}

logfile = None
outfile = None


def eval_on_data(model_base_dir, data_source):
    global logfile, outfile
    if data_source == "distant_dev" or data_source == "distant_test":
        if args.mask:
            out_basename = "distant_test_mask"
        else:
            out_basename = "distant_test_nomask"
    else:
        out_basename = data_source
    #if args.mask_events:
    #    out_basename += "_me"
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
            all_metrics[epoch], results = eval(model_dir+"/",
                                               epoch,
                                               data_source)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', nargs='+',
                        choices=['matres_train', 'matres_dev', 'matres_test',
                                 'beforeafter', 'distant_train', 'distant_test',
                                 'udst_train', 'udst_dev', 'udst_test',
                                 'udst_dev_maj', 'udst_test_maj',
                                 'udst_dev_maj_conf_nt',
                                 'udst_test_maj_conf_nt'],
                        help='data sources to evaluate on')
    parser.add_argument('--mask', action='store_true')
    parser.add_argument('--model_dir', nargs='+',
                        help="model diretory containing model checkpoint directories, i.e. /PATH/TO/MODEL_CHKPTS/ that contains output_<#>/; if multiple arguments are given, then will output ensembled evaluation results per epoch.")
    parser.add_argument('--epoch', type=int,
                        help="only evaluate one model checkpoint, from an 'output_<epoch>/' directory withint --model_dir")
    parser.add_argument('--lm', choices=['bert','roberta','electra'],
                        help='model architecture')
    parser.add_argument('--output_results', action='store_true',
                        help='output model results to csv')
    #parser.add_argument('--save_emb', action='store_true')
    #parser.add_argument('--mask_events', action='store_true')
    #parser.add_argument('--mask_context', action='store_true')
    parser.add_argument('--ens_out_dir',
                        help='directory for ensemble output files')
    parser.add_argument('--disable_tqdm', action='store_true')
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print(device, n_gpu)

    if not args.model_dir:
        print("please specify --model_dir")
        exit()

    for data_source in args.data:
        all_model_results = []
        for model_dir in args.model_dir:
            all_epoch_results = eval_on_data(model_dir, data_source)
            all_model_results.append(all_epoch_results)

        # If multiple model directories are specified, then ensembles
        # the models and writes output.
        if len(args.model_dir) > 1:
            if args.ens_out_dir:
                ensemble_file = open(args.ens_out_dir +
                                     data_source + ".ens", "w+")
            else:
                ts = int(time.time())
                ensemble_file = open(data_source + "_" + str(ts) + ".ens", "w+")
            print(args, file=ensemble_file)

            min_epochs = min([len(model_results)
                              for model_results in all_model_results])
            for i in range(min_epochs):
                print("Epoch " + str(i), file=ensemble_file)
                all_preds = [model_results[i]['preds']
                             for model_results in all_model_results]
                print(len(all_preds))
                print(len(all_preds[0]))
                all_preds = list(map(list, zip(*all_preds)))
                print(all_preds[:4])
                all_preds = [max(set(lst), key=lst.count) for lst in all_preds]
                print(all_preds[:4])
                labels = all_model_results[0][i]['labels']

                get_metrics(labels, all_preds, ensemble_file)
                print_confusion_matrix(labels, all_preds, ensemble_file)

