from constants import *
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import statistics
matplotlib.use('Agg')


MACRO_CLASSES = ['AFTER', 'BEFORE', 'EQUAL']


def accuracy(gold_labels, guess_labels):
    correct = 0
    for gold, guess in zip(gold_labels, guess_labels):
        if gold == guess:
            correct += 1
    return correct / len(gold_labels)


def precision(gold_labels, guess_labels):
    s1_count = 0  # Number of examples where predicted is not "VAGUE"
    correct = 0  # Number of examples correct, not including correct "VAGUE" examples
    for gold, guess in zip(gold_labels, guess_labels):
        if guess != VAGUE:
            if gold == guess:
                correct += 1
            s1_count += 1
    return 0 if s1_count == 0 else correct / s1_count


def recall(gold_labels, guess_labels):
    s2_count = 0  # Number of examples where correct is not "VAGUE"
    correct = 0  # Number of examples correct, not including correct "VAGUE" examples
    for gold, guess in zip(gold_labels, guess_labels):
        if gold != VAGUE:
            if gold == guess:
                correct += 1
            s2_count += 1
    return 0 if s2_count == 0 else correct / s2_count


def f1(gold_labels, guess_labels):
    p = precision(gold_labels, guess_labels)
    r = recall(gold_labels, guess_labels)
    return 2 * p * r / (p + r)


def faware(gold_labels, guess_labels):
    # TODO: understand later
    pass


def count_labels(gold_labels, guess_labels, logfile):
    guesses_count = {}
    for cls in range(len(CLASSES)):
        guesses_count[cls] = [0, 0]
    for gold, guess in zip(gold_labels, guess_labels):
        if gold == guess:
            guesses_count[guess][0] += 1
        else:
            guesses_count[guess][1] += 1

    print("CLASS\tTOTAL\tCORRECT\tWRONG", file=logfile)
    for cls, stats in guesses_count.items():
        print("\t".join([CLASSES[cls], str(stats[0]+stats[1]),
                         str(stats[0]), str(stats[1])]), file=logfile)


def get_class_metrics(gold_labels, guess_labels, class_str):
    metrics = {}
    metrics['Precision'] = precision([gold for gold, guess in zip(gold_labels, guess_labels) if guess == CLASSES.index(
        class_str)], [guess for guess in guess_labels if guess == CLASSES.index(class_str)])
    metrics['Recall'] = precision([gold for gold in gold_labels if gold == CLASSES.index(class_str)], [
                                  guess for gold, guess in zip(gold_labels, guess_labels) if gold == CLASSES.index(class_str)])
    if metrics['Precision'] == 0 and metrics['Recall'] == 0:
        metrics['F1'] = 0
    else:
        metrics['F1'] = 2 * (metrics['Precision'] * metrics['Recall']) / \
            (metrics['Precision'] + metrics['Recall'])
    return metrics


def get_metrics(gold_labels, guess_labels, logfile):
    assert len(gold_labels) == len(guess_labels)
    print("Number of examples: ", len(gold_labels), file=logfile)
    print("Accuracy: ", accuracy(gold_labels, guess_labels), file=logfile)
    print("Precision: ", precision(gold_labels, guess_labels), file=logfile)
    print("Recall: ", recall(gold_labels, guess_labels), file=logfile)
    print("F1: ", f1(gold_labels, guess_labels), file=logfile)

    metrics = {}
    metrics['Accuracy'] = accuracy(gold_labels, guess_labels)
    metrics['Precision'] = precision(gold_labels, guess_labels)
    metrics['Recall'] = recall(gold_labels, guess_labels)
    metrics['F1'] = f1(gold_labels, guess_labels)

    print(metrics['Accuracy'], file=logfile)
    print(metrics['Precision'], file=logfile)
    print(metrics['Recall'], file=logfile)
    print(metrics['F1'], file=logfile)

    metrics['AFTER'] = get_class_metrics(gold_labels, guess_labels, 'AFTER')
    metrics['BEFORE'] = get_class_metrics(gold_labels, guess_labels, 'BEFORE')
    metrics['EQUAL'] = get_class_metrics(gold_labels, guess_labels, 'EQUAL')
    metrics['Macro'] = {}
    metrics['Macro']['Precision'] = statistics.mean(
        [metrics[cl]['Precision'] for cl in MACRO_CLASSES])
    metrics['Macro']['Recall'] = statistics.mean(
        [metrics[cl]['Recall'] for cl in MACRO_CLASSES])
    metrics['Macro']['F1'] = statistics.mean(
        [metrics[cl]['F1'] for cl in MACRO_CLASSES])
    return metrics


def plot_confusion_matrix(true_labels, predictions, title, logfile):
    cmap = plt.cm.Blues

    classes = ["AFTER", "BEFORE", "EQUAL", "VAGUE"]
    cm = confusion_matrix(true_labels, predictions)

    total = sum(sum(cm))
    print(total, file=logfile)
    for row in cm:
        print("\t".join([str(n) for n in row]), file=logfile)
    cm = np.array([c / total for c in cm])

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.3f'  # if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig, ax


def get_fine_metrics(examples, guesses, labels, logfile=None):
    """
    Generates metrics for "said" examples and single- vs. cross-sentence
    examples.
    """
    said_default_guesses = []
    no_said_guesses = []
    no_said_labels = []
    said_correct = 0
    said_incorrect = 0
    same_sent = 0
    diff_sent = 0
    same_sent_correct = 0
    diff_sent_correct = 0
    for i, (ex, guess, label) in enumerate(zip(examples, guesses, labels)):
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
    get_metrics(labels, said_default_guesses, logfile)
    fig, ax = plot_confusion_matrix(labels, said_default_guesses,
                                    "said_default_guess", logfile)
    print("no_said", file=logfile)
    get_metrics(no_said_labels, no_said_guesses, logfile)
    fig, ax = plot_confusion_matrix(no_said_labels, no_said_guesses,
                                    "no_said", logfile)
    print("Number of same sentence examples, ", same_sent, file=logfile)
    print("Number of diff sentence examples, ", diff_sent, file=logfile)
    print("Same sentence acc:\t",
          same_sent_correct / same_sent, file=logfile)
    print("Diff sentence acc:\t",
          diff_sent_correct / diff_sent, file=logfile)
    print("Said exs acc:\t", said_correct /
          (said_correct + said_incorrect), file=logfile)
    print("said correct:\t", said_correct, file=logfile)
    print("said incorrect:\t", said_incorrect, file=logfile)


def output_results(data_source, examples, guesses, filename):
    if "matres" in data_source:
        df = [[ex.filename, ex.sent1 if ex.sent1 == ex.sent2 else ex.sent1+ex.sent2,
               ex.sent1[ex.e1_idx], ex.e1_eid,
               ex.sent2[ex.e2_idx], ex.e2_eid,
               ex.label,
               CLASSES[result]] for ex, result in zip(examples, guesses)]
        df = pd.DataFrame(df, columns=[
            "filename", "text", "e1", "e1_eid", "e2", "e2_eid", "label", "guess"])
        df.to_csv(filename)
    else:
        df = [[ex.sent1 if ex.sent1 == ex.sent2 else ex.sent1+ex.sent2,
               ex.sent1[ex.e1_idx],
               ex.sent2[ex.e2_idx],
               ex.label,
               CLASSES[result]] for ex, result in zip(examples, guesses)]
        df = pd.DataFrame(df,
                          columns=["text", "e1", "e2", "label", "guess"])
        df.to_csv(filename)
