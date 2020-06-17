"""
uda.py: utils for unsupervised KLD loss with data augmentation. 
"""
import random

from utils import apply_random_mask

from .ne import NEReplacer
from .prune import Pruner

class UdaModule(object):
    def __init__(self, examples, batch_size):
        self.examples = examples
        self.batch_size = batch_size
        self.idxs = list(range(len(examples)))
        self.idx_pos = 0

        self.all_preds = 0
        self.high_conf_preds = 0
        
        self.pruner = Pruner()
        self.ne_replacer = NEReplacer()

    def get_new_example(self, old_example, method="random"):
        if method == "random":
            method = random.choice(["prune", "ne"])
            if method == "prune":
                new_example = self.pruner.get_pruned_example(old_example)
                if new_example == old_example:
                    try:
                        new_example = self.ne_replacer.replace(old_example)
                    except:
                        new_example = old_example
            if method == "ne":
                try:
                    new_example = self.ne_replacer.replace(old_example)
                except:
                    new_example = old_example
                if new_example == old_example:
                    new_example = self.pruner.get_pruned_example(old_example)
        elif method == "prune":
            return self.pruner.get_pruned_example(old_example)
        elif method == "ne":
            try:
                new_example = self.ne_replacer.replace(old_example)
            except:
                new_example = old_example
        return new_example
   
    def get_new_examples(self, tokenizer, idxs, method="random", random_mask=None):
        batch = [self.examples[i] for i in idxs]
        if method == "random_mask":
            new_batch = apply_random_mask(batch, tokenizer, threshold=random_mask)
            return batch, new_batch
        old_batch = []
        new_batch = []
        for old_ex in examples:
            new_ex = self.get_new_example(old_ex, method)
            if new_ex != old_ex:
                old_batch.append(old_ex)
                new_batch.append(new_ex)
        return old_batch, new_batch
 
    def get_batch(self, model, tokenizer, method="random", random_mask=None):
        stop = min(self.idxs_pos + self.batch_size, len(self.examples)
        idxs = self.idxs[self.idxs_pos:stop].copy()
        if stop == len(self.examples):
            random.shuffle(self.idxs)
            diff = self.idxs_pos + self.batch_size - len(self.examples)
            idxs += self.idxs[:diff]
            self.idxs_pos = diff
        else:
            self.idxs_pos += self.batch_size
        method = method if method ense "random"
        old_batch, new_batch = self.get_new_examples(tokenizer, idxs=idxs, method=method, random_mask=random_mask)
        return old_batch, new_batch 
