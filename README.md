# xdomain-temprel

Code for training and evaluating a TempRel classifier on MATRES, UDS-T, and distantly labeled examples.

Before running the model, make sure that the desired data files have been downloaded as per:

* `timebank/README.md`: MATRES Dataset
* `udst/README.md`: UDS-T Dataset
* `beforeafter/README.md`: Distantly-labeled BeforeAfter Examples
* `timex/README.md`: Distantly-labeled Timex-anchored Examples

## Training & Evaluating

To train and evaluate on MATRES:

`python train.py --lm roberta --data matres --output_dir /PATH/TO/MODEL_CHKPTS/ --epoch 5 --batch 32`

`python eval.py --lm roberta --data matres_dev matres_test --model_dir /PATH/TO/MODEL_CHKPTS/`


To train and evaluate on UDS-T:

`python train.py --lm roberta --data udst --output_dir /PATH/TO/MODEL_CHKPTS/ --epoch 5 --batch 32`

`python eval.py --lm roberta --data udst_dev_maj_conf_nt udst_test_maj_conf_nt --model_dir /PATH/TO/MODEL_CHKPTS/`

To train and evaluate on timex train/test split from [Goyal and Durrett, 2019](https://arxiv.org/abs/1906.08287):

`python train.py --lm roberta --data distant --output_dir /PATH/TO/MODEL_CHKPTS/ --epoch 5 --batch 32`

`python eval.py --lm roberta --data distant_test --model_dir /PATH/TO/MODEL_CHKPTS/`

To train and evaluate on additional timex data, sampled more evenly from sources in the English Gigaword Fifth Edition, specify a dataset from `timex/data/*pkl`, e.g.:

`python train.py --lm roberta --data timex/data/d1k.pkl --output_dir /PATH/TO/MODEL_CHKPTS/ --epoch 5 --batch 32`

For multiple data sources, simply specify as a space-separated list, e.g.:

`python train.py --lm roberta --data matres timex/data/d1k.pkl --output_dir /PATH/TO/MODEL_CHKPTS/ --epoch 5 --batch 32`

## Fine-Tuned Models

You can download RoBERTa models that have been fine tuned:

 * on MATRES: [link](https://drive.google.com/file/d/17tLQWCJ3Zwz_YKwkYp_bCzO9yj48uv0r/view?usp=sharing)
 * on 1k MATRES examples and 10k DistantTimex examples: [link](https://drive.google.com/file/d/1YvXgCrrfwvfk0PB9CzPEtzzHglt2TO_z/view?usp=sharing)
