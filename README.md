# xdomain-temprel

Code for training and evaluating a TempRel classifier on MATRES, UDS-T, and distantly labeled examples.

Before running the model, make sure that the desired data files have been downloaded as per:

* `timebank/README.md`: MATRES Dataset
* `udst/README.md`: UDS-T Dataset
* `beforeafter/README.md`: Distantly-labeled BeforeAfter Examples
* TODO: Distantly-labeled Timex-anchored Examples

## Training & Evaluating

To train and evaluate on MATRES:

`python train.py --lm roberta --data matres --output_dir /PATH/TO/MODEL_CHKPTS/ --epoch 5 --batch 32`

`python eval.py --lm roberta --data matres_dev matres_test --model_dir /PATH/TO/MODEL_CHKPTS/`

