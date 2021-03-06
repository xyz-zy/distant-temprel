CLASSES = ['AFTER', 'BEFORE', 'EQUAL', 'VAGUE']
VAGUE = CLASSES.index('VAGUE')

# Model Hyper Parameters
TRAIN_BATCH_SIZE = 10
EVAL_BATCH_SIZE = 32
PREDICT_BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_TRAIN_EPOCHS = 5
WARMUP_PROPORTION = 0.1
GRADIENT_ACCUMULATION_STEPS = 1
MAX_SEQ_LENGTH = 200
DOC_STRIDE = 128

