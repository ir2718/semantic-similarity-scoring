from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoConfig, DataCollatorWithPadding, TrainingArguments, Trainer
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import PopulationBasedTraining
from utils import *
from parsing import parse_fine_tune

import os
import numpy as np

args = parse_fine_tune()

class CFG:
    MODEL_NAME = args.model_name
    PRETRAINED_PATH = args.pretrained_path
    
    EPOCHS = args.num_epochs
    TRAIN_BATCH_SIZE = args.train_batch_size
    VAL_BATCH_SIZE = args.val_batch_size
    WEIGHT_DECAY = args.weight_decay
    LEARNING_RATE_START = args.learning_rate_start
    WARMUP_RATIO = args.warmup_ratio
    SCHEDULER = args.scheduler
    
    MAX_LEN = args.max_len

    SEED = args.seed

set_seed_(CFG.SEED)
device = set_device()

dataset = load_dataset_from_huggingface(DATASET_PATH, CONFIG_NAME)
dataset = preprocess_dataset_for_finetuning(dataset)

tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME)#max_length=CFG.MAX_LEN

tokenizer_kwargs={'tokenizer': tokenizer}
tokenized_datasets = dataset.map(
    tokenize_function, 
    batched=True,
    fn_kwargs=tokenizer_kwargs
)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

config = AutoConfig.from_pretrained(CFG.MODEL_NAME)
model = AutoModelForSequenceClassification.from_config(config)
model = AutoModelForSequenceClassification.from_pretrained(CFG.MODEL_NAME, num_labels=1)

## TRAINING END TO END ##
training_args_fine_tune = TrainingArguments(
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='epoch',
    learning_rate=CFG.LEARNING_RATE_START,
    per_device_train_batch_size=CFG.TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=CFG.VAL_BATCH_SIZE,
    num_train_epochs=CFG.EPOCHS,
    output_dir=os.path.join('./finetune', CFG.MODEL_NAME),
    weight_decay=CFG.WEIGHT_DECAY,
    lr_scheduler_type=CFG.SCHEDULER,
    warmup_ratio=CFG.WARMUP_RATIO,
    fp16=True
)

unfreeze_encoder(model)
trainer_fine_tune = Trainer(
    model,
    training_args_fine_tune,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print('starting hyperparameter search . . .')
best_trial = trainer_fine_tune.hyperparameter_search(
    direction='maximize',
    backend='ray',
    search_alg=HyperOptSearch(metric='objective', mode='max'),
    scheduler=PopulationBasedTraining(metric='objective', mode='max')
)

# add training with best hyperparams

print('starting end to end finetuning . . .')
train_output = trainer_fine_tune.train()

trainer_fine_tune.predict(tokenized_datasets['test'])