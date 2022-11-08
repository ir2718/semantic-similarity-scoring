from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import PopulationBasedTraining
from utils import *
from parsing import parse_mlm

import os
import numpy as np

args = parse_mlm()

class CFG:
    MODEL_NAME = args.model_name
    
    EPOCHS = args.num_epochs
    TRAIN_BATCH_SIZE = args.train_batch_size
    VAL_BATCH_SIZE = args.val_batch_size
    WEIGHT_DECAY = args.weight_decay
    LEARNING_RATE_START = args.learning_rate_start
    
    MAX_LEN = args.max_len
    BLOCK_SIZE = args.block_size
    
    SEED = args.seed

set_seed_(CFG.SEED)
device = set_device()

dataset_mlm = load_dataset_from_huggingface(DATASET_PATH, CONFIG_NAME)
dataset_mlm = preprocess_dataset_for_mlm(dataset_mlm)

tokenizer = AutoTokenizer.from_pretrained( CFG.MODEL_NAME, ) #max_length=CFG.MAX_LEN )

tokenize_kwargs = {'tokenizer': tokenizer}
tokenized_datasets = dataset_mlm.map(
    tokenize_function, 
    batched=True, 
    remove_columns=['sentence'], 
    fn_kwargs=tokenize_kwargs
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

group_kwargs = {'block_size': CFG.BLOCK_SIZE}
tokenized_labeled_datasets = tokenized_datasets.map(
    group_texts, 
    batched=True, 
    fn_kwargs=group_kwargs
)
model = AutoModelForMaskedLM.from_pretrained(CFG.MODEL_NAME)

training_args = TrainingArguments(
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='epoch',
    learning_rate=CFG.LEARNING_RATE_START,
    num_train_epochs=CFG.EPOCHS,
    weight_decay=CFG.WEIGHT_DECAY,
    output_dir=os.path.join('./masked_lm', CFG.MODEL_NAME),
    fp16=True,
    seed=CFG.SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
)

print('starting masked language modeling training . . .')

trainer.train()

test_preds = trainer.predict(tokenized_datasets['test'])