from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import PopulationBasedTraining
from utils import *
from parsing import parse_mlm
from configs import MLM_CFG
import os

args = parse_mlm()
MLM_CFG.set_args(args)

set_seed_(MLM_CFG.SEED)
device = set_device()

dataset_mlm = load_dataset_from_huggingface(DATASET_PATH, CONFIG_NAME)
dataset_mlm = preprocess_dataset_for_mlm(dataset_mlm)

tokenizer = AutoTokenizer.from_pretrained(
    MLM_CFG.MODEL_NAME
)

tokenize_kwargs = {'tokenizer': tokenizer}
tokenized_datasets = dataset_mlm.map(
    tokenize_function, 
    batched=True, 
    remove_columns=['sentence'], 
    fn_kwargs=tokenize_kwargs
)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer)

group_kwargs = {'block_size': MLM_CFG.BLOCK_SIZE}
tokenized_labeled_datasets = tokenized_datasets.map(
    group_texts, 
    batched=True, 
    fn_kwargs=group_kwargs
)
model = AutoModelForMaskedLM.from_pretrained(MLM_CFG.MODEL_NAME, truncation=True)

training_args = TrainingArguments(
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='epoch',
    learning_rate=MLM_CFG.LEARNING_RATE_START,
    num_train_epochs=MLM_CFG.EPOCHS,
    weight_decay=MLM_CFG.WEIGHT_DECAY,
    output_dir=os.path.join('../mlm', MLM_CFG.MODEL_NAME),
    fp16=True,
    seed=MLM_CFG.SEED,
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