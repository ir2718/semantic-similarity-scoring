from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoConfig, DataCollatorWithPadding, TrainingArguments, Trainer
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import PopulationBasedTraining
from utils import *
from parsing import parse_fine_tune
from configs import END2END_CFG
import os

args = parse_fine_tune()
END2END_CFG.set_args(args)

set_seed_(END2END_CFG.SEED)
device = set_device()

dataset = load_dataset_from_huggingface(DATASET_PATH, CONFIG_NAME)
dataset = preprocess_dataset_for_finetuning(dataset)

tokenizer = AutoTokenizer.from_pretrained(END2END_CFG.MODEL_NAME, max_length=END2END_CFG.MAX_LEN)

tokenizer_kwargs={'tokenizer': tokenizer}
tokenized_datasets = dataset.map(
    tokenize_function, 
    batched=True,
    fn_kwargs=tokenizer_kwargs
)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

config = AutoConfig.from_pretrained(END2END_CFG.MODEL_NAME)
model = AutoModelForSequenceClassification.from_config(config)
model = AutoModelForSequenceClassification.from_pretrained(END2END_CFG.MODEL_NAME, num_labels=1)

training_args_fine_tune = TrainingArguments(
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='epoch',
    learning_rate=END2END_CFG.LEARNING_RATE_START,
    per_device_train_batch_size=END2END_CFG.TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=END2END_CFG.VAL_BATCH_SIZE,
    num_train_epochs=END2END_CFG.EPOCHS,
    output_dir=os.path.join('finetune', END2END_CFG.MODEL_NAME),
    weight_decay=END2END_CFG.WEIGHT_DECAY,
    lr_scheduler_type=END2END_CFG.SCHEDULER,
    warmup_ratio=END2END_CFG.WARMUP_RATIO,
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