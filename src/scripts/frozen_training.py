from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoConfig, DataCollatorWithPadding, TrainingArguments, Trainer, TrainerCallback
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.schedulers import PopulationBasedTraining
from utils import *
from parsing import parse_fine_tune
from configs import FROZEN_CFG
import os

args = parse_fine_tune()
FROZEN_CFG.set_args(args)

set_seed_(FROZEN_CFG.SEED)
device = set_device()

dataset = load_dataset_from_huggingface(DATASET_PATH, CONFIG_NAME)
dataset = preprocess_dataset_for_finetuning(dataset)

tokenizer = AutoTokenizer.from_pretrained(
    FROZEN_CFG.MODEL_NAME
)

tokenizer_kwargs = {'tokenizer': tokenizer}
tokenized_datasets = dataset.map(
    tokenize_function, 
    batched=True, 
    fn_kwargs=tokenizer_kwargs
)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

model = AutoModelForSequenceClassification.from_pretrained(FROZEN_CFG.MODEL_NAME, num_labels=1)

training_args = TrainingArguments(
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='epoch',
    learning_rate=FROZEN_CFG.LEARNING_RATE_START,
    per_device_train_batch_size=FROZEN_CFG.TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=FROZEN_CFG.VAL_BATCH_SIZE,
    num_train_epochs=FROZEN_CFG.EPOCHS,
    output_dir=os.path.join('../frozen', FROZEN_CFG.MODEL_NAME),
    weight_decay=FROZEN_CFG.WEIGHT_DECAY,
    lr_scheduler_type=FROZEN_CFG.SCHEDULER,
    warmup_ratio=FROZEN_CFG.WARMUP_RATIO,
    fp16=True
)

freeze_encoder(model)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

print('starting hyperparameter search . . .')
best_trial = trainer.hyperparameter_search(
    direction='maximize',
    backend='ray',
    search_alg=HyperOptSearch(metric='objective', mode='max'),
    scheduler=PopulationBasedTraining(metric='objective', mode='max')
)

# add training with best hyperparams

print('starting finetuning with frozen encoder . . .')
train_output = trainer.train()

trainer.predict(tokenized_datasets['test'])