from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoConfig, DataCollatorWithPadding, TrainingArguments, Trainer, TrainerCallback
from ray.tune.schedulers import PopulationBasedTraining
from ray import tune
from utils import *
from parsing import parse_fine_tune
from configs import FROZEN_CFG
import os

args = parse_fine_tune()
FROZEN_CFG.set_args(args)

set_seed_(FROZEN_CFG.SEED)
device = set_device()

dataset = load_dataset_from_huggingface(DATASET_PATH, CONFIG_NAME)

model_name = FROZEN_CFG.MODEL_NAME

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer_kwargs = {
    'tokenizer': tokenizer,
    'max_len': FROZEN_CFG.MAX_LEN
}
tokenized_datasets = dataset.map(
    tokenize_function, 
    batched=True, 
    fn_kwargs=tokenizer_kwargs,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def model_init():
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=1,
    )
    freeze_encoder(model)
    return model

training_args = TrainingArguments(
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='epoch',
    optim=FROZEN_CFG.OPTIM,
    learning_rate=FROZEN_CFG.LEARNING_RATE_START,
    per_device_train_batch_size=FROZEN_CFG.TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=FROZEN_CFG.VAL_BATCH_SIZE,
    num_train_epochs=FROZEN_CFG.EPOCHS,
    output_dir=os.path.join('../frozen', model_name),
    weight_decay=FROZEN_CFG.WEIGHT_DECAY,
    lr_scheduler_type=FROZEN_CFG.SCHEDULER,
    warmup_ratio=FROZEN_CFG.WARMUP_RATIO,
    fp16=True,
)


trainer = Trainer(
    model=None,
    args=training_args,
    model_init=model_init,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

hp_space = {
    'learning_rate': tune.choice([1e-3, 5e-4, 1e-4, 5e-5, 1e-5]),
    'per_device_train_batch_size': tune.choice([8, 16, 32]),
    'weight_decay': tune.choice([1e-2, 1e-3, 1e-4])
}

print('starting hyperparameter search . . .')
scheduler = PopulationBasedTraining(
    time_attr='training_iteration',
    mode='max',
    metric='objective',
)

best_run = trainer.hyperparameter_search(
    hp_space=lambda _: hp_space,
    direction='maximize',
    backend='ray',
    compute_objective=compute_objective,
    keep_checkpoints_num=1,
    scheduler=scheduler,
    verbose=0,
    reuse_actors=True
)

print('BEST TRIAL: ')
print(best_run)

print('starting finetuning with frozen encoder . . .')
for n, v in best_run.hyperparameters.items():
    setattr(trainer.args, n, v)

train_output = trainer.train()
