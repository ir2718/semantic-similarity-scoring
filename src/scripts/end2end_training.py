from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer, EarlyStoppingCallback, AutoConfig
from ray.tune.schedulers import PopulationBasedTraining
from ray import tune
from trainers import get_trainer_with_loss
from utils import *
from parsing import parse_fine_tune_huggingface
from configs import END2END_CFG
import os

args = parse_fine_tune_huggingface()
END2END_CFG.set_args(args)
device = set_device()

if args.stratified:
    dataset = load_stratified_dataset(args.dataset_path)
else:
    dataset = load_dataset_from_huggingface(DATASET_PATH, CONFIG_NAME, args.dataset_path)

model_name = END2END_CFG.MODEL_NAME
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer_kwargs={
    'tokenizer': tokenizer
}
tokenized_datasets = dataset.map(
    tokenize_function, 
    batched=True,
    fn_kwargs=tokenizer_kwargs,
)

best_model_path, config_path = get_best_model_and_config_path(model_name, f'..\\models\\{END2END_CFG.LOSS_FUNCTION}\\frozen')

print(f'\n\nBest model path found: {best_model_path}\n\n')

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

def model_init():
    model = AutoModelForSequenceClassification.from_pretrained(
        best_model_path, 
        local_files_only=True
    )
    unfreeze_encoder(model)
    return model

output_dir = os.path.join(f'..\\models\\{END2END_CFG.LOSS_FUNCTION}\\end2end', model_name.replace('/', '-'))
training_args = TrainingArguments(
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_strategy='epoch',
    optim=END2END_CFG.OPTIM,
    learning_rate=END2END_CFG.LEARNING_RATE_START,
    per_device_train_batch_size=END2END_CFG.TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=END2END_CFG.VAL_BATCH_SIZE,
    num_train_epochs=END2END_CFG.EPOCHS,
    output_dir=output_dir,
    weight_decay=END2END_CFG.WEIGHT_DECAY,
    lr_scheduler_type=END2END_CFG.SCHEDULER,
    warmup_ratio=END2END_CFG.WARMUP_RATIO,
    fp16=False,
    load_best_model_at_end=True,
    metric_for_best_model=END2END_CFG.OPT_METRIC,
    gradient_checkpointing=END2END_CFG.GRAD_CHECKPOINT,
)


early_stopping = EarlyStoppingCallback(early_stopping_patience=END2END_CFG.PATIENCE)
trainer_type = get_trainer_with_loss(END2END_CFG.LOSS_FUNCTION)
tokenized_datasets = tokenized_datasets.map(trainer_type.scale_labels)

trainer = trainer_type(
    model=None,
    args=training_args,
    model_init=model_init,
    #tokenizer=tokenizer,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    callbacks=[early_stopping],
)

if args.hyperopt:
    hp_space = {
        'learning_rate': tune.choice([1e-4, 5e-5, 1e-5, 5e-6, 1e-6]),
        'per_device_train_batch_size': tune.choice([8, 16, 32]),
        'weight_decay': tune.choice([1e-2, 1e-3, 1e-4])
    }
    
    scheduler = PopulationBasedTraining(
        time_attr='training_iteration',
        mode='max',
        metric='objective',
    )

    opt_metric = END2END_CFG.OPT_METRIC
    best_run = trainer.hyperparameter_search(
        hp_space=lambda _: hp_space,
        direction='maximize',
        backend='ray',
        compute_objective=lambda m: m[opt_metric],
        scheduler=scheduler,
        keep_checkpoints_num=1,
        verbose=0,
        reuse_actors=True,
        n_trials=10,
        resources_per_trial={'cpu':1, 'gpu': 1},
        trial_dirname_creator=lambda trial: str(trial)
    )
    for n, v in best_run.hyperparameters.items():
        setattr(trainer.args, n, v)
else:
    best_run = None

trainer.train()
predict_and_save_results(trainer, tokenized_datasets, output_dir, best_run)