import os
import numpy as np
import torch
import pandas as pd
import random
import datasets
import json
import pandas as pd
from sentence_transformers import InputExample
from tqdm import tqdm
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader

from scipy.stats import pearsonr, spearmanr

DATASET_PATH = 'glue'
CONFIG_NAME = 'stsb'

def evaluate(evaluator, model, output_path):
    os.makedirs(output_path, exist_ok=True)
    evaluator(model, output_path=output_path)

def create_evaluator(examples, batch_size=32):
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        examples,
        batch_size=batch_size
    )
    return evaluator

def get_optimizer_from_string(str_):
    opt_dict = {'adamw_torch': torch.optim.AdamW}
    return opt_dict[str_]

def get_dataloaders(train, validation, test, train_batch_size, validation_batch_size):
    train_dataloader = DataLoader(train, batch_size=train_batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation, batch_size=validation_batch_size, shuffle=False)
    test_dataloader = DataLoader(test, batch_size=validation_batch_size, shuffle=False)
    return train_dataloader, validation_dataloader, test_dataloader

def _hf_to_st(hf):
    return [InputExample(texts=[i['sentence1'], i['sentence2']], label=i['label']) for i in tqdm(hf)]

def to_sentence_transformers_dataset_for_similarity(datasets):
    datasets = datasets.remove_columns('idx')

    transform = lambda x: {'label': x['label']/5} # scaling to [0, 1] ??
    train_st = _hf_to_st(datasets['train'].map(transform))
    validation_st = _hf_to_st(datasets['validation'].map(transform))
    test_st = _hf_to_st(datasets['test'].map(transform))

    return train_st, validation_st, test_st

def set_seed_(seed):
    #random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def _load_test_labels(dataset_path):
    f = open(f'{dataset_path}/sts-test.tsv', 'r', encoding='utf-8')

    l = []
    for x in f:
        curr_line = x.split('\t')
        l.append(np.float32(curr_line[4]))
    
    labels = np.array(l)
    df_labels = pd.DataFrame.from_dict({'label':labels})
    hf_df = datasets.Dataset.from_pandas(df_labels)
    return hf_df

def set_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

def predict_and_save_results(trainer, tokenized_datasets, output_dir, best_run=None):
    if best_run is not None:
        with open(os.path.join(output_dir, 'hyperparameters.txt'), 'w') as hyperparams:
            hyperparams.write(json.dumps(best_run.hyperparameters))

    train_res = trainer.predict(tokenized_datasets['train'], metric_key_prefix='train')
    validation_res = trainer.predict(tokenized_datasets['validation'], metric_key_prefix='validation')
    test_res = trainer.predict(tokenized_datasets['test'])

    train_preds_df = pd.DataFrame.from_dict({'preds': train_res.predictions.reshape(-1)})
    validation_preds_df = pd.DataFrame.from_dict({'preds': validation_res.predictions.reshape(-1)})
    test_preds_df = pd.DataFrame.from_dict({'preds': test_res.predictions.reshape(-1)})

    train_preds_df.to_csv(os.path.join(output_dir, 'train_res.csv'), index=False)
    validation_preds_df.to_csv(os.path.join(output_dir, 'validation_res.csv'), index=False)
    test_preds_df.to_csv(os.path.join(output_dir, 'test_res.csv'), index=False)

    metrics_d = {**train_res.metrics, **validation_res.metrics, **test_res.metrics}
    splits = ['train', 'validation', 'test']
    metrics = ['loss', 'spearman_r', 'pearson_r']
    strs = [[f'{s}_{m}' for m in metrics] for s in splits]
    flatten_strs = [i for subl in strs for i in subl]
    metrics_d = {k:v for k, v in metrics_d.items() if k in flatten_strs}
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as metric_info:
        metric_info.write(json.dumps(metrics_d))

def freeze_encoder(model):
    for name, param in model.base_model.named_parameters():
        if name.startswith(('transformer', 'embeddings', 'encoder')):
            param.requires_grad = False

def unfreeze_encoder(model):
    for p in model.base_model.parameters():
        p.requires_grad = True

def check_grad_params(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

def compute_metrics_(y_true, y_pred):
    return {
        'pearson_r': pearsonr(y_pred, y_true)[0],
        'spearman_r': spearmanr(y_pred, y_true)[0]
    }

def compute_metrics(eval_preds):
    output = eval_preds.predictions.reshape(-1)
    labels = eval_preds.label_ids
    return compute_metrics_(labels, output)

def compute_objective(eval_dict):
    return eval_dict['pearson_r']

def scoring_function_pearson(y_true, y_pred):
    return pearsonr(y_pred, y_true)[0]

def load_stsb_dataset_from_disk(dataset_path):
    cols = ['dataset_type', 'dataset', 'split', 'id', 'label', 'sentence1', 'sentence2']

    train = pd.read_csv(f'{dataset_path}/sts-train.tsv', sep='\t', header=None, engine='python', on_bad_lines='skip', names=cols)
    validation = pd.read_csv(f'{dataset_path}/sts-dev.tsv', sep='\t', header=None, engine='python', on_bad_lines='skip', names=cols)
    test = pd.read_csv(f'{dataset_path}/sts-test.tsv', sep='\t', header=None, engine='python',  on_bad_lines='skip', names=cols)

    train = datasets.Dataset.from_pandas(train)
    validation = datasets.Dataset.from_pandas(validation)
    test = datasets.Dataset.from_pandas(test)

    dataset_dict = datasets.DatasetDict({
        'train': train,
        'validation':validation,
        'test':test,
    })
    return dataset_dict

def load_dataset_from_huggingface(dataset_path, config_name, label_dir):
    dataset = datasets.load_dataset(dataset_path, config_name)

    test_labels = _load_test_labels(label_dir)
    dataset['test'] = dataset['test'].remove_columns('label')
    dataset['test'] = dataset['test'].add_column(name='label', column=test_labels['label'])
    return dataset

def load_stratified_dataset(dataset_dir):
    train_stratified = datasets.Dataset.from_pandas(pd.read_csv(os.path.join(dataset_dir, 'train_stratified.csv')))
    validation_stratified = datasets.Dataset.from_pandas(pd.read_csv(os.path.join(dataset_dir, 'validation_stratified.csv')))
    test_stratified = datasets.Dataset.from_pandas(pd.read_csv(os.path.join(dataset_dir, 'test_stratified.csv')))
    dataset_dict = datasets.DatasetDict({
        'train':train_stratified,
        'validation':validation_stratified,
        'test':test_stratified
    })
    return dataset_dict

def tokenize_function(examples, **fn_kwargs):
    result = fn_kwargs['tokenizer'](
        examples['sentence1'], 
        examples['sentence2'], 
        truncation=True
    )
    return result

def get_best_model_and_config_path(model_name, all_models_dir):
    model_dir = os.path.join(all_models_dir, model_name.replace('/', '-'))
    all_dirs = [f for f in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, f)) and f.startswith('checkpoint-')]
    checkpoint_dir = {os.path.join(model_dir, d):int(d.split('-')[-1])  for d in all_dirs}
    final_dir = sorted(checkpoint_dir, key=checkpoint_dir.get, reverse=True)[0]
    trainer_state = json.load(open(os.path.join(final_dir, 'trainer_state.json')))
    best_model_checkpoint = os.path.abspath(trainer_state['best_model_checkpoint'])
    return best_model_checkpoint, os.path.join(best_model_checkpoint, 'config.json')