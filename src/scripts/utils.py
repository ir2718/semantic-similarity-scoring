import os
import numpy as np
import torch
import pandas as pd
import datasets
import json
import pandas as pd

from scipy.stats import pearsonr, spearmanr

DATASET_PATH = 'glue'
CONFIG_NAME = 'stsb'

def set_seed_(seed):
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
    for p in model.base_model.parameters():
        p.requires_grad = False

def unfreeze_encoder(model):
    for p in model.base_model.parameters():
        p.requires_grad = True

def check_grad_params(model):
    for p in model.named_parameters():
        if p[1].requires_grad:
            print(p[0])

def compute_metrics(eval_preds):
    output = eval_preds.predictions
    labels = eval_preds.label_ids
    return {
        'pearson_r': pearsonr(output.reshape(-1), labels)[0],
        'spearman_r': spearmanr(output.reshape(-1), labels)[0]
    }

def compute_objective(eval_dict):
    return eval_dict['pearson_r']

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

def tokenize_function(examples, **fn_kwargs):
    result = fn_kwargs['tokenizer'](
        examples['sentence1'], 
        examples['sentence2'], 
        truncation=True
    )
    return result

def get_best_model_and_config_path(model_name):
    model_dir = os.path.join('..\\frozen', model_name.replace('/', '-'))
    all_dirs = [f for f in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, f)) and f.startswith('checkpoint-')]
    checkpoint_dir = {os.path.join(model_dir, d):int(d.split('-')[-1])  for d in all_dirs}
    final_dir = sorted(checkpoint_dir, key=checkpoint_dir.get, reverse=True)[0]
    trainer_state = json.load(open(os.path.join(final_dir, 'trainer_state.json')))
    best_model_checkpoint = os.path.abspath(trainer_state['best_model_checkpoint'])
    return best_model_checkpoint, os.path.join(best_model_checkpoint, 'config.json')