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


def predict_and_save_results(trainer, tokenized_datasets, output_dir, best_run):
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
    for p in model.base_model.encoder.parameters():
        p.requires_grad = False

def unfreeze_encoder(model):
    for p in model.base_model.encoder.parameters():
        p.requires_grad = True

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


def _get_all_sentences_list(dataset, split):
    return dataset[split]['sentence1'][:] + dataset[split]['sentence2'][:]

def load_dataset_from_huggingface(dataset_path, config_name, label_dir):
    dataset = datasets.load_dataset(dataset_path, config_name)

    test_labels = _load_test_labels(label_dir)
    dataset['test'] = dataset['test'].remove_columns('label')
    dataset['test'] = dataset['test'].add_column(name='label', column=test_labels['label'])
    return dataset

def preprocess_dataset_for_mlm(dataset):
    all_sentences_train = _get_all_sentences_list(dataset, 'train')
    all_sentences_validation = _get_all_sentences_list(dataset, 'validation')
    all_sentences_test = _get_all_sentences_list(dataset, 'test')

    dataset_mlm = datasets.DatasetDict({
        'train': datasets.Dataset.from_dict({'sentence': all_sentences_train}),
        'validation': datasets.Dataset.from_dict({'sentence': all_sentences_validation}),
        'test': datasets.Dataset.from_dict({'sentence': all_sentences_test})
    })
    return dataset_mlm

def preprocess_dataset_for_finetuning(datasets):
    return datasets

def tokenize_function(examples, **fn_kwargs):
    result = fn_kwargs['tokenizer'](
        examples['sentence1'], 
        examples['sentence2'], 
        truncation=True
    )
    return result

def group_texts(examples, **fn_kwargs):
    block_size = fn_kwargs['block_size']
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return results