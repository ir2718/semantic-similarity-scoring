import os
import numpy as np
import torch

import datasets

from scipy.stats import pearsonr, spearmanr

TRAIN_PATH = 'dataset\STS-B\original\sts-train.tsv'
TEST_PATH = 'dataset\STS-B\original\sts-test.tsv'
VALIDATION_PATH = 'dataset\STS-B\original\sts-dev.tsv'

DATASET_PATH = 'glue'
CONFIG_NAME = 'stsb'

def set_seed_(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def set_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'

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
    return eval_dict['eval_spearman_r']

def load_dataset_from_disk(train_path, test_path, validation_path, format_='csv'):
    cwd = os.getcwd()
    dataset = datasets.load_dataset(
        format_, 
        data_files={
            'train': os.path.join(cwd, train_path),
            'validation': os.path.join(cwd, validation_path),
            'test': os.path.join(cwd, test_path)
        },
        encoding='utf-8'
    )
    return dataset


def _get_all_sentences_list(dataset, split):
    return dataset[split]['sentence1'][:] + dataset[split]['sentence2'][:]

def load_dataset_from_huggingface(dataset_path, config_name):
    dataset = datasets.load_dataset(dataset_path, config_name)
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
        max_length=fn_kwargs['max_len'],
    )
    #if tokenizer.is_fast:
    #    result['word_ids'] = [result.word_ids(i) for i in range(len(result['input_ids']))]
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
    return result