import argparse

def str2bool(x):
    if x == 'True':
        return True
    elif x == 'False':
        return False

def _parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='../dataset/stsb')

    parser.add_argument('--model_name', type=str, default='bert-base-cased')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--val_batch_size', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--learning_rate_start', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adamw_torch')
    parser.add_argument('--hyperopt', type=str2bool, default=True)
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--scheduler', type=str, default='cosine')
    return parser

def parse_fine_tune_huggingface():
    parser = _parse()
    parser.add_argument('--loss_function', type=str, default='mse') # 'mse', 'cross_entropy'
    parser.add_argument('--opt_metric', type=str, default='eval_pearson_r') # eval_spearman_r
    parser.add_argument('--grad_checkpoint', type=str2bool, default=False)
    args = parser.parse_args()
    return args


def parse_sentence_transformer():
    parser=_parse()
    args = parser.parse_args()
    return args