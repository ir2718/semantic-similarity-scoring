# semantic-similarity-scoring

# WORK IN PROGRESS #

## SOTA models for this task on STSB:
- [Turing ULR v6](https://arxiv.org/abs/2210.14867)
- Vega v1
- [Turing NLR v5](https://arxiv.org/abs/2204.06644)
- DeBERTa + CLEVER
- [ERNIE](https://github.com/PaddlePaddle/ERNIE)
- [StructBERT + CLEVER](https://github.com/alibaba/AliceMind)
- [DeBERTa / TuringNLRv4](https://github.com/microsoft/DeBERTa)
- MaxALBERT + DKM
- ALVERT + DAAF + NAS
- [T5](https://github.com/google-research/text-to-text-transfer-transformer)

## Project notes:

### Baselines
- supervised: linear regression and SVM with averaged word2vec representations of words
- unsupervised: cosine similarity between averaged word2vec representations of words
- no hyperparameter optimization

| **Model used**      | **Train set**     | **Validation  set** | **Test set**     |
| ------------------- | ----------------- | ------------------- | ---------------- |
| Cosine similarity   | 0.459/0.462       | 0.478/0.540         | 0.367/0.388      |
| Linear regression   | 0.440/0.425       | 0.119/0.118         | 0.194/0.193      |
| SVM                 | 0.585/0.576       | 0.258/0.240         | 0.330/0.301      | 

### Hyperparameter optimization
- done using [Population-based training](https://arxiv.org/pdf/1711.09846.pdf) or hand tuning
- when using PBT with frozen encoder the same values for the learning rate and weight decay are used: 
```
  learning_rate: [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
  weight_decay: [1e-2, 1e-3, 1e-4]
```

- when using PBT with end to end tuning the values for the learning rate and weight decay used are:
```
learning_rate: [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
weight_decay: [1e-2, 1e-3, 1e-4]
```

- the batch size grid used always contains exactly 3 values and varies depending on the size of the model - so far only 8, 16 and 32

### Masked language modeling
- might be useful, but will be left out for now because of computation cost

### Finetuning with the frozen base model
- no gradient updates on all parameters of base model
- finetuning for 10 epochs and using early stopping if no improvement is seen in the last 3 epochs

| **Model**           | **Train set**     | **Validation  set** | **Test set**     | **Batch size** | **Learning rate** | **Weight decay** | **Batch size grid** |
| ------------------- | ----------------- | ------------------- | ---------------- |--------------- | ----------------- | ---------------- | ------------------- |
| BERT base cased     | 0.793/0.749       | 0.814/0.809         | 0.735/0.697      | 32             | 5e-4              | 1e-4             | [8, 16, 32]         |
| BERT large cased    | 0.789/0.749       | 0.824/0.821         | 0.729/0.691      | 8              | 5e-4              | 1e-2             | [8, 16, 32]         |
| RoBERTa base        | 0.631/0.629       | 0.585/0.591         | 0.569/0.578      | 32             | 5e-4              | 1e-4             | [8, 16, 32]         |
| RoBERTa large       | 0.512/0.506       | 0.492/0.486         | 0.493/0.502      | 32             | 5e-4              | 1e-4             | [8, 16, 32]         |
| DistilRoBERTa base  | 0.576/0.581       | 0.485/0.477         | 0.510/0.516      | 32             | 5e-4              | 1e-4             | [8, 16, 32]         |
| DeBERTaV3 small     | 0.782/0.764       | 0.761/0.763         | 0.758/0.758      | 32             | 5e-4              | 1e-4             | [8, 16, 32]         |
| DeBERTaV3 base      | 0.838/0.832       | 0.809/0.823         | 0.824/0.831      | 32             | 5e-4              | 1e-4             | [8, 16, 32]         |
| DeBERTaV3 large     | 0.828/0.822       | 0.807/0.816         | 0.820/0.825      | 8              | 5e-4              | 1e-2             | [8, 16, 32]         |

### End2End finetuning
- finetuning for 10 epochs or using early stopping


| **Model**           | **Train set**     | **Validation  set** | **Test set**     | **Batch size** | **Learning rate** | **Weight decay** | **Batch size grid** |
| ------------------- | ----------------- | ------------------- | ---------------- |--------------- | ----------------- | ---------------- | ------------------- |
| BERT base cased     | 0.995/0.995       | 0.899/0.896         | 0.865/0.856      | 32             |  5e-5             | 1e-4             | [8, 16, 32]         |
| BERT large cased    |                   |                     |                  |                |                   |                  |                     |
| RoBERTa base        |                   |                     |                  |                |                   |                  |                     |
| RoBERTa large       |                   |                     |                  |                |                   |                  |                     |
| DistilRoBERTa base  | 0.988/0.986       | 0.887/0.885         | 0.858/0.849      | 32             | 5e-5              | 1e-4             | [8, 16, 32]         |
| DeBERTaV3 small     | 0.991/0.990       | 0.906/0.904         | 0.892/0.888      | 8              | 5e-5              | 1e-2             | [8, 16, 32]         |
| DeBERTaV3 base      |                   |                     |                  |                |                   |                  |                     |
| DeBERTaV3 large     |                   |                     |                  |                |                   |                  |                     |

### Ensembling the finetuned models
- using a voting classifier or gradient boosting algorithms
---------------------------------------
- cosine annealing for the learning rates with warmup steps
- FP16 training throughout all the steps to reduce the model training time
- regularization using weight decay
- using pearsons' and spearmans' coefficient for evaluation

### Usage
- training is split into two main scripts ```frozen_training.py``` and ```end2end_training.py``` and they should be used in that order
- all scripts are meant to be called from the ```src/scripts``` folder and all of them have default arguments
```
cd src/scripts
python frozen_training.py --model_name bert-base-cased
python end2end_training.py --model_name bert-base-cased
```
