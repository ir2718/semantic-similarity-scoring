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
- when using PBT with the frozen encoder the values used for the hyperparameters used are: 
```
  learning_rate: [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
  weight_decay: [1e-2, 1e-3, 1e-4]
  batch_size: [8, 16, 32]
```

- when using PBT with end to end tuning:
```
  learning_rate: [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]
  weight_decay: [1e-2, 1e-3, 1e-4]
  batch_size: [8, 16, 32]
```

### Finetuning with the frozen base model
- no gradient updates on all parameters of base model, except for pooler
- finetuning for 10 epochs and using early stopping if no improvement is seen in the last 3 epochs

### End-to-end finetuning
- finetuning for 10 epochs and using early stopping if no improvement is seen in the last 3 epochs

### Mean squared error loss function
- finetuning the models using MSE as the loss function

| **Model**           | **Train set**     | **Validation  set** | **Test set**     | **Batch size** | **Learning rate** | **Weight decay** |
| ------------------- | ----------------- | ------------------- | ---------------- |--------------- | ----------------- | ---------------- |
| BERT base cased     | 0.793/0.749       | 0.814/0.809         | 0.735/0.697      | 32             | 5e-4              | 1e-4             |
| BERT large cased    | 0.790/0.754       | 0.824/0.823         | 0.731/0.694      | 8              | 5e-4              | 1e-2             |
| RoBERTa base        | 0.631/0.629       | 0.585/0.591         | 0.569/0.578      | 32             | 5e-4              | 1e-4             |
| RoBERTa large       | 0.512/0.506       | 0.492/0.486         | 0.493/0.502      | 32             | 5e-4              | 1e-4             |
| DistilRoBERTa base  | 0.576/0.581       | 0.485/0.477         | 0.510/0.516      | 32             | 5e-4              | 1e-4             |
| DistilBERT base cased | 0.639/0.604     | 0.604/0.605         | 0.579/0.555      | 32             | 5e-4              | 1e-4             |
| DeBERTaV3 small     | 0.782/0.764       | 0.761/0.763         | 0.758/0.758      | 32             | 5e-4              | 1e-4             |
| DeBERTaV3 base      | 0.838/0.832       | 0.809/0.823         | 0.824/0.831      | 32             | 5e-4              | 1e-4             |
| DeBERTaV3 large     | 0.828/0.822       | 0.807/0.816         | 0.820/0.825      | 8              | 5e-4              | 1e-2             |

| **Model**           | **Train set**     | **Validation  set** | **Test set**     | **Batch size** | **Learning rate** | **Weight decay** |
| ------------------- | ----------------- | ------------------- | ---------------- |--------------- | ----------------- | ---------------- |
| BERT base cased     | 0.995/0.995       | 0.899/0.896         | 0.865/0.856      | 32             | 5e-5              | 1e-4             |
| BERT large cased    | 0.993/0.992       | 0.908/0.904         | 0.869/0.857      | 16             | 1e-5              | 1e-3             |
| RoBERTa base        | 0.989/0.988       | 0.913/0.911         | 0.895/0.890      | 32             | 5e-5              | 1e-4             |
| RoBERTa large       | 0.994/0.994       | 0.921/0.920         | 0.904/0.899      | 32             | 5e-5              | 1e-4             |
| DistilRoBERTa base  | 0.988/0.986       | 0.887/0.885         | 0.858/0.849      | 32             | 5e-5              | 1e-4             |
| DistilBERT base cased | 0.994/0.993     | 0.863/0.861         | 0.814/0.800      | 32             | 5e-5              | 1e-4             |
| DeBERTaV3 small     | 0.991/0.990       | 0.906/0.904         | 0.892/0.888      | 8              | 5e-5              | 1e-2             |
| DeBERTaV3 base      | 0.996/0.996       | 0.917/0.915         | 0.907/0.904      | 8              | 5e-5              | 1e-3             |
| DeBERTaV3 large     | 0.991/0.990       | 0.927/0.926         | 0.922/0.921      | 8              | 1e-5              | 1e-4             |

- due to computation costs, hand tuning was used for end-to-end tuning of larger architectures such as:
  - RoBERTa large
  - BERT large cased
  - DeBERTaV3 base, large

### Cross entropy loss function
- finetuning the models using cross entropy as the loss function
- labels are scaled to $[0, 1]$ for this approach

| **Model**           | **Train set**     | **Validation  set** | **Test set**     | **Batch size** | **Learning rate** | **Weight decay** |
| ------------------- | ----------------- | ------------------- | ---------------- |--------------- | ----------------- | ---------------- |
| BERT base cased     | 0.779/0.723       | 0.813/0.802         | 0.726/0.686      | 8              | 5e-4              | 1e-2             |
| BERT large cased    | 0.791/0.751       | 0.822/0.821         | 0.736/0.699      | 32             | 5e-4              | 1e-2             |
| RoBERTa base        | 0.621/0.612       | 0.575/0.577         | 0.567/0.572      | 32             | 5e-4              | 1e-4             |
| RoBERTa large       | 0.513/0.520       | 0.484/0.478         | 0.494/0.513      | 32             | 5e-4              | 1e-4             |
| DistilRoBERTa base  | 0.590/0.586       | 0.488/0.475         | 0.515/0.510      | 32             | 5e-4              | 1e-4             |
| DistilBERT base cased | 0.656/0.614     | 0.609/0.614         | 0.589/0.561      | 8              | 5e-4              | 1e-2             |
| DeBERTaV3 small     | 0.793/0.767       | 0.769/0.767         | 0.770/0.762      | 32             | 5e-4              | 1e-4             |
| DeBERTaV3 base      | 0.843/0.830       | 0.813/0.819         | 0.828/0.827      | 32             | 5e-4              | 1e-4             |
| DeBERTaV3 large     | 0.835/0.821       | 0.814/0.817         | 0.826/0.824      | 32             | 5e-4              | 1e-4             |

| **Model**           | **Train set**     | **Validation  set** | **Test set**     | **Batch size** | **Learning rate** | **Weight decay** |
| ------------------- | ----------------- | ------------------- | ---------------- |--------------- | ----------------- | ---------------- |
| BERT base cased     | 0.997/0.996       | 0.899/0.896         | 0.861/0.849      | 8              | 5e-5              | 1e-2             |
| BERT large cased    | 0.978/0.976       | 0.909/0.906         | 0.875/0.865      | 8              | 1e-5              | 1e-4             |
| RoBERTa base        | 0.992/0.991       | 0.908/0.906         | 0.886/0.881      | 8              | 5e-5              | 1e-2             |
| RoBERTa large       | 0.989/0.988       | 0.924/0.923         | 0.913/0.909      | 8              | 1e-5              | 1e-2             |
| DistilRoBERTa base  | 0.990/0.989       | 0.890/0.888         | 0.859/0.850      | 8              | 5e-5              | 1e-2             |
| DistilBERT base cased | 0.991/0.990     | 0.860/0.856         | 0.814/0.801      | 8              | 5e-5              | 1e-2             |
| DeBERTaV3 small     | 0.990/0.988       | 0.907/0.904         | 0.893/0.890      | 28             | 5e-5              | 1e-4             |
| DeBERTaV3 base      | 0.988/0.987       | 0.919/0.917         | 0.912/0.911      | 32             | 5e-5              | 1e-4             |
| DeBERTaV3 large     | 0.986/0.984       | 0.927/0.926         | 0.919/0.919      | 8              | 1e-5              | 1e-3             |

- due to computation costs, hand tuning was used for end-to-end tuning of larger architectures such as:
  - RoBERTa large
  - BERT large cased
  - DeBERTaV3 base, large

### Cosine similarity loss function
- finetuning the models using sentence transformers and cosine similarity as the loss function
- labels are scaled to $[-1, 1]$ for this approach
- includes only end to end tuning

| **Model**           | **Train set**     | **Validation  set** | **Test set**     | **Batch size** | **Learning rate** | **Weight decay** |
| ------------------- | ----------------- | ------------------- | ---------------- |--------------- | ----------------- | ---------------- |
| T5 base             |                   |                     |                  |                |                   |                  |
| T5 large            |                   |                     |                  |                |                   |                  |
| MiniLM L6 v2        |                   |                     |                  |                |                   |                  |
| MiniLM L12 v2       |                   |                     |                  |                |                   |                  |
| MPNet base v2       |                   |                     |                  |                |                   |                  |

---------------------------------------
- cosine annealing for the learning rates with warmup steps
- FP16 training throughout all the steps to reduce the model training time
- regularization using weight decay
- using pearsons' and spearmans' coefficient for evaluation

### Usage
- training for mse and cross entropy loss functions is split into two main scripts, ```frozen_training.py``` and ```end2end_training.py``` and they should be used in that order
- training for cosine similarity loss is in end2end_training_sentence.py
- all scripts are meant to be called from the ```src/scripts``` folder and all of them have default arguments
```
cd src/scripts
python frozen_training.py --model_name bert-base-cased
python end2end_training.py --model_name bert-base-cased

python end2end_training_sentence.py --model_name t5-base
```
