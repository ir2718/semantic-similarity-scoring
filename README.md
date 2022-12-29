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
- when using PBT the same values for the learning rate and weight decay are used: 
```
  learning_rate: [1e-3, 5e-4, 1e-4, 5e-5, 1e-5]
  weight_decay: [1e-2, 1e-3, 1e-4]
```
- the batch size grid used always contains exactly 3 values and varies depending on the size of the model

### Masked language modeling
- might be useful, but will be left out for now because of computation cost

### Finetuning with the frozen encoder
- finetuning for 10 epochs or using early stopping

| **Model**           | **Train set**     | **Validation  set** | **Test set**     | **Batch size** | **Learning rate** | **Weight decay** | **Batch size grid** |
| ------------------- | ----------------- | ------------------- | ---------------- |--------------- | ----------------- | ---------------- | ------------------- |
| BERT base cased     | 0.793/0.749       | 0.814/0.809         | 0.735/0.697      | 32             | 5e-4              | 1e-4             | [8, 16, 32]         |
|                     |                   |                     |                  |                |                   |                  |                     |

### End2End finetuning
- finetuning for 5 epochs or using early stopping

### Ensembling the finetuned models
- using a voting classifier or gradient boosting algorithms
---------------------------------------
- cosine annealing/linear scheduling for the learning rates with warmup steps
- FP16 training throughout all the steps to reduce the model training time
- regularization will be done using weight decay
- using pearsons' and spearmans' coefficient for evaluation
