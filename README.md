# semantic-similarity-scoring

## WORK IN PROGRESS ##

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

### Hyperparameter optimization
- will be done using [Population-based training](https://arxiv.org/pdf/1711.09846.pdf)

### Masked language modeling
- might be useful, but will be left out for now because of computation cost

### Finetuning with the frozen encoder
- finetuning for 10 epochs or using early stopping

| Model used      | Train set     | Validation  set | Test set     |
| --------------- | ------------- | --------------- | ------------ |
| bert-base-cased | 0.945/0.948   | 0.8505/0.843    | 0.742/0.709  |

### End2End finetuning
- finetuning for 5 epochs or using early stopping

### Ensembling the finetuned models
- using a voting classifier or gradient boosting algorithms
---------------------------------------
- cosine annealing/linear scheduling for the learning rates with warmup steps
- FP16 training throughout all the steps to reduce the model training time
- regularization will be done using weight decay
- using pearsons' and spearmans' coefficient for evaluation
