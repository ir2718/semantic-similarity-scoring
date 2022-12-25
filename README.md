# semantic-similarity-scoring

## SOTA models for this task on STSB:
- [SMART-RoBERTa Large](https://arxiv.org/pdf/1911.03437v5.pdf)
- [StructBERT](https://arxiv.org/pdf/1908.04577v3.pdf)
- [MNet-Sim](https://arxiv.org/ftp/arxiv/papers/2111/2111.05412.pdf)
- [T5-11B](https://arxiv.org/pdf/1910.10683v3.pdf)
- [XLNet](https://arxiv.org/pdf/1906.08237v2.pdf)
- [RoBERTa](https://arxiv.org/pdf/1907.11692v1.pdf)
- [Vector-wise](https://arxiv.org/pdf/2208.07339v2.pdf)
- [EFL](https://arxiv.org/pdf/2104.14690v1.pdf)
- [Ernie 2.0 Large](https://arxiv.org/pdf/1907.12412v2.pdf)
- [DistilBERT](https://arxiv.org/pdf/1910.01108v4.pdf)

## Project notes:

### Hyperparameter optimization
- will be done using [Population-based training](https://arxiv.org/pdf/1711.09846.pdf)

### Masked language modeling
- might be useful, but will be left out for now because of computation cost

### Finetuning with the frozen encoder
- finetuning for 10 epochs or using early stopping

### End2End finetuning
- finetuning for 5 epochs or using early stopping

---------------------------------------
- cosine annealing/linear scheduling for the learning rates with warmup steps
- FP16 training throughout all the steps to reduce the model training time
- regularization will be done using weight decay
- using pearsons' and spearmans' coefficient for evaluation

### Ensembling the finetuned models
- using a voting classifier or gradient boosting algorithms
