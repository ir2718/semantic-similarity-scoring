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

### 1. Masked language modeling
Since this project uses a private dataset that comprises of students' theses the pretrained models will behave poorly on the language modeling task. In order to tackle this domain adaptation will be used, in the form of MLM pretraining.

### 2. Finetuning with the frozen encoder
Next up, the task to be solved is determining the semantic textual similarity of pairs of theses/summaries/abstracts. Besides improving training time, this technique will learn the weights for the regression head.

### 3. End2End finetuning
The final step for each of these models will be finetuning the whole model, including the encoder. This will be done using a significantly lower learning rate compared to the last step.

- Each of the steps will feature hyperparameter tuning using [Population-based training](https://arxiv.org/pdf/1711.09846.pdf)
- Cosine annealing/linear scheduling for the learning rates with warmup steps
- FP16 training throughout all the steps to reduce the model training time
- Regularization will be done using weight decay
- Each model will be evaluated using Pearsons' and Spearmans' coefficient as these are the default metrics for the task

### 4. Ensembling the finetuned models
Finally, the last step will be combining the finetuned models using some kind of ensemble algorithm (voting, stacking, bagging, boosting).
