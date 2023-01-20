from torch import nn
from transformers import Trainer
from abc import abstractmethod

class MeanSquaredErrorTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get('labels').view(-1)

        outputs = model(**inputs)
        logits = outputs.get('logits').view(-1)
        
        loss_fct = nn.MSELoss()
        
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

    @abstractmethod
    def scale_labels(x):
        return x

class CrossEntropyLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get('labels').view(-1)

        outputs = model(**inputs)

        logits = outputs.get('logits').view(-1) 
        loss_fct = nn.BCEWithLogitsLoss()

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
        loss, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        sigmoid_fct = nn.Sigmoid()
        logits = sigmoid_fct(logits)
        return loss, logits, labels

    @abstractmethod
    def scale_labels(x):
        x['label'] = x['label'] / 5. # from range [0, 5] to range [0, 1]
        return x

def get_trainer_with_loss(loss_str):
    trainer_class_dict = {
        'mse': MeanSquaredErrorTrainer,
        'cross_entropy': CrossEntropyLossTrainer,
    }
    return trainer_class_dict[loss_str]