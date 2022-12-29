import os

def sh(x):
    os.system(x)

#sh('python frozen_training.py --model_name roberta-base')
#sh('python frozen_training.py --model_name distilroberta-base')
sh('python frozen_training.py --model_name distilbert-base-cased')
sh('python frozen_training.py --model_name xlnet-base-cased')
sh('python frozen_training.py --model_name t5-small')

#sh('python frozen_training.py --model_name roberta-large')
#sh('python frozen_training.py --model_name t5-base')
#sh('python frozen_training.py --model_name xlnet-large-cased')

#sh('python frozen_training.py --model_name microsoft/deberta-v3-small')
#sh('python frozen_training.py --model_name microsoft/deberta-v3-base')


