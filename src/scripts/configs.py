class MLM_CFG:
    @staticmethod
    def set_args(args):
        MLM_CFG.MODEL_NAME = args.model_name
        
        MLM_CFG.EPOCHS = args.num_epochs
        MLM_CFG.TRAIN_BATCH_SIZE = args.train_batch_size
        MLM_CFG.VAL_BATCH_SIZE = args.val_batch_size
        MLM_CFG.WEIGHT_DECAY = args.weight_decay
        MLM_CFG.LEARNING_RATE_START = args.learning_rate_start
        
        MLM_CFG.BLOCK_SIZE = args.block_size
        
        MLM_CFG.SEED = args.seed

class FROZEN_CFG:
    @staticmethod
    def set_args(args):
        FROZEN_CFG.MODEL_NAME = args.model_name
        FROZEN_CFG.PRETRAINED_PATH = args.pretrained_path
        
        FROZEN_CFG.EPOCHS = args.num_epochs
        FROZEN_CFG.TRAIN_BATCH_SIZE = args.train_batch_size
        FROZEN_CFG.VAL_BATCH_SIZE = args.val_batch_size
        FROZEN_CFG.WEIGHT_DECAY = args.weight_decay
        FROZEN_CFG.LEARNING_RATE_START = args.learning_rate_start
        FROZEN_CFG.WARMUP_RATIO = args.warmup_ratio
        FROZEN_CFG.SCHEDULER = args.scheduler
        
        FROZEN_CFG.SEED = args.seed

class END2END_CFG:
    @staticmethod
    def set_args(args):
        END2END_CFG.MODEL_NAME = args.model_name
        END2END_CFG.PRETRAINED_PATH = args.pretrained_path
        
        END2END_CFG.EPOCHS = args.num_epochs
        END2END_CFG.TRAIN_BATCH_SIZE = args.train_batch_size
        END2END_CFG.VAL_BATCH_SIZE = args.val_batch_size
        END2END_CFG.WEIGHT_DECAY = args.weight_decay
        END2END_CFG.LEARNING_RATE_START = args.learning_rate_start
        END2END_CFG.WARMUP_RATIO = args.warmup_ratio
        END2END_CFG.SCHEDULER = args.scheduler
        
        END2END_CFG.SEED = args.seed