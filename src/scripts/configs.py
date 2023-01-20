class FROZEN_CFG:
    @staticmethod
    def set_args(args):
        FROZEN_CFG.MODEL_NAME = args.model_name
        
        FROZEN_CFG.EPOCHS = args.num_epochs
        FROZEN_CFG.TRAIN_BATCH_SIZE = args.train_batch_size
        FROZEN_CFG.VAL_BATCH_SIZE = args.val_batch_size
        FROZEN_CFG.WEIGHT_DECAY = args.weight_decay
        FROZEN_CFG.LEARNING_RATE_START = args.learning_rate_start
        FROZEN_CFG.WARMUP_RATIO = args.warmup_ratio
        FROZEN_CFG.SCHEDULER = args.scheduler
        FROZEN_CFG.OPTIM = args.optim
        FROZEN_CFG.OPT_METRIC = args.opt_metric if hasattr(args, 'opt_metric') else None
        FROZEN_CFG.LOSS_FUNCTION = args.loss_function if hasattr(args, 'loss_function') else None

        FROZEN_CFG.SEED = args.seed
        FROZEN_CFG.PATIENCE = args.patience


class END2END_CFG:
    @staticmethod
    def set_args(args):
        END2END_CFG.MODEL_NAME = args.model_name
        
        END2END_CFG.EPOCHS = args.num_epochs
        END2END_CFG.TRAIN_BATCH_SIZE = args.train_batch_size
        END2END_CFG.VAL_BATCH_SIZE = args.val_batch_size
        END2END_CFG.WEIGHT_DECAY = args.weight_decay
        END2END_CFG.LEARNING_RATE_START = args.learning_rate_start
        END2END_CFG.WARMUP_RATIO = args.warmup_ratio
        END2END_CFG.SCHEDULER = args.scheduler
        END2END_CFG.OPTIM = args.optim
        END2END_CFG.OPT_METRIC = args.opt_metric if hasattr(args, 'opt_metric') else None
        END2END_CFG.LOSS_FUNCTION = args.loss_function if hasattr(args, 'loss_function') else None
        
        END2END_CFG.SEED = args.seed
        END2END_CFG.PATIENCE = args.patience

        END2END_CFG.GRAD_CHECKPOINT = args.grad_checkpoint if hasattr(args, 'grad_checkpoint') else None