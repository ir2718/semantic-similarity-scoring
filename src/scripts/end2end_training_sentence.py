import torch.nn as nn
from sentence_transformers import SentenceTransformer, models, losses
from utils import *
from parsing import parse_sentence_transformer
from configs import END2END_CFG

def train_model(batch_size, lr, weight_decay, train_dataloader):
    model_name = END2END_CFG.MODEL_NAME
    model = SentenceTransformer(model_name, device=device)
    output_dir = os.path.join(f'..\\models\\cosine_similarity\\end2end', model_name.replace('/', '-'))
    loss = losses.CosineSimilarityLoss(model)

    validation_evaluator = create_evaluator(validation, END2END_CFG.VAL_BATCH_SIZE)

    warmup_steps = int(END2END_CFG.WARMUP_RATIO * len(train_dataloader))
    optimizer = get_optimizer_from_string(END2END_CFG.OPTIM)

    model.fit(
        train_objectives=[(train_dataloader, loss)],
        evaluator=validation_evaluator,
        epochs=END2END_CFG.EPOCHS,
        scheduler=f'Warmup{END2END_CFG.SCHEDULER.capitalize()}',
        optimizer_class=optimizer,
        optimizer_params={'lr': lr},
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        use_amp=True,
        show_progress_bar=True,
        output_path=output_dir,
        save_best_model=True,
    )

    print('Evaluating best model . . .')
    model = SentenceTransformer(output_dir)
    train_evaluator = create_evaluator(train, batch_size)
    test_evaluator = create_evaluator(test, END2END_CFG.VAL_BATCH_SIZE)

    res_dir = os.path.join(output_dir, f'batchsize_{batch_size}_lr_{lr}_decay_{weight_decay}')
    evaluate(train_evaluator, model, output_path=os.path.join(res_dir, 'train_results'))
    evaluate(validation_evaluator, model, output_path=os.path.join(res_dir, 'validation_results'))
    evaluate(test_evaluator, model,  output_path=os.path.join(res_dir, 'test_results'))


args = parse_sentence_transformer()

END2END_CFG.set_args(args)
set_seed_(END2END_CFG.SEED)
device = set_device()

dataset = load_dataset_from_huggingface(DATASET_PATH, CONFIG_NAME, args.dataset_path)
train, validation, test = to_sentence_transformers_dataset_for_similarity(dataset)


if args.hyperopt:
    learning_rates = [1e-4, 3e-4, 5e-4]
    weight_decays = [1e-4, 1e-2]
    batch_sizes = [8, 32]

    for lr in learning_rates:
        for weight_decay in weight_decays:
            for batch_size in batch_sizes:

                train_dataloader, validation_dataloader, test_dataloader = get_dataloaders(
                    train, validation, test,
                    batch_size, END2END_CFG.VAL_BATCH_SIZE
                )

                print(f'Trying combination: {batch_size}, {lr}, {weight_decay}')
                train_model(batch_size, lr, weight_decay, train_dataloader)
else:

    train_dataloader, validation_dataloader, test_dataloader = get_dataloaders(
        train, validation, test,
        END2END_CFG.TRAIN_BATCH_SIZE,
        END2END_CFG.VAL_BATCH_SIZE
    )
    train_model(END2END_CFG.TRAIN_BATCH_SIZE, END2END_CFG.LEARNING_RATE_START, END2END_CFG.WEIGHT_DECAY, train_dataloader)
