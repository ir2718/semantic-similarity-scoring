from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import make_scorer
import pickle
import pandas as pd
from bisect import insort
import os
import itertools
from tqdm import tqdm
from configs import ENSEMBLE_CFG
from parsing import *
from utils import *

def read_features(loss, model_name, clip=True):
    df_train = pd.read_csv(f'..\\models\\{loss}\\end2end\\{model_name}\\train_res.csv')
    df_validation = pd.read_csv(f'..\\models\\{loss}\\end2end\\{model_name}\\validation_res.csv')
    df_test = pd.read_csv(f'..\\models\\{loss}\\end2end\\{model_name}\\test_res.csv')

    if clip:
        df_train.clip(0, 5, inplace=True)
        df_validation.clip(0, 5, inplace=True)
        df_test.clip(0, 5, inplace=True)

    return df_train, df_validation, df_test

def prepare_data_for_model(X_train, X_validation, X_test, y_train, y_validation, y_test):
    train = pd.concat((X_train, y_train), axis=1)
    validation = pd.concat((X_validation, y_validation), axis=1)
    test = pd.concat((X_test, y_test), axis=1)
    return train, validation, test

def prepare_data_for_grid_search(X_train, X_validation, y_train, y_validation):
    x = np.concatenate([X_train.to_numpy(), X_validation.to_numpy()])
    y = np.concatenate([y_train.to_numpy(), y_validation.to_numpy()])
    train_val_fold = np.concatenate([
        np.ones(X_train.shape[0]) * -1,
        np.zeros(X_validation.shape[0])
    ])
    cv = PredefinedSplit(train_val_fold)

    return cv, x, y

def get_transformer_features(models, loss):
    X_train, X_validation, X_test = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    for m in models:
        train, validation, test = read_features(loss, m)
        X_train[m] = train 
        X_validation[m] = validation
        X_test[m] = test

    return X_train, X_validation, X_test

def ensemble_hyperparameter_search(loss, model_name, num_models, dataset, seed, dataset_path, save_path=None):
    models = os.listdir(f'..\\models\\{loss}\\end2end')
    combs = list(itertools.combinations(models, num_models))

    d = dataset.remove_columns(['sentence1', 'sentence2', 'idx'])
    y_train, y_validation = pd.DataFrame(d['train']), pd.DataFrame(d['validation'])
    
    hyperparams = ensemble_hyperparameter_space(model_name)
    train_features = pd.read_csv(f'{dataset_path}\\train_features.csv')
    validation_features = pd.read_csv(f'{dataset_path}\\validation_features.csv')

    best = []
    print('Starting hyperparameter search . . .')
    for c in tqdm(combs):
        X_train, X_validation, _ = get_transformer_features(c, loss)

        X_train = pd.concat((X_train, train_features), axis=1)
        X_validation = pd.concat((X_validation, validation_features), axis=1)

        cv, x, y = prepare_data_for_grid_search(X_train, X_validation, y_train, y_validation)
        model_f = get_ensemble_model(model_name)

        search = GridSearchCV(
            estimator=model_f(random_state=seed), 
            param_grid=hyperparams,
            scoring=make_scorer(scoring_function_pearson),
            n_jobs=-1,
            cv=cv,
        )
        search.fit(x, y.reshape(-1))
        insort(best, (search.best_score_, c, search.best_params_))

    results = sorted(best, reverse=True)
    hp_df = create_hyperparam_df(results)

    if save_path:
        output_dir = f'{save_path}\\{model_name}_{loss}_{num_models}'
        os.makedirs(output_dir, exist_ok=True)
        hp_df.to_csv(os.path.join(output_dir, f'hp_search_res.txt'), index=False)

    return hp_df

def create_hyperparam_df(best):
    d = {'pearson_r': [], 'models': []}
    
    for k in best[0][2].keys():
        d[k] = []

    for x in best:
        d['pearson_r'].append(x[0])
        d['models'].append(x[1])
        for k in x[2].keys():
            d[k].append(x[2][k])

    return pd.DataFrame.from_dict(d)

def get_ensemble_model(model_name):
    models = {
        'xgboost': XGBRegressor,
        'lgbm': LGBMRegressor,
        'adaboost': AdaBoostRegressor,
    }
    return models[model_name]

def train_ensemble(model_name, models, params, dataset, save_path, loss, seed, dataset_path, stratified):
    d = dataset.remove_columns(['sentence1', 'sentence2', 'idx'])
    y_train, y_validation, y_test = pd.DataFrame(d['train']), pd.DataFrame(d['validation']), pd.DataFrame(d['test'])
    X_train, X_validation, X_test = get_transformer_features(models, loss)

    train_features = pd.read_csv(f'{dataset_path}\\train_features.csv' if not stratified else f'{dataset_path}\\stratified_train_features.csv')
    validation_features = pd.read_csv(f'{dataset_path}\\validation_features.csv' if not stratified else f'{dataset_path}\\stratified_validation_features.csv')
    test_features = pd.read_csv(f'{dataset_path}\\test_features.csv' if not stratified else f'{dataset_path}\\stratified_test_features.csv')

    X_train = pd.concat((X_train, train_features), axis=1)
    X_validation = pd.concat((X_validation, validation_features), axis=1)
    X_test = pd.concat((X_test, test_features), axis=1)

    model_f = get_ensemble_model(model_name)
    model = model_f(random_state=seed, **params)
    model.fit(X_train, y_train)

    output_dir = f'{save_path}\\{model_name}_{loss}_{len(models)}'
    model_save_path = os.path.join(output_dir, f'best_model_{model_name}_{loss}_{len(models)}.pkl')
    pickle.dump(model, open(model_save_path, 'wb'))

    train_res = compute_metrics_(model.predict(X_train), y_train.to_numpy().reshape(-1))
    validation_res = compute_metrics_(model.predict(X_validation), y_validation.to_numpy().reshape(-1))
    test_res = compute_metrics_(model.predict(X_test), y_test.to_numpy().reshape(-1))

    train_metrics = {f'train_{k}': v for k, v in train_res.items()}
    validation_metrics = {f'validation_{k}': v for k, v in validation_res.items()}
    test_metrics = {f'test_{k}':v for k, v in test_res.items()}

    metrics_d = {**train_metrics, **validation_metrics, **test_metrics}
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as metric_info:
        metric_info.write(json.dumps(metrics_d))


def ensemble_hyperparameter_space(model_name):
    hyperparams = {
        'xgboost': {
            'max_depth': [2, 3, 4, 5, 6], 
            'min_child_weight': [1, 3, 5],
            'subsample': np.linspace(3, 10, 5) * 0.1,
            'eta': np.logspace(1e-4, 1e-1, 5)
        },
        'adaboost': {
            'n_estimators': [20, 50, 100, 200, 300, 500],
            'learning_rate': [1e-2, 1e-1, 1],
            'loss': ['linear', 'square', 'exponential']
        },
        'lgbm': {
            'boosting_type': ['gbdt', 'dart', 'goss'],
            'num_leaves': [20, 22, 25, 28, 31, 33, 35, 38, 40, 45, 50, 60, 80, 100],
            'max_depth': [2, 3, 4, 5, 8, 10],

        },
    }
    return hyperparams[model_name]

def main():
    args = parse_ensemble()
    set_seed_(args.seed)
    ENSEMBLE_CFG.set_args(args)
        
    if args.stratified:
        dataset = load_stratified_dataset(args.dataset_path)
    else:
        dataset = load_dataset_from_huggingface(DATASET_PATH, CONFIG_NAME, args.dataset_path)

    if args.hyperopt:
        hp_df = ensemble_hyperparameter_search(
            args.loss_function, 
            ENSEMBLE_CFG.MODEL_NAME, 
            ENSEMBLE_CFG.NUM_MODELS, 
            dataset, 
            ENSEMBLE_CFG.SEED,
            args.dataset_path,
            args.save_path,
        )
        print('Finished hyperparameter search')

        models = hp_df[['models']].iloc[0].to_dict()['models']
        best_params = {k: v[0] for k, v in hp_df.drop(columns=['pearson_r', 'models']).to_dict().items()}

        print('Retraining best model . . .')
        train_ensemble(
            model_name=ENSEMBLE_CFG.MODEL_NAME,
            models=models, 
            params=best_params,
            dataset=dataset,
            save_path=args.save_path, 
            loss=args.loss_function,
            seed=args.seed,
            dataset_path=args.dataset_path,
            stratified=args.stratified
        )
    else:
        train_ensemble(
            model_name=ENSEMBLE_CFG.MODEL_NAME,
            models=args.models, 
            params=args.kwargs,
            dataset=dataset,
            save_path=args.save_path, 
            loss=args.loss_function,
            seed=args.seed,
            dataset_path=args.dataset_path,
            stratified=args.stratified
        )


main()