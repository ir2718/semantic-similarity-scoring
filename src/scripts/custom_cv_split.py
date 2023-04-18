from utils import *
from sklearn.model_selection import train_test_split

seed = 42
num_bins = 20

label_dir = '../dataset/stsb'
dataset = load_dataset_from_huggingface(DATASET_PATH, CONFIG_NAME, label_dir)

train_df, validation_df, test_df = dataset['train'].to_pandas(), dataset['validation'].to_pandas(), dataset['test'].to_pandas()
df = pd.concat((train_df, validation_df, test_df), axis=0)

y = df['label'].values
bins = np.linspace(0, 5, num_bins)
y_binned = np.digitize(y, bins)

train_valid_percentage = (train_df.shape[0] + validation_df.shape[0])/df.shape[0]
X_train, X_test, y_train, y_test = train_test_split(
    df[['idx', 'sentence1', 'sentence2']], df[['label']], 
    test_size=1 - train_valid_percentage, 
    stratify=y_binned,
    random_state=seed
)

y = y_train['label'].values
bins = np.linspace(0, 5, num_bins)
y_binned = np.digitize(y, bins)

train_percentage = (train_df.shape[0])/(train_df.shape[0] + validation_df.shape[0])
X_train, X_validation, y_train, y_validation = train_test_split(
    X_train[['idx', 'sentence1', 'sentence2']], y_train[['label']], 
    test_size=1 - train_percentage, 
    stratify=y_binned,
    random_state=seed
)


train_new = pd.concat((X_train, y_train), axis=1)
validation_new = pd.concat((X_validation, y_validation), axis=1)
test_new = pd.concat((X_test, y_test), axis=1)

train_new.to_csv('../dataset/stsb/train_stratified.csv', index=False)
validation_new.to_csv('../dataset/stsb/validation_stratified.csv', index=False)
test_new.to_csv('../dataset/stsb/test_stratified.csv', index=False)