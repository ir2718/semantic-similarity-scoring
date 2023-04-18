import spacy
from tqdm import tqdm
import pandas as pd
from utils import *

def calculate_handmade_features(dataset, path, stratify=False):
    d = dataset.remove_columns(['idx', 'label'])
    train_sentences, validation_sentences, test_sentences = pd.DataFrame(d['train']), pd.DataFrame(d['validation']), pd.DataFrame(d['test'])
    splits = [train_sentences, validation_sentences, test_sentences]

    nlp = spacy.load("en_core_web_trf")

    print('Calculating handmade features . . .')
    list_dicts = []
    for split in splits:
        num_chars_s1, num_chars_s2 = [], []
        num_tokens_s1, num_tokens_s2 = [], []
        num_stopwords_s1, num_stopwords_s2 = [], []
        num_token_overlap, num_lemma_overlap = [], []
        num_nouns_s1, num_nouns_s2 = [], []
        num_verbs_s1, num_verbs_s2 = [], []
        num_adj_s1, num_adj_s2 = [], []
        for i, row in tqdm(split.iterrows()):
            s1, s2 = row['sentence1'], row['sentence2']
            
            doc1, doc2 = nlp(s1), nlp(s2)
            tokens1, tokens2 = [token for token in doc1], [token for token in doc2]

            num_chars_s1.append(len(list(s1)))
            num_chars_s2.append(len(list(s2)))

            num_tokens_s1.append(len(doc1))
            num_tokens_s2.append(len(doc2))
            
            num_stopwords_s1.append(len([token for token in doc1 if token.is_stop]))
            num_stopwords_s2.append(len([token for token in doc2 if token.is_stop]))
            
            num_token_overlap.append(len(set([token.text for token in tokens1]) & set([token.text for token in tokens2])))
            num_lemma_overlap.append(len(set([token.lemma_ for token in tokens1]) & set([token.lemma_ for token in tokens2])))
            
            num_nouns_s1.append(len([token for token in doc1 if token.pos_ == 'NOUN']))
            num_nouns_s2.append(len([token for token in doc2 if token.pos_ == 'NOUN']))
            
            num_verbs_s1.append(len([token for token in doc1 if token.pos_ == 'VERB']))
            num_verbs_s2.append(len([token for token in doc2 if token.pos_ == 'VERB']))
            
            num_adj_s1.append(len([token for token in doc1 if token.pos_ == 'ADJ']))
            num_adj_s2.append(len([token for token in doc2 if token.pos_ == 'ADJ']))

        features = pd.DataFrame.from_dict({
            'num_chars_s1': num_chars_s1,
            'num_chars_s2': num_chars_s2,
            'num_tokens_s1': num_tokens_s1,
            'num_tokens_s2': num_tokens_s2,
            'num_stopwords_s1': num_stopwords_s1,
            'num_stopwords_s2': num_stopwords_s2,
            'num_token_overlap': num_token_overlap,
            'num_lemma_overlap': num_lemma_overlap,
            'num_nouns_s1': num_nouns_s1,
            'num_nouns_s2': num_nouns_s2,
            'num_verbs_s1': num_verbs_s1,
            'num_verbs_s2': num_verbs_s2,
            'num_adj_s1': num_adj_s1,
            'num_adj_s2': num_adj_s2
        })
        list_dicts.append(features)

    list_dicts[0].to_csv(f'{path}\\stratified_train_features.csv' if stratify else f'{path}\\train_features.csv', index=False)
    list_dicts[1].to_csv(f'{path}\\stratified_validation_features.csv' if stratify else f'{path}\\validation_features.csv', index=False)
    list_dicts[2].to_csv(f'{path}\\stratified_test_features.csv' if stratify else f'{path}\\test_features.csv', index=False)
    
def main():
    stratify = True

    dataset_path = '../dataset/stsb'
    if stratify:
        dataset = load_stratified_dataset(dataset_path)
    else:
        dataset = load_dataset_from_huggingface(DATASET_PATH, CONFIG_NAME, dataset_path)
    print('Calculating . . .')
    calculate_handmade_features(dataset, dataset_path, stratify)

main()