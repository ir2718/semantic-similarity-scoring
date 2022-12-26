import pandas as pd
import datasets
import numpy as np

def main():
    f = open("sts-test.tsv", "r", encoding='utf-8')
    
    l = []
    for i, x in enumerate(f):
        curr_line = x.split('\t')
        l.append(curr_line)
    
    labels = np.array([np.float32(x[4]) for x in l])
    sentence1 = np.array([x[5].strip() for x in l])
    sentence2 = np.array([x[6].strip() for x in l])

    df = pd.DataFrame.from_dict({
        'label':labels,
        'sentence1':sentence1,
        'sentence2':sentence2,
    })

    hf_df = datasets.load_dataset('glue', 'stsb', split='test')
    for i in range(1379):
        a = df.iloc[i]['sentence1']
        b = hf_df['sentence1'][i]
        a2 = df.iloc[i]['sentence2']
        b2 = hf_df['sentence2'][i]
        if a != b or a2 != b2:
            print(f'problem index: {i}')
            print(a)
            print(b)


main()