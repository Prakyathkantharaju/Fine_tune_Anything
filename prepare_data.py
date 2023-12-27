import os
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split


# generating a json with the parsing data, split by the \n

def generate_data(file_name):
    with open(file_name, 'r') as f:
        data = f.readlines()
    data = [x for x in data if x != '\n']
    if len(data) < 5:
        return None
    return [{'label':0, 'text': c} for c in data]


def generate_dataframe(files):
    data  = [] 
    for file in tqdm(files):
        a = generate_data(file)
        if a is not None:
            data += a

    return pd.DataFrame.from_records(data)

def generate_parquet(df, file_name):
    df.to_parquet(file_name)


def main():
    files_path = []
    for root, dirs, files in os.walk('data'):
        for file in files:
            if file.endswith('.py'):
                files_path.append(os.path.join(root, file))
    df = generate_dataframe(files_path)
    train, test = train_test_split(df, test_size=0.2)
    generate_parquet(train, 'data/train.parquet')
    generate_parquet(test, 'data/test.parquet')

if __name__ == '__main__':
    main()
