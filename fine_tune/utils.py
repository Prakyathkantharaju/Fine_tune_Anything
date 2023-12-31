from typing import List, Tuple, Dict, Union
from tqdm import tqdm
import zipfile, io, os, logging, requests
import pandas as pd
from sklearn.model_selection import train_test_split


def _download_extract_zip(url: str, destination_folder: str) -> None:
    response = requests.get(url)
    if response.status_code == 200:
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
        zip_file.extractall(destination_folder)
        logging.info(f"Files extracted to {destination_folder}")
    else: logging.warn(f"Failed to download: Status code {response.status_code}")


def download_data(args: Dict, location: str) -> None:
    repositories = args['repositories']
    for key, value in repositories.items():
        logging.info(f"Downloading {key} from {value}") 
        if not os.path.exists("data/" + key):
            os.mkdir("data/" + key)
            repo_url = value + "/zipball/main/"
            destination_folder =  location + key if location.endswith('/') else location + '/' + value
            print(destination_folder)
            _download_extract_zip(repo_url, destination_folder)

# generating a json with the parsing data, split by the \n

def _generate_data(file_name: str) -> Union[List[Dict], None]:
    with open(file_name, 'r') as f:
        data = f.readlines()
    data = [x for x in data if x != '\n']
    if len(data) < 5:
        return None
    return [{'label':i, 'text': c} for i, c in enumerate(data)]


def _generate_dataframe(files: List[str]) -> pd.DataFrame:
    data  = [] 
    for file in files:
        a = _generate_data(file)
        if a is not None:
            data += a

    return pd.DataFrame.from_records(data)

def _generate_parquet(df, file_name):
    df.to_parquet(file_name)


def save_parquet_files(path: str) -> None:
    files_path = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.py'):
                files_path.append(os.path.join(root, file))
    df = _generate_dataframe(files_path)
    train, test = train_test_split(df, test_size=0.2)
    _generate_parquet(train, 'data/train.parquet')
    _generate_parquet(test, 'data/test.parquet')