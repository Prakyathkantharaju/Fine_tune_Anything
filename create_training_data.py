from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
from pprint import pprint
import pandas as pd
from datasets import Dataset, DatasetDict
warnings.filterwarnings("ignore")


# setup tokenizer
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")

ts = pd.read_parquet('data/train.parquet')
vs = pd.read_parquet('data/test.parquet')
ds = Dataset.from_pandas(ts)
vs = Dataset.from_pandas(vs)

ds = DatasetDict({'train': ds, 'validation': vs})

def convert_sentence(file):
    test = tokenizer(["".join(x) + "<|endoftext|>"  for x in file['text']])
    test['labels'] = test['input_ids'].copy()
    return test

tokenized_ds = ds.map(convert_sentence, batched = True, num_proc = 4, remove_columns = ['label', '__index_level_0__', 'text'])



