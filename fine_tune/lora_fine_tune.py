from typing import Any, Dict, List, Union, Tuple
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, DataCollatorForLanguageModeling
from datasets import Dataset, DatasetDict
import warnings


class Lora_fine_tuning:
    def __init__(self,
                 model_name: str,
                 tokenizer_name: str,
                 config: Dict) -> None:

        self.model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self._path = config['path']

    def _dataset(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        self._generate_token()
        self._setup_data_collator()
        training = self._create_training_dataset()
        validation = self._create_validation_dataset()
        return training, validation
        
        
    def _create_training_dataset(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.ds['train'],
                                    collate_fn=self.data_collator,
                                    batch_size=32, #TODO this needs to be a hyperparameter
                                    shuffle=True)
    def _create_validation_dataset(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.ds['validation'],
                                    collate_fn=self.data_collator,
                                    batch_size=32, #TODO this needs to be a hyperparameter
                                    shuffle=True)
    

    def _generate_token(self) -> None:
        ts = Dataset.from_pandas(pd.read_parquet(self._path + 'train.parquet'))
        vs = Dataset.from_pandas(pd.read_parquet(self._path + 'test.parquet'))
        self.ds = DatasetDict({'train': ts, 'validation': vs})
        self.tokenized_ds = self.ds.map(self._convert_sentence, batched=True, num_proc=4,
                                        remove_columns=['label', '__index_level_0__', 'text'])

    def _convert_sentence(self, file: pd.DataFrame) -> Dict:
        test = self.tokenizer(["".join(x) for x in file['text']])
        test['labels'] = test['input_ids'].copy()
        return test
    
    def _setup_data_collator(self) -> None:
        self.tokenizer.return_special_tokens_mask = True
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

    def _peft(self, arg: Dict) -> None:
        pass
    
    def _train(self) -> None:
        pass

    def _eval(self) -> None:
        pass 

    def _save(self) -> None:
        pass

    def _load(self) -> None:
        pass

    def _generate_one_completion(self, prompt: str) -> str:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to("cuda")
        generated_ids = self.model.generate(input_ids, max_length=128, pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)