from typing import Any, Dict, List, Union, Tuple
import warnings, logging, re
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, DataCollatorForLanguageModeling, AutoModelForCausalLM
from datasets import Dataset, DatasetDict
import peft
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss
from transformers import Trainer, TrainingArguments


def LoadQbitmodel(model_name: str, quantization_config: dict) -> Any:
    """
    Load a quantized model.
    """
    from transformers import BitsAndBytesConfig
    config = BitsAndBytesConfig(
        load_in_4bit=quantization_config['load_in_4bit'],
        bnb_4bit_quant_type=quantization_config['bnb_4bit_quant_type'],
        bnb_4bit_use_double_quant=quantization_config['bnb_4bit_use_double_quant'],
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModel.from_pretrained(model_name, quantization_config=quantization_config)
    return model

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation for cross-entropy loss.
        """
        # Shift input_ids to the right to create labels
        input_ids = inputs["input_ids"]
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100  # Typically, -100 is used to mask the loss to ignore the prediction
        inputs['labels'] = labels

        # Perform forward pass
        outputs = model(**inputs)
        logits = outputs.logits

        # Compute loss
        loss_fct = CrossEntropyLoss()
        # Reshape logits and labels to be [batch_size * sequence_length, vocab_size]
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                                                    
        return (loss, outputs) if return_outputs else loss


class Lora_fine_tuning:
    def __init__(self,
                 model_name: str,
                 tokenizer_name: str,
                 config: Dict) -> None:
        if config['Optimization']['quantization']:
            self.model = LoadQbitmodel(model_name, config['quantization_config'])
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        # self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen2-1B")
        # self.model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen2-1B", trust_remote_code=True, revision="main")

        self._config = config
        logging.info("Model and tokenizer loaded")
        self._path = config['Optimization']['data_dir']
        if 'lora_config' not in config:
            raise Exception("lora_config not found in config")
        self._lora_config = config['lora_config']
        logging.info("loading data")
        training, validation = self._dataset()
        logging.info("data loaded")
        print(self.tokenized_ds.column_names)
        self._indetify_modules()
        self._peft(self._lora_config)
        self._train(training, validation)
        

    def _dataset(self) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        self._generate_token()
        self._setup_data_collator()
        training = self._create_training_dataset()
        validation = self._create_validation_dataset()
        return training, validation
        
        
    def _create_training_dataset(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.tokenized_ds['train'],
                                    collate_fn=self.data_collator,
                                    batch_size=32, #TODO this needs to be a hyperparameter
                                    shuffle=True)
    def _create_validation_dataset(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(self.tokenized_ds['validation'],
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
        self.tokenizer.pad_token = self.tokenizer.eos_token
        test = self.tokenizer(["".join(x) for x in file['text']], 
                              max_length=128, truncation=True) #TODO: change the max_length to a config file
        # make sure the length is less than 2048
        test['input_ids'] = [x for x in test['input_ids']]
        test['attention_mask'] = [x for x in test['attention_mask']]
        # test['labels'] = test['input_ids'].copy()
        return test
    
    def _setup_data_collator(self) -> None:
        self.tokenizer.return_special_tokens_mask = True
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)

    def _indetify_modules(self) -> None:
        self._target_modules = []
        pattern = r'' + self._lora_config['target_modules']
        layers = re.findall(pattern, str(self.model.modules))
        self._lora_config['target_modules'] = list(set(layers))
        logging.info("Target modules identified")
        logging.info("Target modules: {}".format(self._lora_config['target_modules']))
    
    # def _delete_dataframes(sielf) -> None:
    #     del self.ds


    def _peft(self, arg: Dict) -> None:
        lora_config = dict(self._lora_config)
        config = peft.LoraConfig(r=8,
                target_modules=['lm_head', 'fc_in', 'out_proj', 'fc_out', 'qkv_proj'],
                lora_dropout=0.05,
                task_type = "CAUSAL_LM")
        logging.info("Lora config loaded")
        logging.info(f"lora config: {config}")
        if self._config['quantization']:
            from peft import prepare_model_for_kbit_training
            self.model = prepare_model_for_kbit_training(self.model)
            self.peft_model = peft.get_peft_model(self.model, config)
        else:
            self.peft_model = peft.get_peft_model(self.model, config)
        logging.info("PEFT model loaded")
        self.peft_model.print_trainable_parameters()
    
    def _train(self, training: torch.utils.data.DataLoader, validation: torch.utils.data.DataLoader) -> None:
        training_args = TrainingArguments(
            output_dir='./results',          # output directory
            num_train_epochs=5,              # total number of training epochs
            per_device_train_batch_size=32,  # batch size per device during training
            per_device_eval_batch_size=32,   # batch size for evaluation
            logging_dir='./logs',            # directory for storing logs
            logging_steps=10,
        )
        trainer = CustomTrainer(
            model=self.peft_model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            train_dataset=self.tokenized_ds['train'],         # training dataset
            eval_dataset=self.tokenized_ds['validation'],
            data_collator=self.data_collator             # evaluation dataset
        )
        trainer.train()
        
        # training_loss = []
        # validation_loss = []
        # self.model.to('cuda')
        # # training loop
        # for epoch in trange(0, self._config['Optimization']['epochs']):
        #     self.peft_model.train()
        #     _traininig_loss = []
        #     for batch in training:
        #         self.peft_model.zero_grad()
        #         print(batch.keys())
        #         outputs = self.peft_model(batch['input_ids'].to("cuda"), batch['attention_mask'].to("cuda"))
        #         loss = outputs.loss
        #         loss.backward()
        #         self.peft_model.optimizer.step()
        #         _traininig_loss.append(loss.item())
        #         print(f"Training loss: {loss.item()}")
        #     training_loss.append(sum(_traininig_loss) / len(_traininig_loss))
            
            
        #     # validation loop
        #     self.peft_model.eval()
        #     _validataion_loss = []
        #     for batch in validation:
        #         outputs = self.peft_model(batch['input_ids'].to("cuda"), batch['attention_mask'].to("cuda"))
        #         loss = outputs.loss
        #         self.peft_model.scheduler.step(loss)
        #         _validataion_loss.append(loss.item())
        #     validation_loss.append(sum(_validataion_loss) / len(_validataion_loss))
        #     # This is for debugging
        #     print(f"Epoch: {epoch} Training loss: {training_loss[-1]} Validation loss: {validation_loss[-1]}")

        #     if epoch % 10 == 0:
        #         logging.info(f"Epoch: {epoch} Training loss: {training_loss[-1]} Validation loss: {validation_loss[-1]}")
        #         self._save(training_loss, validation_loss)



    def _save(self, training_loss: List, validation_loss: List) -> None:
        pass

    def _load(self) -> None:
        pass
