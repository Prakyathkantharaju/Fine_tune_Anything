from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
from pprint import pprint
from transformers import BitsAndBytesConfig
import torch
config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
)
warnings.filterwarnings("ignore")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-2b-mono", trust_remote_code=True, quantization_config = config)
# model.to("cuda")
# pprint([(n, type(m)) for n, m in model.named_modules()])

text = "def hello_world(): <|endoftext|>"

eos = "<|endoftext|>"





input_ids = tokenizer(text, return_tensors="pt").input_ids
print(str(model.modules))
import re
pattern = r'\((\w+)\): Linear'
linear_layers = re.findall(pattern, str(model.modules))
target_modules = list(set(linear_layers))
print(target_modules)


import peft
from peft import prepare_model_for_kbit_training
model = prepare_model_for_kbit_training(model)
config = peft.LoraConfig(
    r=16,
    lora_alpha=8,
    target_modules=target_modules,
    lora_dropout=0.01,
    bias="none",
    task_type="CAUSAL_LM",
)

peft_model =peft.get_peft_model(model, config)
print(peft_model.print_trainable_parameters())
# print(input_ids)
# print(tokenizer.all_special_tokens)
# print(tokenizer.all_special_ids)
# print(tokenizer.batch)
#input_ids = input_ids.to("cuda")

#generated_ids = model.generate(input_ids, max_length=128)
#print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))


# def generate_one_completion(prompt):
#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids
#     input_ids = input_ids.to("cuda")
#     generated_ids = model.generate(input_ids, max_length=128, pad_token_id=tokenizer.eos_token_id)
#     return tokenizer.decode(generated_ids[0], skip_special_tokens=True)



# def eval():
#     from human_eval.data import write_jsonl, read_problems
#     from tqdm import tqdm, trange

#     problems = read_problems()

#     num_samples_per_task = 100
#     samples = [
#         dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
#         for task_id in tqdm(problems)
#         for _ in trange(num_samples_per_task)
#     ]
#     write_jsonl("samples.jsonl", samples)
