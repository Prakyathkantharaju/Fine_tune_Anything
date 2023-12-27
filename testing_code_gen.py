from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings("ignore")
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")
model = AutoModelForCausalLM.from_pretrained("Salesforce/codegen-350M-mono")
model.to("cuda")

#text = "def hello_world():"







#input_ids = tokenizer(text, return_tensors="pt").input_ids
#input_ids = input_ids.to("cuda")

#generated_ids = model.generate(input_ids, max_length=128)
#print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))


def generate_one_completion(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    input_ids = input_ids.to("cuda")
    generated_ids = model.generate(input_ids, max_length=128, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


from human_eval.data import write_jsonl, read_problems
from tqdm import tqdm, trange

problems = read_problems()

num_samples_per_task = 200
samples = [
    dict(task_id=task_id, completion=generate_one_completion(problems[task_id]["prompt"]))
    for task_id in tqdm(problems)
    for _ in trange(num_samples_per_task)
]
write_jsonl("samples.jsonl", samples)
