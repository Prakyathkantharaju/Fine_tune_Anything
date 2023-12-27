from datasets import load_dataset


df = load_dataset("bigcode/the-stack", data_dir = "data/python",split="train", streaming=True)
for i, row in enumerate(df):
    print(row)
    if i > 10:
        break
