# Fine-Tuning Repository for Causal Language Models

## Overview 🌟
This repository offers tools for fine-tuning causal language models, leveraging the power of the PEFT (PyTorch Efficient Fine-Tuning) library. It's designed for enhancing code generation tailored to specific repositories.

## Features 🛠️
- **Model Flexibility** 📈: Load any causal language model.
- **Configuration-Based Setup** ⚙️: Easily define model selection and tokenization through a `conf/config.yaml` file.
- **Efficient Fine-Tuning** 🏃‍♂️: Currently utilizing LoRA (Low-Rank Adaptation) via the PEFT library. More methods coming soon!
- **Future-Ready** 🔮: Stay tuned for training methods for instruct and QA models.

## Getting Started 🚦

### Prerequisites 📋
- Python 3.x 🐍
- PyTorch 🔥
- PEFT Library 📚
- Other dependencies listed in `requirements.txt`

### Installation 📥
1. Clone the repository: `git clone [repository URL]` :git:
2. Make a place for storing data `mkdir data` 📂 make sure to update `config.yaml` accordingly.
3. Install required packages: `pip install -r requirements.txt` 

### Configuration 🛠️
Tweak the `conf\config.yaml` file to set your model and tokenization parameters.

### Usage 🖥️
Kick off the fine-tuning process by running the main script (`python main.py`). Fine tuned model will be saved in  directory specified in `config.yaml` -> "optimization" -> result_dir.

## Contributing 🤝
We welcome your contributions to make this tool even more efficient and feature-rich. Please adhere to standard open-source contribution guidelines.

## License 📄
Apache License 2.0
