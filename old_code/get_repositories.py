


import requests
import zipfile
import io
import os

repositories = {
    'matplotlib': "https://github.com/matplotlib/matplotlib",
    'numpy': "https://github.com/numpy/numpy",
    'pandas': "https://github.com/pandas-dev/pandas",
    'opencv': "https://github.com/opencv/opencv-python",
    'scikit-learn': "https://github.com/scikit-learn/scikit-learn",
    'pytorch': 'https://github.com/pytorch/pytorch',
    'tensorflow': 'https://github.com/tensorflow/tensorflow',
    'huggingface': 'https://github.com/huggingface/transformers',
    'accelerate': 'https://github.com/huggingface/accelerate',
}

def download_extract_zip(url, destination_folder):
    response = requests.get(url)
    if response.status_code == 200:
        zip_file = zipfile.ZipFile(io.BytesIO(response.content))
        zip_file.extractall(destination_folder)
        print(f"Files extracted to {destination_folder}")
    else: print(f"Failed to download: Status code {response.status_code}")

# Example usage
repo_url_appending = "/zipball/main/"  # Replace with your repo ZIP link


for folder_name, repo in repositories.items():
    if not os.path.exists("data/" + folder_name):
        os.mkdir("data/" + folder_name)
        repo_url = repo + repo_url_appending
        destination_folder = "data/" + folder_name
        download_extract_zip(repo_url, destination_folder)
