# loading the libraries
import hydra
from omegaconf import DictConfig, OmegaConf
from fine_tune.utils import download_data, save_parquet_files
from fine_tune.lora_fine_tune import Lora_fine_tuning

# input args
# from fine_tune.utils import get_args


@hydra.main(config_path="conf", config_name="config", version_base = None)
def main(cfg: DictConfig):
    config = OmegaConf.to_object(cfg)
    # Creating the data set using utils.
    # download the data and saves in the datadirectory this is just once
    download_data(config, config['Optimization']['data_dir'])

    # process the data and save it in the parquet file for easy loading. 
    save_parquet_files(config['Optimization']['data_dir'])

    # lora training
    Lora_fine_tuning(config['Optimization']['model_name'],
                     config['Optimization']['tokenizer_name'],
                     config)
    
    




if __name__ == "__main__":
    main()
