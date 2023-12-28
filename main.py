# loading the libraries
import hydra
from omegaconf import DictConfig, OmegaConf

# input args
from fine_tune.utils import get_args


@hydra.main(config_path="conf", config_name="config", version_base = None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
