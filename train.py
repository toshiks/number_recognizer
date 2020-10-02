import hydra

from omegaconf import DictConfig

from app import train_model


@hydra.main(config_path="conf", config_name="train_conf")
def main(cfg: DictConfig) -> None:
    train_model(cfg)


if __name__ == '__main__':
    hydra.output_subdir = None
    main()
