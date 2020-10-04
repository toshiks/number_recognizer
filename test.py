import hydra

from omegaconf import DictConfig

from app import RecognizeNumbers


@hydra.main(config_path="conf", config_name="test_conf")
def main(cfg: DictConfig) -> None:
    recognizer = RecognizeNumbers(cfg.model_config.path_to_graph, cfg.model_config.n_mels, cfg.model_config.device)
    recognizer.prediction(cfg.dataset_path)


if __name__ == '__main__':
    main()
