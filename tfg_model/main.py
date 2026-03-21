import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

import logging

from .engine import GraphEngine

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    
    log.info("Starting Conjecturer-Prover pipeline...")
    log.info(f"Experiment: {cfg.experiment_name} | Seed: {cfg.seed}")
    
    wandb.init(
        project="tfg-model",
        name=cfg.experiment_name,
        config=OmegaConf.to_container(cfg, resolve=True) # type: ignore
    )
    
    log.info("W&B initialized successfully.")
    
    try :
        log.info("Initializing Graph Engine...")
        engine = GraphEngine()

        
    except Exception as e :
        log.exception(f"Pipeline crashed due to an error: {e}")
        raise e
    finally:
        wandb.finish()
        log.info("Run finished and W&B synced.")
        
if __name__ == "__main__" :
    main()
    