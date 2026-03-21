import logging
from pathlib import Path

import optax
from flax import nnx

from .engine import GraphEngine
from .model import Model
from .dataset import Dataset

log = logging.getLogger(__name__)

class Trainer() :
    
    engine: GraphEngine
    
    def __init__(self, engine: GraphEngine, cfg, model: Model) -> None :
        log.info("Initializing Trainer...")
        self.engine = engine
        self.cfg = cfg
        self.model = model
        
        self.body_optimizer = nnx.Optimizer(model.body, optax.adam(cfg.model.body.learning_rate), wrt=nnx.Param)
        self.conjecturer_optimizer = nnx.Optimizer(model.conjecturer, optax.adam(cfg.model.conjecturer.learning_rate), wrt=nnx.Param)
        self.prover_optimizer = nnx.Optimizer(model.prover, optax.adam(cfg.model.prover.learning_rate), wrt=nnx.Param)
        
    def train(self) -> None :
        log.info("Start training")
        
        log.info("Supervised Training for aligning the conjecturer")
        
        log.info("Load dataset...")
        dataset = Dataset.load(Path(self.cfg.dataset.path))
        