from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Config :
    model: "ModelConfig"
    
@dataclass
class ModelConfig :
    embed_dim: int = 10
    hidden_dim: int = 10
    body_out_dim: int = 10
    conjecturer_out_dim: int = 3
    prover_out_dim: int = 3
    layers: int = 3
    
CONFIG = Config(
    ModelConfig()
)