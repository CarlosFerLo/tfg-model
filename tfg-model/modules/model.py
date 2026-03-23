import jax
from flax import nnx

from .embeddings import EmbeddingLayer
from .body import GNNBody
from .projection import ProjectionHead

class Model(nnx.Module) :
    
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        body_out_dim: int,
        conjecturer_out_dim: int,
        prover_out_dim: int,
        layers: int,
        rngs: nnx.Rngs
    ) -> None:
        self.embed = EmbeddingLayer(embed_dim, rngs=rngs)
        
        self.body = GNNBody(
            in_dim=embed_dim,
            hidden_dim=hidden_dim,
            out_dim=body_out_dim,
            layers=layers,
            rngs=rngs
        )
        
        self.conjecturer = ProjectionHead(body_out_dim, conjecturer_out_dim, rngs=rngs)
        self.prover = ProjectionHead(body_out_dim, prover_out_dim, rngs=rngs)
    