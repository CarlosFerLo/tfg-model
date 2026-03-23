import jax
import jax.numpy as jnp
from flax import nnx

from .gnn_layer import GNNLayer

class GNNBody(nnx.Module) :
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, layers: int, rngs: nnx.Rngs, activation = nnx.relu, dropout: float = 0.0, residual: bool = True) -> None :
        assert layers >= 2, "at least must have two layers"
        
        self.residual = residual
        
        self.layers = [
            GNNLayer(in_dim, hidden_dim, rngs=rngs, activation=activation)
        ] + [
            GNNLayer(hidden_dim, hidden_dim, rngs=rngs, activation=activation)
            for _ in range(layers - 2)
        ] + [
            GNNLayer(hidden_dim, out_dim, rngs=rngs, activation=activation)
        ]
        
        self.res_proj = [
            nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        ] + [None] * max(layers - 2, 0) + [
            nnx.Linear(hidden_dim, out_dim, rngs=rngs)
        ]
        
        self.dropouts = [
            nnx.Dropout(dropout, rngs=rngs) if dropout > 0.0 else None
            for _ in range(layers-1)
        ]
        
    def __call__(
        self,
        x: jax.Array,
        senders: jax.Array,
        receivers: jax.Array,
        *,
        training: bool = False
    ) -> jax.Array :
        
        for layer, res_proj, dropout in zip(self.layers, self.res_proj, self.dropouts) :
            h = layer(x, senders, receivers)
            
            if self.residual:
                skip = res_proj(x) if res_proj is not None else x
                h = h + skip
            
            if dropout is not None:
                h = dropout(h, deterministic = not training)
                
            x = h
            
        return x