import jax
from flax import nnx

class ProjectionHead(nnx.Module) :
    
    def __init__(self, in_dim: int, out_dim: int, rngs: nnx.Rngs, activation = None) -> None:
        self.proj = nnx.Linear(in_dim, out_dim, rngs=rngs)
        self.activation = activation
        
    def __call__(self, x: jax.Array) -> jax.Array:
        h = self.proj(x)
        return self.activation(h) if self.activation is not None else h