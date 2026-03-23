import hashlib

import jax
import jax.numpy as jnp
from flax import nnx

class EmbeddingLayer (nnx.Module) :
    
    def __init__(self, embed_dim: int, rngs: nnx.Rngs) -> None:
        self.embed_dim = embed_dim
        self.linear = nnx.Linear(2 * embed_dim, embed_dim, rngs=rngs)
        
        self.base_key = rngs.params()
        
    def combine(self, x: jax.Array, y: jax.Array) -> jax.Array :
        return self.linear(jnp.concatenate([x, y], axis=-1))

    def embed_variable(self, name: str, rngs: nnx.Rngs) -> jax.Array :
        
        digest = hashlib.sha256(name.encode()).digest()
        name_int = int.from_bytes(digest[:4], "big")
        key = jax.random.fold_in(self.base_key, name_int)
        
        vec = jax.random.normal(key, shape=(self.embed_dim,))
        return vec / jnp.linalg.norm(vec)