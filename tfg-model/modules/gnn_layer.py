import jax
import jax.numpy as jnp
from flax import nnx

class GNNLayer(nnx.Module) :
    
    def __init__(self, in_dim: int, out_dim: int, rngs: nnx.Rngs, activation=nnx.relu) -> None:
        self.activation = activation
        
        self.self_proj = nnx.Linear(in_dim, out_dim, rngs=rngs)
        self.in_proj = nnx.Linear(in_dim, out_dim, rngs=rngs)
        self.out_proj = nnx.Linear(in_dim, out_dim, rngs=rngs)
        
        self.combine = nnx.Linear(3 * out_dim, out_dim, rngs=rngs)
        
    def _mean_aggregate(
        self,
        messages: jax.Array,
        targets: jax.Array,
        N: int
    ) -> jax.Array:
        counts = jnp.zeros(N).at[targets].add(1)
        agg = jnp.zeros((N, messages.shape[-1])).at[targets].add(messages)
        safe = jnp.where(counts > 0, counts, 1.0)
        return agg / safe[:, None]
    
    def __call__(self, x: jax.Array, senders: jax.Array, receivers: jax.Array) -> jax.Array :
        N = x.shape[0]
        
        self_out = self.self_proj(x)
        
        in_msgs = self.in_proj(x[senders])
        in_agg = self._mean_aggregate(in_msgs, receivers, N)
        
        out_msgs = self.out_proj(x[receivers])
        out_agg = self._mean_aggregate(out_msgs, senders, N)
        
        combined = jnp.concatenate([self_out, in_agg, out_agg], axis=-1)
        return self.activation(self.combine(combined))