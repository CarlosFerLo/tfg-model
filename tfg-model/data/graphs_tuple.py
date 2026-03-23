from typing import NamedTuple
import jax

class GraphsTuple(NamedTuple):
    nodes: jax.Array
    senders: jax.Array
    receivers: jax.Array