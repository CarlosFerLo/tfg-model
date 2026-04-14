from typing import NamedTuple, List
import jax

class GraphsTuple(NamedTuple):
    nodes: jax.Array
    senders: jax.Array
    receivers: jax.Array
    
    uid_mapping: List[str]