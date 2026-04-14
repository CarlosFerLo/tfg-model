from typing import Dict, List

import jax
import jax.numpy as jnp
from flax import nnx

import implica

from .data import GraphsTuple
from .modules import Model

def _embed_type(t: implica.Type, cache: Dict[str, jax.Array], model: Model, rngs: nnx.Rngs) -> jax.Array :
    
    cached = cache.get(t.uid())
    if cached is not None:
        return cached
    
    name = t.as_var() 
    pair = t.as_arrow()
    
    if name is not None :
        out = model.embed.embed_variable(name, rngs=rngs)
        cache[t.uid()] = out
        return out
    elif pair is not None :
        left = _embed_type(pair[0], cache, model, rngs=rngs)
        right = _embed_type(pair[1], cache, model, rngs=rngs)
        out = model.embed.combine(left, right)
        cache[t.uid()] = out
        return out
    else :
        raise RuntimeError("Every type should be either a variable or an arrow, hence this should never happen.")
     
def _map_uid_to_int(uids: List[str], uid_mapping: List[str]) -> List[int] :
    def _uid_to_int(uid: str) -> int :
        return uid_mapping.index(uid)
        
    return list(map(_uid_to_int, uids))

def graph_to_tuple(graph: implica.Graph, model: Model, rngs: nnx.Rngs) -> GraphsTuple :
    
    embedding_cache = {}
    nodes = graph.nodes() 
    
    # Get node uid_mapping and embeddings
    uid_mapping, node_embeddings = zip(*[
        (n.uid(), _embed_type(n.type(), embedding_cache, model, rngs=rngs))
        for n in nodes
    ])

    
    edges = graph.edges()
    senders_uid, receivers_uid = zip(*[
        e.uid()
        for e in edges
    ])
    
    senders = _map_uid_to_int(list(senders_uid), list(uid_mapping))
    receivers = _map_uid_to_int(list(receivers_uid), list(uid_mapping))
    
    return GraphsTuple(
        nodes=jnp.array(node_embeddings),
        senders=jnp.array(senders),
        receivers=jnp.array(receivers),
        uid_mapping=list(uid_mapping)
    )

if __name__ == "__main__" :
    
    model = Model(10, 10, 10, 10, 10, 2, rngs=nnx.Rngs(1))
    
    graph = implica.Graph(constants=[implica.Constant("f", "A -> B")])
    
    graph.query().create("(:A)").create("(:B)").create("()-[::@f()]->()").execute()
    
    tuple = graph_to_tuple(graph, model, nnx.Rngs(1))
    print(tuple)
    
    