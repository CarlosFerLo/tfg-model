from typing import Dict

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
        

def graph_to_tuple(graph: implica.Graph, model: Model, rngs: nnx.Rngs) -> GraphsTuple :
    
    embedding_cache = {}
    nodes = graph.nodes() 
    
    uid_mapping, node_embeddings = zip(*[
        (n.uid(), _embed_type(n.type(), embedding_cache, model, rngs=rngs))
        for n in nodes
    ])
    
    print(uid_mapping)
    print(node_embeddings)
    
    return GraphsTuple(jnp.zeros(0), jnp.zeros(0), jnp.zeros(0))

if __name__ == "__main__" :
    
    model = Model(10, 10, 10, 10, 10, 2, rngs=nnx.Rngs(1))
    
    print(model)
    
    graph = implica.Graph(constants=[implica.Constant("f", "A -> B")])
    
    print(graph.nodes())
    
    graph.query().create("(:A)").create("(:B)").create("()-[::@f()]->()").execute()
    
    graph_to_tuple(graph, model, nnx.Rngs(1))
    
    