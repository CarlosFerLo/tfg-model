import jax 
from flax import nnx
from jraphx.nn.conv import GCNConv

class ModelBody(nnx.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, N: int, dropout: float, rngs: nnx.Rngs):
        self.input_layer = GCNConv(in_features, hidden_features, rngs=rngs)
        self.output_layer = GCNConv(hidden_features, out_features, rngs=rngs)
        
        self.N = N
        self.hidden_layers = [
            GCNConv(hidden_features, hidden_features, rngs=rngs)
            for _ in range(N)
        ]
        
        self.dropout = nnx.Dropout(dropout, rngs=rngs)
        
    def __call__(self, x, edge_index, deterministic: bool = False):
        # Process input layer
        x = self.input_layer(x, edge_index)
        
        x = jax.nn.relu(x)
        x = self.dropout(x, deterministic=deterministic)
        
        # Process hidden layers
        for hidden_layer in self.hidden_layers :
            residual = x
            x = hidden_layer(x, edge_index)
            x = x + residual
            x = jax.nn.relu(x)
            x = self.dropout(x, deterministic=deterministic)
        
        # Process output layer 
        x = self.output_layer(x, edge_index)
        
        return x