import flax.nnx as nnx

class ProverHead(nnx.Module) :
    def __init__(self, in_features: int, out_classes: int, rngs: nnx.Rngs):
        self.projection = nnx.Linear(in_features, out_classes, rngs=rngs)
        
    def __call__(self, x):
        return self.projection(x)