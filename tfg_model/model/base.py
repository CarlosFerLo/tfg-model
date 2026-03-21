from flax import nnx

from .body import ModelBody
from .conjecturer import ConjecturerHead
from .prover import ProverHead

class Model (nnx.Module) :
    
    def __init__ (self, body: ModelBody, conjecturer: ConjecturerHead, prover: ProverHead) :
        self.body = body
        self.conjecturer = conjecturer
        self.prover = prover
        
    def conjecturer_forward(self, x, edge_index, deterministic: bool = False) :
        x = self.body(x, edge_index, deterministic=deterministic)
        x = self.conjecturer(x)
        return x
    
    def prover_forward(self, x, edge_index, deterministic: bool = False) :
        x = self.body(x, edge_index, deterministic=deterministic)
        x = self.prover(x)
        return x