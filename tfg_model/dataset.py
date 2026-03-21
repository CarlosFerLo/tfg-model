from typing import List
from pathlib import Path
import logging

import hydra

import jax
import jax.numpy as jnp
import numpy as np
import jraphx

from .engine import GraphEngine

log = logging.getLogger(__name__)

_GRAPH_FIELDS = ("x", "edge_index", "edge_attr", "y", "pos", "batch", "ptr")


class DatasetExample:
    graph: jraphx.Data
    labels: jax.Array

    def __init__(self, graph: jraphx.Data, labels: jax.Array) -> None:
        self.graph = graph
        self.labels = labels

    # -- serialization --------------------------------------------------------

    def serialize(self, path: Path) -> None:
        """Serialize this example to a single ``.npz`` file at *path*."""
        arrays: dict[str, np.ndarray] = {}
        for field in _GRAPH_FIELDS:
            value = getattr(self.graph, field, None)
            if value is not None:
                arrays[f"graph_{field}"] = np.asarray(value)
        arrays["labels"] = np.asarray(self.labels)
        np.savez(str(path), **arrays)  # type: ignore[arg-type]
        log.debug("Saved example to %s", path)

    @classmethod
    def deserialize(cls, path: Path) -> "DatasetExample":
        """Load an example from a ``.npz`` file at *path*."""
        data = np.load(path)
        graph_kwargs: dict[str, jnp.ndarray] = {}
        for field in _GRAPH_FIELDS:
            key = f"graph_{field}"
            if key in data:
                graph_kwargs[field] = jnp.array(data[key])
        graph = jraphx.Data(**graph_kwargs)
        labels = jnp.array(data["labels"])
        log.debug("Loaded example from %s", path)
        return cls(graph=graph, labels=labels)


class Dataset:

    examples: List[DatasetExample]

    def __init__(self, examples: List[DatasetExample]) -> None:
        self.examples = examples

    def save(self, directory: Path) -> None:
        """Save every example as ``<directory>/<index>.npz``."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for idx, example in enumerate(self.examples):
            example.serialize(directory / f"{idx}.npz")
        log.info("Saved dataset with %d examples to %s", len(self.examples), directory)

    @classmethod
    def load(cls, directory: Path) -> "Dataset":
        """Load all ``.npz`` files from *directory* (sorted by index)."""
        directory = Path(directory)
        files = sorted(directory.glob("*.npz"), key=lambda p: int(p.stem))
        examples = [DatasetExample.deserialize(f) for f in files]
        log.info("Loaded dataset with %d examples from %s", len(examples), directory)
        return cls(examples=examples)
   
@hydra.main(version_base=None, config_path="conf", config_name="config")
def generate_dataset(cfg) -> None :
    log.info("Generating dataset...")
    variables: List[str] = cfg.dataset.variables
    conjectures: List[str] = cfg.dataset.conjectures
    
    log.debug(f"Variables: {', '.join(variables)}")
    conjectures_str = '\n - '.join(conjectures)
    log.debug(f"Conjectures:\n - {conjectures_str}")
        

if __name__ == "__main__":
    log.setLevel(logging.DEBUG)
    generate_dataset()
        
        
        
         