from dataclasses import dataclass

@dataclass
class Files:
    dataset_path : str
    dataset_cache : str
    model_checkpoint : str

@dataclass
class Params:
    model : str
    max_history: int
    no_sample : bool
    max_length: int
    device: str
    min_length: int
    seed: int
    temperature: float
    top_k: int
    top_p: float

@dataclass
class ChatConfig:
    files: Files
    params: Params
