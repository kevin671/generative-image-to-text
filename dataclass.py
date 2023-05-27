from dataclasses import dataclass

@dataclass
class ModelArgs:
    image_encoder_type: str = "RN50"
    language_model: str = "facebook/opt-2.7b"
    mapping_type: str = "linearLn"
    input_resolution: int = 224
    output_grid: bool = True

@dataclass
class DistArgs:
    distributed: bool = False
    world_size: int = 0
    rank: int = 0
    local_rank: int = 0
    device: str = None
    dist_backend: str = "nccl"
    dist_url: str = "env://"