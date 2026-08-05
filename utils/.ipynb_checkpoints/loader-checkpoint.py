import torch
from models.vit_core import ViTModel
from models.microvit_core import MicroViTModel

def load_pretrain_encoder(checkpoint_path: str, device: torch.device) -> ViTModel:

    # Load the Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location = "cpu", weights_only = False)
    full_state_dict = checkpoint["model_state_dict"]
    config = checkpoint["config"]
    config.num_hidden_layers = 12

    
    # Filter state_dict to Obtain Only the Encoder Weights
    encoder_state_dict = {}
    for key, value in full_state_dict.items():
        key = key.replace("_orig_mod.", "")
        
        if key.startswith("encoder."):
            new_key = key.replace("encoder.", "")
            encoder_state_dict[new_key] = value

    # Create Base Model
    encoder = ViTModel(config)

    # Load the State Dict
    msg = encoder.load_state_dict(encoder_state_dict, strict = True)
    print(f"Encoder Load State: {msg}")

    return encoder.to(device)


def load_student_model(checkpoint_path: str, device: torch.device) -> ViTModel:

    # Load the Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location = "cpu", weights_only = False)
    full_state_dict = checkpoint["student_state_dict"]
    config = checkpoint["config"]

    # Create Base Model
    student = ViTModel(config)

    # Load the State Dict
    msg = student.load_state_dict(full_state_dict, strict = True)
    print(f"Student Load State: {msg}")

    return student.to(device)