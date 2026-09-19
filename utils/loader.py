import torch
from models.vit_core import ViTModel
from models.microvit_core import MicroViTModel
from models.mobilenet_core import MobileNetModel
from models.resnet_core import ResNetModel

from typing import Union

def load_pretrain_encoder(checkpoint_path: str, device: torch.device, num_classes: int = None) -> ViTModel:

    # Load the Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location = "cpu", weights_only = False)
    full_state_dict = checkpoint["model_state_dict"]
    config = checkpoint["config"]

    if num_classes is not None:
        config.num_classes = num_classes
    
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
    msg = encoder.load_state_dict(encoder_state_dict, strict = False)
    print(f"Encoder Load State: {msg}")

    return encoder.to(device)


def load_student_model(
                       checkpoint_path: str,
                       model_type: str, 
                       device: torch.device,
                       num_classes: int = None) -> Union[ViTModel, MicroViTModel, MobileNetModel, ResNetModel]:

    # Load the Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location = "cpu", weights_only = False)
    full_state_dict = checkpoint["student_state_dict"]
    config = checkpoint["config"]

    if num_classes is not None:
        config.num_classes = num_classes

    # Clean the state_dict to remove the torch.compile prefix '_orig_mod.'
    clean_state_dict = {}
    for key, value in full_state_dict.items():
        clean_key = key.replace("_orig_mod.", "")
        clean_state_dict[clean_key] = value

    # Create Base Model dynamically based on the requested architecture
    if model_type.lower() == "vit":
        student = ViTModel(config)
    elif model_type.lower() == "microvit":
        student = MicroViTModel(config)
    elif model_type.lower() == "mobilenet":
        student = MobileNetModel(config)
    else:
        raise ValueError(
            f"'{model_type}' is not supported."
            f"Allowed values: 'ViT', 'MicroViT', 'MobileNet'.")

    # Load the State Dict
    msg = student.load_state_dict(clean_state_dict, strict = False)
    print(f"Student Load State: {msg}")

    return student.to(device)
