from trainer import KD_Trainer, build_kd_projector
from utils.data import get_distillation_dataloaders
from utils.loader import load_pretrain_encoder

from configs.config import BaseConfig, MicroViTConfig, MobileNetConfig, ResNetConfig

from models.vit_core import ViTModel
from models.microvit_core import MicroViTModel
from models.mobilenet_core import MobileNetModel
from models.resnet_core import ResNetModel


import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Student Model Configuration

    # MicroViT
    microvit_student_config = MicroViTConfig(
        model = "S3",
        image_size = 224,
        num_channels = 3,
        layerscale_value = 1e-5
    )

    # ViT-Tiny
    vit_student_config = BaseConfig(
        hidden_size = 192,
        num_hidden_layers = 12,
        num_attention_heads = 3,
        intermediate_size = 768,
        hidden_dropout_prob = 0.1,
        attention_probs_dropout_prob = 0.1,
        image_size = 224,
        patch_size = 16,
        num_channels = 3
    )

    # MobileNet
    mobilenet_student_config = MobileNetConfig(
        model = "V2",
        image_size = 224,
        num_channels = 3
    )

    # ResNet
    resnet_student_config = ResNetConfig(
        image_size = 224,
        num_channels = 3
    )

    teacher_model = load_pretrain_encoder(
            checkpoint_path = "AlgaeJEPA_Pretrain_lambda01/checkpoints/best_model.pth",
            device = device
    )

    # Make Student Model

    # MicroViT
    #student_model = MicroViTModel(microvit_student_config)

    # ViT-Tiny
    student_model = ViTModel(vit_student_config)
    
    # MobileNet
    #student_model = MobileNetModel(mobilenet_student_config)
    
    # ResNet
    #student_model = ResNetModel(resnet_student_config)

    # Move to Device
    student_model = student_model.to(device)

    # Channel Projector (Student -> Teacher): 1x1 Conv2d, or Identity if both have the same channels.
    # It is built here, before the optimizer and the scheduler, so that its parameters are optimized
    # and managed by the LR scheduler like the rest of the parameters.
    projector = build_kd_projector(teacher_model, student_model).to(device)

    # Compile the Model
    student_model = torch.compile(student_model)


    EPOCHS = 30
    BASE_LR = 7.5e-4 # ViT: 7.5e-4, CNN: 1e-3, MicroViT: 7.5e-4
    BETA_1 = 0.9 # ViT: 0.9, CNN: 0.9, MicroViT: 0.9
    BETA_2 = 0.95 # ViT: 0.95, CNN: 0.999, MicroViT: 0.999
    WEIGHT_DECAY = 0.05 # ViT 0.05, CNN: 1e-4, MicroViT: 0.01
    MIN_LR = 1e-5
    WARMUP_EPOCHS = 7 # ViT: 5-7, CNN: 2-3, MicroViT: 5
    START_FACTOR = 0.1 # ViT: 0.1, CNN: 0.2, MicroViT: 0.15

    BATCH_SIZE = 192
    NUM_WORKERS = 8
    PREFETCH_FACTOR = 3


    # Get Distillation Dataloaders
    train_dataloader, test_dataloader = get_distillation_dataloaders(
        h5_path_pretrain = "./Pretrain_Data/Pretrain_data.h5",
        h5_path_test = "./Out_Distribution_Dataset/ODD.h5",
        test_csv = "./Out_Distribution_Dataset/ODD_labels.csv",
        batch_size = BATCH_SIZE,
        shuffle_train = True,
        num_workers = NUM_WORKERS,
        prefetch_factor = PREFETCH_FACTOR
    )

    # Optimizer: student AND projector parameters in the same optimizer, from the start
    optimizer = AdamW(
        list(student_model.parameters()) + list(projector.parameters()),
        lr = BASE_LR,
        betas = (BETA_1, BETA_2),
        weight_decay = WEIGHT_DECAY
    )

    # LRScheduler: created AFTER the optimizer already contains every parameter
    warmup = LinearLR(optimizer, start_factor = START_FACTOR, total_iters = WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(optimizer, T_max = (EPOCHS - WARMUP_EPOCHS), eta_min = MIN_LR)
    scheduler = SequentialLR(optimizer, schedulers = [warmup, cosine], milestones = [WARMUP_EPOCHS])

    # Trainer
    trainer = KD_Trainer(
        teacher = teacher_model,
        student = student_model,
        train_dataloader = train_dataloader,
        test_dataloader = test_dataloader,
        optimizer = optimizer,
        lr_scheduler = scheduler,
        device = device,
        epochs = EPOCHS,
        loss_type = "mse",
        save_dir = "ViT-Tiny_Student",
        projector = projector
    )

    trainer.train()

if __name__ == "__main__":
    main()
