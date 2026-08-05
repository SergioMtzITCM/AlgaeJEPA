from trainer import KD_Trainer
from utils.data import get_distillation_dataloaders
from utils.loader import load_pretrain_encoder
from configs.config import BaseConfig, MicroViTConfig
from models.vit_core import ViTModel
from models.microvit_core import MicroViTModel

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Student Model Configuration

    # MicroViT
    microvit_student_config = MicroViTConfig(
        model = "S1",
        image_size = 224,
        num_channels = 3,
        ayerscale_value = 1e-5
    )

    # ViT
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

    # Make Student Model

    # MicroViT
    microvit_student_model = MicroViTModel(microvit_student_config)
    microvit_student_model = torch.compile(microvit_student_model)

    # ViT
    #vit_student_model = ViTModel(vit_student_config)
    #vit_student_model = torch.compile(vit_student_model)

    # Load Teacher Encoder
    teacher_model = load_pretrain_encoder(
        checkpoint_path = "./Pretrain_Outputs/checkpoints/best_model.pth",
        device = device
    )

    # Get Distillation Dataloaders
    train_dataloader, test_dataloader = get_distillation_dataloaders(
        pretrain_dir = "./Preprocessed_data/",
        test_dir = "./Test_data_preprocessed/",
        test_csv = "./Test_data_preprocessed/classes.csv",
        target_size = 224,
        batch_size = 64,
        shuffle_train = True,
        num_workers = 8
    )

    EPOCHS = 30
    BASE_LR = 2.5e-4
    BETA_1 = 0.9
    BETA_2 = 0.95
    WEIGHT_DECAY = 0.02
    MIN_LR = 1e-5
    WARMUP_EPOCHS = 5
    START_FACTOR = 0.15

    # Optimizer and LRScheduler
    optimizer = AdamW(microvit_student_model.parameters(), lr = BASE_LR, betas = (BETA_1,
                                                                        BETA_2), weight_decay = WEIGHT_DECAY)
    warmup = LinearLR(optimizer, start_factor = START_FACTOR, total_iters = WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(optimizer, T_max = (EPOCHS - WARMUP_EPOCHS), eta_min = MIN_LR)
    scheduler = SequentialLR(optimizer, schedulers = [warmup, cosine], milestones = [5])

    # Trainer
    trainer = KD_Trainer(
        teacher = teacher_model,
        student = microvit_student_model,
        train_dataloader = train_dataloader,
        test_dataloader = test_dataloader,
        optimizer = optimizer,
        lr_scheduler = scheduler,
        device = device,
        epochs = EPOCHS,
        loss_type = "mse",
        save_dir = "MicroViTS1_Student"
    )

    trainer.train()

if __name__ == "__main__":
    main()
