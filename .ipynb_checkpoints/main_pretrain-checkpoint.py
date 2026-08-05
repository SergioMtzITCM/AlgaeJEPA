from trainer import SIGReg_IJEPA_Trainer
from models.i_jepa import SIGReg_IJEPA
from utils.data import get_pretrain_dataloaders
from configs.config import BaseConfig

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Base Config
    config = BaseConfig(
        hidden_size = 384,
        num_hidden_layers = 12,
        num_attention_heads = 6,
        intermediate_size = 1536,
        hidden_dropout_prob = 0.1,
        attention_probs_dropout_prob = 0.1,
        image_size = 224,
        patch_size = 16,
        num_channels = 3,
        prediction_ratio = 0.65
    )

    
    # IJEPA Model
    i_jepa_model = SIGReg_IJEPA(
        config = config,
        loss_type = "mse",
        lambda_sigreg = 0.1
    ).to(device)
    i_jepa_model = torch.compile(i_jepa_model)

    # Dataloaders
    pretrain_dataloader, test_dataloader = get_pretrain_dataloaders(
        pretrain_dir = "./Preprocessed_data/",
        test_dir = "./Test_data_preprocessed/",
        test_csv = "./Test_data_preprocessed/classes.csv",
        batch_size = 192,
        shuffle_train = True,
        num_workers = 8
    )

    EPOCHS = 50
    BASE_LR = 2.5e-4
    BETA_1 = 0.9
    BETA_2 = 0.95
    WEIGHT_DECAY = 0.02
    MIN_LR = 1e-5
    WARMUP_EPOCHS = 5
    START_FACTOR = 0.15
    

    # Optimizer and LRScheduler
    optimizer = AdamW(i_jepa_model.parameters(), lr = BASE_LR, betas = (BETA_1,
                                                                        BETA_2), weight_decay = WEIGHT_DECAY)
    warmup = LinearLR(optimizer, start_factor = START_FACTOR, total_iters = WARMUP_EPOCHS)
    cosine = CosineAnnealingLR(optimizer, T_max = (EPOCHS - WARMUP_EPOCHS), eta_min = MIN_LR)
    scheduler = SequentialLR(optimizer, schedulers = [warmup, cosine], milestones = [5])

    # Trainer
    trainer = SIGReg_IJEPA_Trainer(
        model = i_jepa_model,
        train_dataloader = pretrain_dataloader,
        test_dataloader = test_dataloader,
        optimizer = optimizer,
        lr_scheduler = scheduler,
        device = device,
        epochs = EPOCHS,
        save_dir = "Main_Pretrain_lambda01"
    )

    trainer.train()

if __name__ == "__main__":
    main()
