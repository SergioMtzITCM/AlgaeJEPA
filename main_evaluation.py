from utils.evaluation import evaluate_model_knn
from utils.loader import load_student_model, load_pretrain_encoder
from utils.data import get_knn_dataloaders

import torch
from torchsummary import summary

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    h5_path_test = "./Test_data_preprocessed/Test_data_preprocessed.h5"
    test_csv = "./Test_data_preprocessed/classes.csv"


    num_shots = 10
    model_role = "student" # 'teacher' or 'student'
    student_arch = "MicroViT" # Valid Options: 'ViT', 'MicroViT', 'MobileNet'
    checkpoint_path = "./MicroViTS3_Student/checkpoints/best_student.pth"

    print(f"Initiating Evaluation...")

    if model_role == "teacher":
        model = load_pretrain_encoder(checkpoint_path, device)
        mode = "test"
        target_size = None

    elif model_role == "student":
        model = load_student_model(checkpoint_path, student_arch, device)
        mode = "test_distillation"
        target_size = model.config.image_size if hasattr(model, "config") and hasattr(model.config, "image_size") else None
    else:
        raise ValueError("'model_role' must be 'teacher' or 'student'")

    #print(summary(model))

    # Get Dataloaders
    train_dataloader, test_dataloader = get_knn_dataloaders(
        h5_path_test = h5_path_test,
        test_csv = test_csv,
        num_shots = num_shots,
        mode = mode,
        target_size = target_size,
        batch_size = 32
    )

    # Evaluate using knn
    n_neighbors = 5
    n_neighbors = min(5, n_neighbors)
    
    evaluate_model_knn(
        model = model,
        train_dataloader = train_dataloader,
        test_dataloader = test_dataloader,
        device = device,
        n_neighbors = n_neighbors,
        metric = "cosine"
    )

if __name__ == "__main__":
    main()
