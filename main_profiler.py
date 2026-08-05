import torch
from utils.loader import load_student_model, load_pretrain_encoder
from utils.model_profiler import profile_model

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_role = "student" # 'teacher' or 'student'
    student_arch = "MicroViT" # Valid Options: 'ViT', 'MicroViT', 'MobileNet'
    checkpoint_path = "./MicroViTS3_Student/checkpoints/best_student.pth"

    if model_role == "teacher":
        model = load_pretrain_encoder(checkpoint_path, device)

    elif model_role == "student":
        model = load_student_model(checkpoint_path, student_arch, device)

    else:
        raise ValueError("'model_role' must be 'teacher' or 'student'")

    profile_model(model = model, device = device)


if __name__ == "__main__":
    main()
