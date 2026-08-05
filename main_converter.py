import torch
from utils.loader import load_pretrain_encoder, load_student_model
from utils.model_converter import convert_to_onnx_and_rknn, convert_onnx_to_tflite

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_role = "student" # 'teacher' or 'student'
    student_arch = "MobileNet" # Valid Options: 'ViT', 'MicroViT', 'MobileNet'
    checkpoint_path = "./MobileNetV2_Student/checkpoints/best_student.pth"
    output_path = "./MobileNetV2_Student/"
    test_data_path = "./Test_data_preprocessed/Test_data_preprocessed.h5"

    if model_role == "teacher":
        model = load_pretrain_encoder(checkpoint_path, device)

    elif model_role == "student":
        model = load_student_model(checkpoint_path, student_arch, device)

    else:
        raise ValueError("'model_role' must be 'teacher' or 'student'")

    convert_to_onnx_and_rknn(
        model = model,
        output_path = output_path,
        test_data_path = test_data_path,
        do_quantization = False
    )

    if student_arch == "MobileNet":
        convert_onnx_to_tflite(
                onnx_file_path = "./MobileNetV2_Student/MobileNetV2_Student.onnx",
                test_data_path = test_data_path,
                output_dir = output_path

        )

if __name__ == "__main__":
    main()



