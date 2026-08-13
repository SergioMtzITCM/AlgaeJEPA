import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 
from utils.loader import load_student_model, load_pretrain_encoder
from utils.model_profiler import profile_model

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--model-role", required = True, type = str,
                         choices = ["teacher", "student"])
    parser.add_argument("--student-arch", default = None, type = str,
                         help = "Required when --model-role=student. Valid "
                              "options: 'ViT', 'MicroViT', 'MobileNet'.")
    parser.add_argument("--checkpoint-path", required = True, type = str)
    parser.add_argument("--output-dir", required = True, type = str)
    parser.add_argument("--output-name", required = True, type = str)
    parser.add_argument("--device", default = "cuda", type = str,
                         choices = ["cuda", "cpu"])
    parser.add_argument("--num-warmup", default = 10, type = int)
    parser.add_argument("--num-runs", default = 100, type = int)
    return parser.parse_args()


def main() -> None:
 
    args = parse_args()
 
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} has not been found.")
 
    requested_device = args.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("[WARN] --device=cuda was requested but CUDA is not available "
              "in this process/environment; falling back to CPU.")
        requested_device = "cpu"
    device = torch.device(requested_device)

    print(f"** 1. Loading '{args.model_role}' model from checkpoint...")
    if args.model_role == "teacher":
        model = load_pretrain_encoder(str(checkpoint_path), device)
    elif args.model_role == "student":
        if not args.student_arch:
            raise ValueError("--student-arch is required when --model-role=student")
        model = load_student_model(str(checkpoint_path), args.student_arch, device)
    else:
        raise ValueError("--model-role must be 'teacher' or 'student'")

    print("** 2. Running torch/ptflops profiling...")
    profile_model(
        model = model,
        device = device,
        output_dir = args.output_dir,
        output_name = args.output_name,
        num_warmup = args.num_warmup,
        num_runs = args.num_runs,
    )

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file = sys.stderr)
        sys.exit(1)
