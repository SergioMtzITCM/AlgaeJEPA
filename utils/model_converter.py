import os
import subprocess
import torch
from pathlib import Path

def convert_to_onnx_and_rknn(model, output_path: str, test_data_path: str, 
                             do_quantization: bool = False):

    model.eval()
    model = model.cpu()

    

    # Make output dir
    OUT_PATH = Path(output_path)
    OUT_PATH.mkdir(parents = True, exist_ok = True)

    base_name = OUT_PATH.name

    ONNX_PATH = OUT_PATH / f"{base_name}.onnx"
    RKNN_PATH = OUT_PATH / f"{base_name}.rknn"

    num_channels = model.config.num_channels
    image_size = model.config.image_size
    input_shape = (num_channels, image_size, image_size)

    OP_SET_VERSION = 14

    print("Exporting model to ONNX...")

    dummy_input = torch.randn((1, *input_shape))

    torch.onnx.export(
            model,
            dummy_input,
            str(ONNX_PATH),
            export_params = True,
            opset_version = OP_SET_VERSION,
            do_constant_folding = True,
            input_names = ["input"],
            output_names = ["output"],
    )

    print("Model conversion has been completed succesfully...")
    print()

    rknn_script = f"""
    
import os
import sys
from rknn.api import RKNN

onnx_model = '{ONNX_PATH}'
rknn_output = '{RKNN_PATH}'

print("Exporting ONNX model to RKNN...")
rknn = RKNN(verbose = True)

# Model config
rknn.config(
    mean_values = [[0, 0, 0]],
    std_values = [[1, 1, 1]],
    target_platform = 'rk3588'
)

# Load ONNX model
print("Loading ONNX model...")
ret = rknn.load_onnx(model = onnx_model, input_size_list = [list({input_shape})])
if ret != 0:
    raise RuntimeError(f"Error when loading ONNX (code: {{ret}})")

# Make the RKNN model
print("Building RKNN model...")
ret = rknn.build(do_quantization = {do_quantization})
if ret != 0:
    raise RuntimeError(f"Error when building RKNN (code: {{ret}})")

# Export .rknn file
print("Exporting RKNN model...")
ret = rknn.export_rknn(rknn_output)
if ret != 0:
    raise RuntimeError(f"Error when exporting RKNN (code: {{ret}})")

rknn.release()
print(f"RKNN model saved in: {{rknn_output}}")

"""

    temp_script_path = OUT_PATH / "temp_rknn_export.py"
    with open(temp_script_path, "w") as f:
        f.write(rknn_script)

    try:
        result = subprocess.run(
            ["uv", "run", "--python", "rknn_env/bin/python", str(temp_script_path)],
            check = True,
            capture_output = False,
            text = True
        )

        print("Model conversion to RKNN has been completed succesfully...")

    except subprocess.CalledProcessError as e:
        print(f"\nERROR: RKNN conversion has failed with code {e.returncode}.")
        raise

    except FileNotFoundError:
        print(f"\nERROR: 'uv' was not found.")

    finally:
        if temp_script_path.exists():
            temp_script_path.unlink()
        print()


def convert_onnx_to_tflite(onnx_file_path: str, 
                           test_data_path: str,
                           output_dir: str,
                           quant_type: str = "per-channel"):
 
    OUT_PATH = Path(output_dir)
    OUT_PATH.mkdir(parents = True, exist_ok = True)
    base_name = OUT_PATH.name

    onnx_path = Path(onnx_file_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file {onnx_path} has not found.")
 
    print("*** Exporting ONNX model to TFLite...")

    worker_script = Path(__file__).resolve().parent / "tflite_export_worker.py"
    if not worker_script.exists():
        raise FileNotFoundError(
            f"Auxiliar script '{worker_script}' has not found. It must to be in the same directory."
        )

    try:
        subprocess.run(
            [
                "uv", "run", "--python", "tf_env/bin/python", str(worker_script),
                "--onnx-path", str(onnx_path),
                "--test-data-path", str(test_data_path),
                "--output-dir", str(OUT_PATH),
                "--output-name", base_name,
                "--quant-type", quant_type,
            ],
            check = True,
            capture_output = False,
            text = True

        )
        print("Model Conversion to TFLite has been completed succesfully...")

    except subprocess.CalledProcessError as e:
        print(f"\nERROR: TFLite conversion has failed with code {e.returncode}")
        raise

    except FileNotFoundError:
        print(f"\n ERROR: 'uv' was not found.")

    finally:
        print()
