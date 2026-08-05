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
                           output_dir: str):
 
    OUT_PATH = Path(output_dir)
    OUT_PATH.mkdir(parents = True, exist_ok = True)
    base_name = OUT_PATH.name
 
    print("*** Exporting ONNX model to TFLite...")
 
    tflite_script = f"""
import os
import tensorflow as tf
from onnx2tf import convert
import h5py
import numpy as np
from pathlib import Path
import shutil
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.tflm_converter import convert_tflite_to_tflm
 
onnx_path = Path('{onnx_file_path}')
if not onnx_path.exists():
    raise FileNotFoundError(f"ONNX File {{onnx_path}} has not found.")
 
OUT_DIR = Path('{OUT_PATH}')
saved_model_path = OUT_DIR / "saved_model"
 
print("** 1. Generating TFLite (Float32) for inspection with onnx2tf...")
try:
    convert(
        input_onnx_file_path = str(onnx_path),
        output_folder_path = str(saved_model_path),
        output_signaturedefs = True,
        non_verbose = False,
        disable_strict_mode = True,
        tflite_backend = "tf_converter"
    )
except Exception as e:
    raise RuntimeError(f"Error in onnx2tf (Phase 1): {{e}}") from e
 
tflite_files = list(saved_model_path.glob("*.tflite"))
float_tflite_files = [f for f in tflite_files if 'int8' not in f.name and 'quant' not in f.name]
 
if not float_tflite_files:
    raise FileNotFoundError("Base .tflite model was not found in the saved_model directory.")
 
float32_tflite_path = float_tflite_files[0]
print(f"* Inspecting generated TFLite model: {{float32_tflite_path.name}}...")
 
interpreter = tf.lite.Interpreter(model_path=str(float32_tflite_path))
interpreter.allocate_tensors()
 
input_details = interpreter.get_input_details()[0]
input_key = input_details['name']
model_shape = input_details['shape'].tolist()
model_dtype = input_details['dtype']
 
if len(model_shape) >= 4:
    expected_spatial_dims = model_shape[1:]
else:
    raise ValueError(f"Expected input shape: {{model_shape}}")
 
print(f"Input Name: {{input_key}}, Expected Shape: {{model_shape}}, Dtype: {{model_dtype}}")
 
if len(expected_spatial_dims) == 3:
    if expected_spatial_dims[0] in [1, 3]:
        model_format = 'NCHW'
        channels = expected_spatial_dims[0]
    elif expected_spatial_dims[0] > 3 and expected_spatial_dims[1] > 3 and expected_spatial_dims[2] in [1, 3]:
        model_format = 'NHWC'
        channels = expected_spatial_dims[2]
    else:
        model_format = 'UNKNOWN'
        channels = 3
else:
    model_format = 'UNKNOWN'
    channels = 3
 
print(f"Model's input format: {{model_format}}")
 
print(f"* Loading calibration data from: {test_data_path}...")
h5_file = h5py.File('{test_data_path}', "r")
data_shape = h5_file['images'].shape
num_images = data_shape[0]
 
data_for_calib = h5_file['images'][:]
h5_file.close()
 
if model_format == 'NCHW' and data_for_calib.shape[1:] != tuple(expected_spatial_dims):
    print("Transposing the data from 'NHWC' to 'NCHW'.")
    data_for_calib = np.transpose(data_for_calib, (0, 3, 1, 2))
elif model_format == 'NHWC' and data_for_calib.shape[1:] != tuple(expected_spatial_dims):
    print("Transposing the data from 'NCHW' to 'NHWC'.")
    data_for_calib = np.transpose(data_for_calib, (0, 2, 3, 1))
else:
    print("Data format matches the model's format.")
 
calib_data_path = OUT_DIR / "calib_data.npy"
sample_limit = min(num_images, 150)
np.save(calib_data_path, data_for_calib[:sample_limit].astype(np.float32))
print(f"Calibration data saved in: {{calib_data_path}}")

# Dynamically calculating Mean and Std arrays based on channel count
mean_vals = [0.0] * channels
std_vals = [1.0] * channels
 
print("** 2. Applying native INT8 quantization with onnx2tf...")
try:
    convert(
        input_onnx_file_path = str(onnx_path),
        output_folder_path = str(saved_model_path),
        output_signaturedefs = True,
        non_verbose = False,
        disable_strict_mode = True,
        tflite_backend = "tf_converter",
        output_integer_quantized_tflite = True,
        quant_type = "per-tensor",
        # Passing [input_name, numpy_file_path, mean, std] as strictly required by flatbuffer_direct
        custom_input_op_name_np_data_path = [['input', str(calib_data_path), mean_vals, std_vals]]
    )
except Exception as e:
    raise RuntimeError(f"Error in onnx2tf (Quantization Phase): {{e}}") from e
 
int8_tflite_path = list(saved_model_path.glob("*_integer_quant.tflite"))
if not int8_tflite_path:
    int8_tflite_path = list(saved_model_path.glob("*_int8.tflite"))
 
if int8_tflite_path:
    final_tflite_path = OUT_DIR / "{base_name}.tflite"
    shutil.copy(int8_tflite_path[0], final_tflite_path)
    size_kb = os.path.getsize(final_tflite_path) / 1024.0
    print(f"Quantized TFLite model successfully saved in: {{final_tflite_path}} ({{size_kb:.2f}} KB)")


    interpreter_int8 = tf.lite.Interpreter(model_path=str(final_tflite_path))
    interpreter_int8.allocate_tensors()

    input_details = interpreter_int8.get_input_details()[0]
    output_details = interpreter_int8.get_output_details()[0]

    in_scale, in_zero_point = input_details['quantization']
    out_scale, out_zero_point = output_details['quantization']

    quant_info = (
        f"--- Quantization Parameters (INT8) ---\\n"
        f"Model: {base_name}.tflite\\n\\n"
        f"INPUT:\\n"
        f"  Name: {{input_details['name']}}\\n"
        f"  Scale: {{in_scale}}\\n"
        f"  Zero Point: {{in_zero_point}}\\n"
        f"  Data Type: {{input_details['dtype']}}\\n\\n"
        f"OUTPUT:\\n"
        f"  Name: {{output_details['name']}}\\n"
        f"  Scale: {{out_scale}}\\n"
        f"  Zero Point: {{out_zero_point}}\\n"
        f"  Data Type: {{output_details['dtype']}}\\n"
    )

    txt_path = OUT_DIR / "quantization_params.txt"
    with open(txt_path, "w") as f:
        f.write(quant_info)
    print(f"* Quantization parameters saved in : {{txt_path}}")


    random_idx = np.random.randint(0, num_images)
    sample_image_float = data_for_calib[random_idx]

    if in_scale > 0.0:
        # Equation: Q = round(F / scale) + zero_point
        sample_image_quant = np.round(sample_image_float / in_scale) + in_zero_point
        sample_image_quant = np.clip(sample_image_quant, -128, 127).astype(np.int8)
    else:
        print("* Warning: Scale value is 0. The image could not be quantized.")
        sample_image_quant = sample_image_float.astype(np.float32)

    sample_image_contiguous = np.ascontiguousarray(sample_image_quant)

    bin_path = OUT_DIR / "random_test_sample_quantized.bin"
    with open(bin_path, "wb") as f:
        f.write(sample_image_contiguous.tobytes())

    print(f"* Random image (index {{random_idx}}) quantized and saved in: {{bin_path}}")
    
else:
    raise FileNotFoundError("onnx2tf did not generate the INT8 quantized model.")


final_tflite = Path('{OUT_PATH}') / f"{base_name}.tflite"

if final_tflite.exists():
    convert_tflite_to_tflm(str(final_tflite), str(Path('{OUT_PATH}')))
else:
    print("Advertencia: No se encontró el archivo .tflite final para convertir a .h")
 
"""
    
    temp_script_path = OUT_PATH / "temp_tflite_export.py"
    with open(temp_script_path, "w") as f:
        f.write(tflite_script)
 
    try:
        result = subprocess.run(
            ["uv", "run", "--python", "tf_env/bin/python", str(temp_script_path)],
            check = True,
            capture_output = False,
            text = True
        )
        print("Model conversion to TFLite has been completed succesfully...")
 
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: TFLite conversion has failed with code {e.returncode}.")
        raise
 
    except FileNotFoundError:
        print(f"\nERROR: 'uv' was not found.")
 
    finally:
        if temp_script_path.exists():
            temp_script_path.unlink()
        print()

        



