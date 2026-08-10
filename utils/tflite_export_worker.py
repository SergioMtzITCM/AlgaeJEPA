import argparse
import shutil
import sys
from pathlib import Path

import h5py
import numpy as np
import tensorflow as tf
from onnx2tf import convert


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-path", required=True, type=str)
    parser.add_argument("--test-data-path", required=True, type=str,
                         help=".h5 File with representative dataset.")
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument("--output-name", required=True, type=str,
                         help="Base name for the final '.tflite' file and the reports.")
    parser.add_argument("--quant-type", default="per-channel",
                         choices=["per-channel", "per-tensor"],
                         help="")
    parser.add_argument("--calib-samples", type=int, default = 350,
                         help="Max number of images used to calibration. (100-500 recommended).")
    parser.add_argument("--val-samples", type=int, default = 50,
                         help="Max number of images only to measure the quantization error")
    parser.add_argument("--seed", type=int, default = 42)
    return parser.parse_args()

def match_format(data: np.ndarray, expected_dims, model_format: str) -> np.ndarray:
    """Convert the data to format (NCHW/NHWC), if it is required."""
    if model_format == "NCHW" and data.shape[1:] != tuple(expected_dims):
        return np.transpose(data, (0, 3, 1, 2))
    if model_format == "NHWC" and data.shape[1:] != tuple(expected_dims):
        return np.transpose(data, (0, 2, 3, 1))
    return data


def pick_float32_reference(saved_model_path: Path) -> Path:
    """
    Choose the true float32 .tflite file.
    """
    tflite_files = list(saved_model_path.glob("*.tflite"))
    float_files = [f for f in tflite_files if "int8" not in f.name and "quant" not in f.name]
    if not float_files:
        raise FileNotFoundError(
            f".tflite file without quantization has not been found in {saved_model_path}."
        )
 
    float32_matches = sorted(f for f in float_files if "float32" in f.name)
    if float32_matches:
        return float32_matches[0]
 
    fallback = sorted(float_files)[0]
    print(
        "* Warning: no file with 'float32' explicitly was found;"
        f"'{fallback.name}' will be used as a reference (verify that it is the correct one)."
    )
    return fallback

def pick_int8_io_model(saved_model_path: Path) -> Path:
    """
    Choose between all the candidates only the *_integer_quant.tflite and *_ful_integer_quant.tflite files.
    Verify if their INPUT and OUTPUT are int8.
    """
    candidates: list[Path] = []
    seen = set()
    for pattern in ("*_full_integer_quant.tflite", "*_integer_quant.tflite"):
        for f in sorted(saved_model_path.glob(pattern)):
            if f not in seen:
                seen.add(f)
                candidates.append(f)
 
    if not candidates:
        raise FileNotFoundError(
            "onnx2tf did not generate neither *_integer_quant.tflite nor"
            "*_full_integer_quant.tflite files."
        )
 
    print("* INT8 Candidates founded (verifying real input/output dtype:")
    for candidate in candidates:
        interp = tf.lite.Interpreter(model_path=str(candidate))
        interp.allocate_tensors()
        in_dtype = interp.get_input_details()[0]["dtype"]
        out_dtype = interp.get_output_details()[0]["dtype"]
        print(f"  - {candidate.name}: input={in_dtype.__name__}, output={out_dtype.__name__}")
        if in_dtype == np.int8 and out_dtype == np.int8:
            print(f"  -> Selected: {candidate.name} (input and output are int8)")
            return candidate
 
    raise FileNotFoundError(
        "None of the candidates have int8 input and output (probably only quantized the weights:)"
        "'integer quantization with float fallback'")

def main() -> None:

    args = parse_args()
 
    onnx_path = Path(args.onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX File {onnx_path} has not been found.")
 
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_model_path = out_dir / "saved_model"

    # ---- Phase 1: Conversion without quantization, only to inspect shape/format ----
    print("** 1. Generating TFLite (Float32) for inspect it with onnx2tf...")

    try:
        convert(
            input_onnx_file_path = str(onnx_path),
            output_folder_path = str(saved_model_path),
            output_signaturedefs = True,
            non_verbose = False,
            disable_strict_mode = True,
            tflite_backend = "tf_converter",
    )
    except Exception as e:
        raise RuntimeError(f"Error in onnx2tf (Phase 1): {e}") from e

    float32_tflite_path = pick_float32_reference(saved_model_path)
    print(f"* Inspecting the generated TFLite model: {float32_tflite_path.name}...")

    interpreter_f32 = tf.lite.Interpreter(model_path = str(float32_tflite_path))
    interpreter_f32.allocate_tensors()
    f32_input_details = interpreter_f32.get_input_details()[0]
    f32_output_details = interpreter_f32.get_output_details()[0]
    model_shape = f32_input_details["shape"].tolist()
    model_dtype = f32_input_details["dtype"]

    if len(model_shape) < 4:
        raise ValueError(f"Unexpected input shape: {model_shape}")
    expected_spatial_dims = model_shape[1:]

    print(f"Input Name: {f32_input_details['name']}, Expected Shape: {model_shape}, "
          f"Dtype: {model_dtype.__name__}")

    if len(expected_spatial_dims) == 3:
        if expected_spatial_dims[0] in (1, 3):
            model_format = "NCHW"
            channels = expected_spatial_dims[0]
        elif expected_spatial_dims[0] > 3 and expected_spatial_dims[1] > 3 and expected_spatial_dims[2] in (1, 3):
            model_format = "NHWC"
            channels = expected_spatial_dims[2]
        else:
            model_format = "UNKNOWN"
            channels = 3
    else:
        model_format = "UNKNOWN"
        channels = 3

    print(f"Model's input format: {model_format}")

    # ---- Representative data loading and division (calibration / validation) ----
    print(f"* Loading calibration data from: {args.test_data_path}...")
    with h5py.File(args.test_data_path, "r") as h5_file:
        num_images = h5_file["images"].shape[0]
        all_images = h5_file["images"][:]

    rng = np.random.default_rng(args.seed)
    shuffled_indices = rng.permutation(num_images)

    calib_count = min(num_images, args.calib_samples)
    calib_indices = shuffled_indices[:calib_count]
    remaining_indices = shuffled_indices[calib_count:]

    val_count = min(len(remaining_indices), args.val_samples)
    if val_count >= 10:
        val_indices = remaining_indices[:val_count]
        val_is_heldout = True
    else:
        val_indices = calib_indices[: min(len(calib_indices), 30)]
        val_is_heldout = False
        print(
            f"* Warning: there are only {num_images} in total; not enough to"
            "separate a validation set independent of the calibration set."
            f"{len(val_indices)} calibration images will be reused solely as a rough"
            "reference for the quantization error."
        )

    data_for_calib = match_format(all_images[calib_indices], expected_spatial_dims, model_format).astype(np.float32)
    data_for_val = match_format(all_images[val_indices], expected_spatial_dims, model_format).astype(np.float32)

    print(
        f"* Calibration data statistics -> min: {data_for_calib.min():.4f}"
        f"max: {data_for_calib.max():.4f}, mean: {data_for_calib.mean():.4f}, "
        f"std: {data_for_calib.std():.4f}"
    )
    

    calib_data_path = out_dir / "calib_data.npy"
    np.save(calib_data_path, data_for_calib)
    print(f"Calibration data saved in: {calib_data_path}")

    mean_vals = [0.0] * channels
    std_vals = [1.0] * channels

    # ---- Phase 2: INT8 quantization ----
    try:
        convert(
            input_onnx_file_path = str(onnx_path),
            output_folder_path = str(saved_model_path),
            output_signaturedefs = True,
            non_verbose = False,
            disable_strict_mode = True,
            tflite_backend = "tf_converter",
            output_integer_quantized_tflite = True,
            quant_type = args.quant_type,
            custom_input_op_name_np_data_path = [["input", str(calib_data_path), mean_vals, std_vals]],
        )
    except Exception as e:
        raise RuntimeError(f"Error in onnx2tf (cuantization phase): {e}") from e

    int8_source_path = pick_int8_io_model(saved_model_path)
 
    final_tflite_path = out_dir / f"{args.output_name}.tflite"
    shutil.copy(int8_source_path, final_tflite_path)
    size_kb = final_tflite_path.stat().st_size / 1024.0
    print(f"Quantized TFLite model successfully saved in: {final_tflite_path} ({size_kb:.2f} KB)")

    interpreter_int8 = tf.lite.Interpreter(model_path=str(final_tflite_path))
    interpreter_int8.allocate_tensors()
    int8_input_details = interpreter_int8.get_input_details()[0]
    int8_output_details = interpreter_int8.get_output_details()[0]
 
    in_scale, in_zero_point = int8_input_details["quantization"]
    out_scale, out_zero_point = int8_output_details["quantization"]

    # ---- Phase 3: cuantization error (float32 vs dequantized int8) ----
    print("** 3. Validating quantization error (cosine / MSE)...")
    cos_sims, mses = [], []

    for img in data_for_val:
        sample = np.expand_dims(img, axis=0).astype(f32_input_details["dtype"])
 
        interpreter_f32.set_tensor(f32_input_details["index"], sample)
        interpreter_f32.invoke()
        out_f32 = interpreter_f32.get_tensor(f32_output_details["index"])
 
        if in_scale > 0.0:
            sample_quant = np.round(sample / in_scale) + in_zero_point
            sample_quant = np.clip(sample_quant, -128, 127).astype(np.int8)
        else:
            sample_quant = sample.astype(np.int8)
 
        interpreter_int8.set_tensor(int8_input_details["index"], sample_quant)
        interpreter_int8.invoke()
        out_int8_raw = interpreter_int8.get_tensor(int8_output_details["index"])
        out_int8_dequant = (out_int8_raw.astype(np.float32) - out_zero_point) * out_scale
 
        f32_flat = out_f32.flatten()
        q_flat = out_int8_dequant.flatten()
        denom = np.linalg.norm(f32_flat) * np.linalg.norm(q_flat)
        cos_sims.append(float(np.dot(f32_flat, q_flat) / denom) if denom > 0 else float("nan"))
        mses.append(float(np.mean((f32_flat - q_flat) ** 2)))

    cos_sims_arr = np.array(cos_sims)
    mses_arr = np.array(mses)
    avg_cos = float(np.nanmean(cos_sims_arr))
    avg_mse = float(np.mean(mses_arr))

    print(f"* Average cosine similarity (float32 vs dequantized int8): {avg_cos:.5f}")
    print(f"* Average MSE: {avg_mse:.6f}")
    if not val_is_heldout:
        print("* (Este numero es optimista: el set de 'validacion' se reutilizo del de calibracion.)")

    # ---- Quantization parameters report ----
    quant_info = (
        "--- Quantization Parameters (INT8) ---\n"
        f"Source model (ONNX): {onnx_path.stem}\n"
        f"Final file: {args.output_name}.tflite\n"
        f"Used onnx2tf candidate: {int8_source_path.name}\n"
        f"quant_type (pesos): {args.quant_type}\n\n"
        "INPUT:\n"
        f"  Name: {int8_input_details['name']}\n"
        f"  Scale: {in_scale}\n"
        f"  Zero Point: {in_zero_point}\n"
        f"  Data Type: {int8_input_details['dtype'].__name__}\n\n"
        "OUTPUT:\n"
        f"  Name: {int8_output_details['name']}\n"
        f"  Scale: {out_scale}\n"
        f"  Zero Point: {out_zero_point}\n"
        f"  Data Type: {int8_output_details['dtype'].__name__}\n\n"
        "--- Quantization Error (float32 vs dequantized int8) ---\n"
        f"Validation set independent of the calibration: {val_is_heldout}\n"
        f"Number of validation images: {len(data_for_val)}\n"
        f"Average cosine similarity: {avg_cos:.5f} "
        f"(min: {float(np.nanmin(cos_sims_arr)):.5f}, max: {float(np.nanmax(cos_sims_arr)):.5f})\n"
        f"Average MSE: {avg_mse:.6f}\n"
    )

    txt_path = out_dir / "quantization_params.txt"
    with open(txt_path, "w") as f:
        f.write(quant_info)
    print(f"* Quantization parameters saved in: {txt_path}")

    # ---- Random quantized sample for example (taken of the validation set) ---
    random_idx = int(np.random.default_rng(args.seed + 1).integers(0, len(data_for_val)))
    sample_image_float = data_for_val[random_idx]

    if in_scale > 0.0:
        sample_image_quant = np.round(sample_image_float / in_scale) + in_zero_point
        sample_image_quant = np.clip(sample_image_quant, -128, 127).astype(np.int8)
    else:
        print("* Warning: Scale value is 0. The image could not be quantized.")
        sample_image_quant = sample_image_float.astype(np.float32)

    sample_image_contiguous = np.ascontiguousarray(sample_image_quant)
    bin_path = out_dir / "random_test_sample_quantized.bin"
    with open(bin_path, "wb") as f:
        f.write(sample_image_contiguous.tobytes())

    print(f"* Random image (validation index {random_idx}) quantized and saved in: {bin_path}")
    print("\nModel conversion to TFLite has been completed succesfully...")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file = sys.stderr)
        sys.exit(1)

