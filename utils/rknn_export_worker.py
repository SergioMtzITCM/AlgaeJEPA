import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from rknn.api import RKNN

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-path", required = True, type = str)
    parser.add_argument("--test-data-path", required = True, type = str,
                         help = ".h5 File with representative dataset. Only read it "
                              "if --do-quantization is True.")
    parser.add_argument("--output-dir", required = True, type = str)
    parser.add_argument("--output-name", required = True, type = str,
                         help = "Base name for the final '.rknn' file.")
    parser.add_argument("--input-shape", required = True, type = int, nargs = 3,
                         metavar = ("C", "H", "W"),
                         help = "Model's input shape (canales, alto, ancho), "
                              "NCHW format, without batch dimension.")
    parser.add_argument("--do-quantization", action = "store_true",
                         help = "Builds INT8 quantized model. Requires --test-data-path.")
    parser.add_argument("--target-platform", default = "rk3588", type = str)
    parser.add_argument("--quantized-algorithm", default = "normal",
                         choices = ["normal", "mmse", "kl_divergence"],
                         help="normal: faster, 20-100 calibration images. mmse: slow "
                              "but more precise, 20-50 images. kl_divergence: "
                              "intermedium, 20-100 images.")
    parser.add_argument("--quantized-method", default = "channel",
                         choices = ["layer", "channel"],
                         help = "'channel' (per channel) more precise than 'layer' (per tensor).")
    parser.add_argument("--calib-samples", type = int, default = 100,
                         help="Number of used images for calibration (Rockchip "
                              "recommends 20-100).")
    parser.add_argument("--seed", type = int, default = 42)
    return parser.parse_args()


def match_format(images: np.ndarray, 
                 expected_shape: tuple) -> np.ndarray:
    """Convert the data to format NCHW, if it is required."""

    if images.shape[1:] == tuple(expected_shape):
        return images
    if images.ndim == 4 and images.shape[-1] == expected_shape[0]:
        # Data are in NHWC format -> transpose to NCHW
        return np.transpose(images, (0, 3, 1, 2))
    raise ValueError(
        f"Calibration data shape {images.shape[1:]} does not match "
        f"with model's expected input shape {tuple(expected_shape)}."
    )

def build_calibration_dataset(h5_path: Path, 
                              out_dir: Path, 
                              expected_shape: tuple,
                              num_samples: int, 
                              seed: int) -> Path:
    """
    Reads the '.h5' representative dataset, chooses random 'num_samples' images,
    saves each one int '.npy' file and writes a 'dataset.txt' file.
    """
    calib_dir = out_dir / "rknn_calib_images"
    calib_dir.mkdir(parents = True, exist_ok = True)
 
    print(f"* Loading calibration data from: {h5_path}...")
    with h5py.File(h5_path, "r") as h5_file:
        num_images = h5_file["images"].shape[0]
        all_images = h5_file["images"][:]
 
    rng = np.random.default_rng(seed)
    count = min(num_images, num_samples)
    indices = rng.choice(num_images, size = count, replace = False)
 
    images = match_format(all_images[indices], expected_shape).astype(np.float32)
 
    print(
        f"* Calibration data statistics -> min: {images.min():.4f}, "
        f"max: {images.max():.4f}, mean: {images.mean():.4f}, std: {images.std():.4f}"
    )
 
    dataset_txt_path = out_dir / "dataset.txt"
    with open(dataset_txt_path, "w") as f:
        for i, img in enumerate(images):
            npy_path = calib_dir / f"calib_{i:04d}.npy"
            np.save(npy_path, img)
            f.write(f"{npy_path}\n")
 
    print(f"* {count} calibration data images saved in '.npy' file in: {calib_dir}")
    print(f"* 'dataset.txt' file written in: {dataset_txt_path}")
    return dataset_txt_path


def main() -> None:
 
    args = parse_args()
 
    onnx_path = Path(args.onnx_path)
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file {onnx_path} has not been found.")
 
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents = True, exist_ok = True)
    rknn_path = out_dir / f"{args.output_name}.rknn"
 
    input_shape = tuple(args.input_shape)  # (C, H, W)
 
    dataset_txt_path = None
    if args.do_quantization:
        test_data_path = Path(args.test_data_path)
        if not test_data_path.exists():
            raise FileNotFoundError(f"Calibration file {test_data_path} has not been found.")
 
        print("** 1. Preparing calibration dataset for RKNN quantization...")
        dataset_txt_path = build_calibration_dataset(
            h5_path = test_data_path,
            out_dir = out_dir,
            expected_shape = input_shape,
            num_samples = args.calib_samples,
            seed = args.seed,
        )
    else:
        print("* do_quantization=False -> exporting a float RKNN model, calibration data is not required.")
 
    print("** 2. Configuring RKNN...")
    rknn = RKNN(verbose = True)
 
    rknn.config(
        mean_values = [[0] * input_shape[0]],
        std_values = [[1] * input_shape[0]],
        target_platform = args.target_platform,
        quantized_algorithm = args.quantized_algorithm,
        quantized_method = args.quantized_method,
    )
 
    print("** 3. Loading ONNX model...")
    ret = rknn.load_onnx(model = str(onnx_path), input_size_list = [list(input_shape)])
    if ret != 0:
        raise RuntimeError(f"Error when loading ONNX (code: {ret})")
 
    print("** 4. Building RKNN model...")
    build_kwargs = {"do_quantization": args.do_quantization}
    if dataset_txt_path is not None:
        build_kwargs["dataset"] = str(dataset_txt_path)
    ret = rknn.build(**build_kwargs)
    if ret != 0:
        raise RuntimeError(f"Error when building RKNN (code: {ret})")
 
    print("** 5. Exporting RKNN model...")
    ret = rknn.export_rknn(str(rknn_path))
    if ret != 0:
        raise RuntimeError(f"Error when exporting RKNN (code: {ret})")
 
    rknn.release()
    print(f"RKNN model saved in: {rknn_path}")
 
 
if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {exc}", file = sys.stderr)
        sys.exit(1)
