import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from ptflops import get_model_complexity_info
from torch.profiler import profile, ProfilerActivity, schedule

def profile_model(model, 
                  device: torch.device,
                  output_dir: str = None,
                  output_name: str = None,
                  num_warmup: int = 10,
                  num_runs: int = 100):
    
    model = model.to(device)
    model.eval()

    print("Initiating model profiling...")
    print()

    print("======= MACs & FLOPs =======")

    img_size = model.config.image_size
    num_channels = model.config.num_channels
    input_shape = (num_channels, img_size, img_size)

    dummy_input = torch.randn((8, *input_shape)).to(device)

    # Measure GMACs and GFLOPS
    macs, params = get_model_complexity_info(
            model,
            input_shape,
            as_strings = False,
            print_per_layer_stat = False
    )

   
    flops = 2 * macs
    print(f"MACs: {macs / 1e9:.4f} GMACs")
    print(f"FLOPS: {flops / 1e9:.4f} GFLOPS")
    print(f"Params: {params / 1e6:.4f} M")
    print()

    print("======= Memory & Time Footprint (GPU & CPU) =======")

    # Profiler Configuration
    prof_schedule = schedule(wait = 1, warmup = 1, active = 2, repeat = 1)

    # Activities (CPU & GPU)
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    # Profiler Cycle
    with profile(
        activities = activities,
        schedule = prof_schedule,
        record_shapes = True,
        profile_memory = True,
        with_stack = True,
        with_flops = True
    ) as prof:

        for step in range(5):
            # Inference
            with torch.no_grad():
                _ = model(dummy_input)

            # Notify to profiler
            prof.step()

    print("*** Time (CPU & GPU)")
    print(prof.key_averages().table(
        sort_by = "self_cpu_time_total",
        row_limit = 15
        )
    )

    print()

    print(prof.key_averages().table(
        sort_by = "self_cuda_time_total",
        row_limit = 15
        )
    )

    print()

    print("*** Memory Usage (CPU & GPU)")
    print(prof.key_averages().table(
        sort_by = "self_cpu_memory_usage",
        row_limit = 10
        )
    )

    print()

    print(prof.key_averages().table(
        sort_by = "self_cuda_memory_usage",
        row_limit = 10
        )
    )

    print()

    if output_dir is not None:

        output_name = output_name or "model"

        print("======= Structured Summary (single-sample latency & peak memory) =======")

        # --- Single-sample (batch = 1) latency ---
        single_input = torch.randn((1, *input_shape)).to(device)
 
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
 
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(single_input)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
 
        latencies_ms = []
        with torch.no_grad():
            for _ in range(num_runs):
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                _ = model(single_input)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t1 = time.perf_counter()
                latencies_ms.append((t1 - t0) * 1000.0)
 
        latencies_ms = np.asarray(latencies_ms)
        latency_stats = {
            "mean": float(latencies_ms.mean()),
            "std": float(latencies_ms.std()),
            "min": float(latencies_ms.min()),
            "max": float(latencies_ms.max()),
            "p50": float(np.percentile(latencies_ms, 50)),
            "p95": float(np.percentile(latencies_ms, 95)),
            "p99": float(np.percentile(latencies_ms, 99)),
        }
        throughput_fps = 1000.0 / latency_stats["mean"] if latency_stats["mean"] > 0 else None
 
        # --- Peak memory -----------------------------------------------------
        peak_memory = {}
        if device.type == "cuda":
            peak_memory["cuda_max_allocated_mb"] = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
            peak_memory["cuda_max_reserved_mb"] = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        try:
            import resource  # POSIX-only (Linux/macOS); not available on Windows.
            # ru_maxrss is reported in KB on Linux (assumed here, since that is
            # this project's typical training/inference server OS) but in
            # bytes on macOS; if you profile on macOS, divide by 1024 less.
            peak_memory["host_max_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        except ImportError:
            peak_memory["host_max_rss_mb"] = None
 
        print(f"Single-sample (batch=1) latency -> mean: {latency_stats['mean']:.3f} ms, "
              f"p95: {latency_stats['p95']:.3f} ms, throughput: {throughput_fps:.2f} FPS")
        print(f"Peak memory: {peak_memory}")
        print()
 
        summary = {
            "device": str(device),
            "model_config": {
                "image_size": img_size,
                "num_channels": num_channels,
            },
            "complexity": {
                "gmacs": macs / 1e9,
                "gflops": flops / 1e9,
                "params_m": params / 1e6,
            },
            "batched_profile": {
                "batch_size": dummy_input.shape[0],
                "note": ("Detailed per-op CPU/CUDA time & memory tables were "
                         "printed to stdout via torch.profiler above; they are "
                         "not duplicated here, only the aggregate single-sample "
                         "figures below are stored for programmatic analysis."),
            },
            "single_sample_latency_ms": {
                "batch_size": 1,
                "num_warmup": num_warmup,
                "num_runs": num_runs,
                **latency_stats,
            },
            "throughput_fps_single_sample": throughput_fps,
            "peak_memory": peak_memory,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
 
        out_dir = Path(output_dir)
        out_dir.mkdir(parents = True, exist_ok = True)
        summary_path = out_dir / f"{output_name}_torch_profile.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent = 2, default = str)
 
        print(f"Structured summary saved in: {summary_path}")
        print()

def profile_torch_model_from_checkpoint(model_role: str,
                                        checkpoint_path: str,
                                        output_dir: str,
                                        student_arch: str = None,
                                        output_name: str = None,
                                        device: str = "cuda",
                                        num_warmup: int = 10,
                                        num_runs: int = 100) -> None:


    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} has not been found.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents = True, exist_ok = True)

    if output_name is None:
        output_name = f"{model_role}_{student_arch}" if student_arch else model_role

    worker_script = Path(__file__).resolve().parent.parent / "workers" / "torch_profiler_worker.py"
    if not worker_script.exists():
        raise FileNotFoundError(
            f"Auxiliar script '{worker_script}' has not been found. It must to be "
            "in the 'workers' directory."
        )

    cmd = [
        sys.executable, str(worker_script),
        "--model-role", model_role,
        "--checkpoint-path", str(checkpoint_path),
        "--output-dir", str(out_dir),
        "--output-name", output_name,
        "--device", device,
        "--num-warmup", str(num_warmup),
        "--num-runs", str(num_runs),
    ]
    if model_role == "student":
        if not student_arch:
            raise ValueError("'student_arch' is required when model_role='student'")
        cmd += ["--student-arch", student_arch]

    print("*** Profiling torch model...")


    try:
        subprocess.run(
            cmd,
            check = True,
            capture_output = False,
            text = True
        )
        print("Torch profiling has been completed succesfully...")
 
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Torch profiling subprocess has failed with code {e.returncode}.")
        raise
 
    except FileNotFoundError:
        print(f"\nERROR: Python interpreter '{sys.executable}' was not found.")
        raise
 
    finally:
        print()











