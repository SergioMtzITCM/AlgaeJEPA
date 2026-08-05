import torch
from ptflops import get_model_complexity_info
from torch.profiler import profile, ProfilerActivity, schedule

def profile_model(model, device: torch.device):
    
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















