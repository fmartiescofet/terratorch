import socket
from datetime import datetime

import torch

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"


def trace_handler(prof: torch.profiler.profile):
    # Prefix for file names.
    host_name = socket.gethostname()
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"{host_name}_{timestamp}"

    # Construct the trace file.
    # prof.export_chrome_trace(f"{file_prefix}.json.gz")

    # Construct the memory timeline file.
    # print("timeline in tarce_handler")
    # prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")


PROFILER = torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    schedule=torch.profiler.schedule(wait=0, warmup=0, active=53, repeat=1),
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    on_trace_ready=trace_handler,
)
