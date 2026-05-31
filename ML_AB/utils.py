import random
import os

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_arg(name):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def configure_torch_threads(default_threads=3):
    """Limit CPU parallelism for long-running training jobs.

    On this 10-core machine, 3 threads is roughly a 30% CPU budget. This is not
    a hard OS scheduler cap, but it prevents PyTorch/BLAS from occupying every
    core during training and eval.
    """
    threads = int(os.environ.get("CPU_THREADS", default_threads))
    threads = max(1, threads)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(max(1, min(threads, 2)))
    except RuntimeError:
        pass
    return threads
