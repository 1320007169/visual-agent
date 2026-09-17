#!/usr/bin/env python3
"""Bind one CPU tool process to its reserved cores before importing torch."""

import argparse
import os
from pathlib import Path
import runpy
import sys


def server_cores(cores, threads, index):
    ids = [int(value) for value in cores.split(",")]
    if threads <= 0 or index < 0 or len(set(ids)) != len(ids) or min(ids) < 0:
        raise ValueError("Invalid core IDs, thread count or server index")
    selected = ids[index * threads:(index + 1) * threads]
    if len(selected) != threads:
        raise ValueError("Not enough reserved CPU cores for all tool servers")
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cores", required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--server-index", type=int, required=True)
    args, server_args = parser.parse_known_args()
    cores = server_cores(args.cores, args.threads, args.server_index)
    os.sched_setaffinity(0, cores)
    os.environ.update(CUDA_VISIBLE_DEVICES="", GROUNDING_DINO_DEVICE="cpu",
                      OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads),
                      OPENBLAS_NUM_THREADS=str(args.threads), TOKENIZERS_PARALLELISM="false")
    import torch

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    print(f"CPU tool server: cores={cores}, torch_threads={torch.get_num_threads()}", flush=True)
    server = Path(__file__).with_name("visual_tool_server.py")
    sys.argv = [str(server), *server_args]
    runpy.run_path(str(server), run_name="__main__")


if __name__ == "__main__":
    main()
