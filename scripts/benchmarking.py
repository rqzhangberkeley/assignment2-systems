# benchmarking the forward and the backward pass.

import os, json, sys, torch, math, time, timeit, random
import argparse, logging
import numpy as np
import pandas as pd
from tqdm import tqdm

from typing import IO, Any, BinaryIO, Tuple
from jaxtyping import Bool, Float, Int
from torch import Tensor

from contextlib import nullcontext # this creates a context manager that does nothing.

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW, get_cosine_lr
from cs336_basics.nn_utils import cross_entropy

def setup_logger(logger_name: str):
    # RZ: This is taken from the HW1.
    logger = logging.getLogger(logger_name) # a logging.Logger object. Name it.
    logger.setLevel(logging.INFO) # set the minimum severity level for this logger.
    logger.handlers.clear() # remove any existing handlers to avoid duplicate logs.

    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    # return a logging.Formatter object which defines the logging format.
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger

def build_parser():
    parser = argparse.ArgumentParser(
        'Benchmarking the forward and the backward pass.'
    )
    
    parser.add_argument(
        '--model_sizes',
        type=str,
        default='all',
        help='Comma-separated GPT-2 sizes: small,medium,large,xl or "all".'
    )
    parser.add_argument('--vocab_size', type=int, default=10_000)
    parser.add_argument('--context_length', type=int, default=256)
    parser.add_argument('--d_model', type=int, default=768)
    parser.add_argument('--num_layers', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--d_ff', type=int, default=3072)
    parser.add_argument('--rope_theta', type=int, default=10_000)

    # data
    parser.add_argument('--batch_size', type=int, default=4)

    # training
    parser.add_argument('--warmup_steps', type=int, default=5)
    parser.add_argument('--training_steps', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.01)

    # device
    parser.add_argument('--device', type=str, default='cuda')

    # whether to use autocast
    parser.add_argument('--use_autocast', action='store_true')

    # whether we only do the forward pass
    parser.add_argument('--only_forward', action='store_true')

    # whether do the memory profiling
    parser.add_argument('--do_memory_profiling', action='store_true')
    parser.add_argument('--memory_profiling_file', type=str, default='./results/memory_profile.pkl')
    return parser


GPT2_CONFIGS = {
    "small": dict(num_layers=12, num_heads=12, d_model=768, d_ff=3072),
    "medium": dict(num_layers=24, num_heads=16, d_model=1024, d_ff=4096),
    "large": dict(num_layers=36, num_heads=20, d_model=1280, d_ff=5120),
    "xl": dict(num_layers=48, num_heads=25, d_model=1600, d_ff=6400),
    "2.7B": dict(num_layers=32, num_heads=32, d_model=2560, d_ff=10240)
}


def _iter_model_sizes(model_sizes: str) -> list[str]:
    sizes = [s.strip() for s in model_sizes.split(",") if s.strip()]
    if not sizes or sizes == ["all"]:
        return list(GPT2_CONFIGS.keys())
    for size in sizes:
        if size not in GPT2_CONFIGS:
            raise ValueError(f"Unknown model size '{size}'. Choose from {list(GPT2_CONFIGS.keys())} or 'all'.")
    return sizes

def generate_random_data(
    batch_size: int,
    context_length: int,
    vocab_size: int,
    device: str
):
    return torch.randint(
        low = 0,
        high = vocab_size, # exclusive
        size = (batch_size, context_length),
        device = device
    ).long() # return Torch.long object for indexing.


def benchmarking(args):
    # logger.
    logger = setup_logger('Benchmarking.....')
    logger.info(
        f'vocab_size = {args.vocab_size}\n'
        f'context_length = {args.context_length}\n'
        f'd_model = {args.d_model}\n'
        f'num_layers = {args.num_layers}\n'
        f'num_heads = {args.num_heads}\n'
        f'd_ff = {args.d_ff}\n'
        f'rope_theta = {args.rope_theta}\n'
        f'batch_size = {args.batch_size}\n'
        f'warmup_steps = {args.warmup_steps}\n'
        f'training_steps = {args.training_steps}\n'
        f'use_autocast = {args.use_autocast}\n'
        f'only_forward = {args.only_forward}\n'
        f'do_memory_profiling = {args.do_memory_profiling}\n'
        f'memory_profiling_file = {args.memory_profiling_file}\n'
    )

    device = args.device

    # memory profiling.
    if args.do_memory_profiling:
        torch.cuda.memory._record_memory_history(max_entries=1000000)

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta
    ).to(device)

    optimizer = AdamW(
        params = model.parameters(),
        lr = args.lr
    )

    m, n = args.warmup_steps, args.training_steps
    amp_ctx = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if args.use_autocast else nullcontext()
    grad_ctx = torch.no_grad() if args.only_forward else nullcontext()

    # warmup.
    for i in tqdm(range(m)):
        inputs = generate_random_data(
            batch_size=args.batch_size,
            context_length=args.context_length,
            vocab_size=args.vocab_size,
            device=device
        )
        targets = generate_random_data(
            batch_size=args.batch_size,
            context_length=args.context_length,
            vocab_size=args.vocab_size,
            device=device
        )
        optimizer.zero_grad()
        with grad_ctx, amp_ctx:
            logits = model(inputs)

        # if only forward pass.
        if args.only_forward:
            continue
        loss = cross_entropy(logits, targets)
        start_time = timeit.default_timer()
        loss.backward()
        end_time = timeit.default_timer()
        optimizer.step()

    if args.do_memory_profiling:
        # Exclude warmup from peak stats and ensure kernels are finished.
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device=device)

    forward_time_lst = []
    backward_time_lst = []

    for i in tqdm(range(n)):
        inputs = generate_random_data(
            batch_size=args.batch_size,
            context_length=args.context_length,
            vocab_size=args.vocab_size,
            device=device
        )

        targets = generate_random_data(
            batch_size=args.batch_size,
            context_length=args.context_length,
            vocab_size=args.vocab_size,
            device=device
        )

        optimizer.zero_grad()

        start_time = timeit.default_timer()
        with grad_ctx, amp_ctx: # if we only measure the forward pass, we must run this without the grad graph.
            logits = model(inputs)
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        forward_time_lst.append(end_time-start_time)

        # if only forward pass.
        if args.only_forward:
            continue

        with amp_ctx:
            loss = cross_entropy(logits, targets)

        start_time = timeit.default_timer()
        with amp_ctx:
            loss.backward()
        torch.cuda.synchronize()
        end_time = timeit.default_timer()
        backward_time_lst.append(end_time-start_time)

        with amp_ctx:
            optimizer.step()
        torch.cuda.synchronize()
        # RZ: CUDA calls are asynchronous. So for example, when we call torch.matmul, the function call returns control to your code without waiting for the matrix multiplication to finish, and the CPU continues while the GPU works on the matrix multiplication.
        # So directly measuring the time of the torch.matmul is not accurate. We need to call torch.cuda.synchronize() to wait for the GPU to finish the work before measuring the time.

    # save a pickle file to be loased by pytorch's online tool
    if args.do_memory_profiling:
        torch.cuda.memory._dump_snapshot(args.memory_profiling_file)

    # stop recording history.
    if args.do_memory_profiling:
        torch.cuda.memory._record_memory_history(enabled=None)
        print(f'memory usage = {torch.cuda.memory_summary(device=device, abbreviated=True)}')
        print(f'Peak memory usage = {torch.cuda.max_memory_allocated(device=device) / 1024**3} GB')
        print(f'max reserved memory = {torch.cuda.max_memory_reserved(device=device) / 1024**3} GB')

    # print the results.
    print(f'The mean of the forward pass time = {np.mean(forward_time_lst)}')
    print(f'The std of the forward pass time = {np.std(forward_time_lst)}')

    print(f'The mean of the backward pass time = {np.mean(backward_time_lst)}')
    print(f'The std of the backward pass time = {np.std(backward_time_lst)}')

    summary = {
        'num_layers': args.num_layers,
        'num_heads': args.num_heads,
        'd_model': args.d_model,
        'd_ff': args.d_ff,
        'forward_time_mean': np.mean(forward_time_lst),
        'forward_time_std': np.std(forward_time_lst),
        'backward_time_mean': np.mean(backward_time_lst),
        'backward_time_std': np.std(backward_time_lst),
    }

    return summary

if __name__ == '__main__':
    args = build_parser().parse_args()
    summaries = []
    for size in _iter_model_sizes(args.model_sizes):
        cfg = GPT2_CONFIGS[size]
        run_args = argparse.Namespace(**vars(args))
        run_args.num_layers = cfg["num_layers"]
        run_args.num_heads = cfg["num_heads"]
        run_args.d_model = cfg["d_model"]
        run_args.d_ff = cfg["d_ff"]
        print(f"\n=== Running GPT-2 {size} ===")
        summary = benchmarking(run_args)
        summary['model_size'] = size
        summaries.append(summary)
    # df = pd.DataFrame(summaries)
    # if args.use_autocast:
    #     df.to_markdown(f'./results/benchmarking_autocast_bf16.md', index=False)
    # else:
    #     df.to_markdown(f'./results/benchmarking.md', index=False)



