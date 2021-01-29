"""
Benchmarking script for PyTorch models.


##########
Command help:
usage: pytorch_benchmar.py [-h] --model-file-path MODEL_FILE_PATH --batch-size
                           BATCH_SIZE [--device DEVICE]
                           [--num-iterations NUM_ITERATIONS]
                           [--num-warmup-iterations NUM_WARMUP_ITERATIONS]
                           [--input-shape INPUT_SHAPE [INPUT_SHAPE ...]]

benchmark a PyTorch script model

optional arguments:
  -h, --help            show this help message and exit
  --model-file-path MODEL_FILE_PATH
                        Path to torch model script to benchmark. Should be
                        serialized with `torch.jit.save`
  --batch-size BATCH_SIZE
                        Batch size to benchmark with
  --device DEVICE       Device to benchmark the model with. Default is cpu
  --num-iterations NUM_ITERATIONS
                        Number of benchmarking iterations
  --num-warmup-iterations NUM_WARMUP_ITERATIONS
                        Number of warm up iterations before benchmarking
  --input-shape INPUT_SHAPE [INPUT_SHAPE ...]
                        Shape of FP32 input to the model not including the
                        batch dimension. Default is 3 224 224


##########
Example:
python scripts/pytorch_benchmar.py \
    --model-file-path PATH/TO/MODEL.pt \
    --batch-size 16

##########
Benchmarking with CUDA example:
python scripts/pytorch_benchmar.py \
    --model-file-path PATH/TO/MODEL.pt \
    --batch-size 16 \
    --num-iterations 100 \
    --num-warmup-iterations 25 \
    --device cuda:0
"""


import argparse
import time
from typing import List, Union

import torch
from tqdm.auto import trange

from deepsparse.benchmark import BenchmarkResults
from sparseml import get_main_logger


LOGGER = get_main_logger()
DEFAULT_INPUT_SHAPE = [3, 224, 224]


def _parse_device(device: Union[str, int]) -> Union[str, int]:
    try:
        return int(device)
    except:
        return device


def parse_args():
    parser = argparse.ArgumentParser(description="benchmark a PyTorch script model")
    parser.add_argument(
        "--model-file-path",
        type=str,
        required=True,
        help="Path to torch model script to benchmark. Should be serialized "
        "with `torch.jit.save`",
    )
    parser.add_argument(
        "--batch-size", type=int, required=True, help="Batch size to benchmark with"
    )
    parser.add_argument(
        "--device",
        type=_parse_device,
        default="cpu",
        help="Device to benchmark the model with. Default is cpu",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=20,
        help="Number of benchmarking iterations",
    )
    parser.add_argument(
        "--num-warmup-iterations",
        type=int,
        default=5,
        help="Number of warm up iterations before benchmarking",
    )
    parser.add_argument(
        "--input-shape",
        type=int,
        default=DEFAULT_INPUT_SHAPE,
        nargs="+",
        help="Shape of FP32 input to the model not including the batch dimension. "
        "Default is 3 224 224",
    )

    return parser.parse_args()


def main(args):
    # load model and input
    model = torch.jit.load(args.model_file_path)
    batch = torch.randn([args.batch_size] + args.input_shape)

    # set device
    model = model.to(args.device)

    # benchmark
    results = BenchmarkResults()
    LOGGER.info("Benchmarking...")
    for step in trange(args.num_warmup_iterations + args.num_iterations):
        inp = batch.to(args.device)
        if inp.is_cuda:
            torch.cuda.synchronize()

        start = time.time()
        outputs = model(inp)
        if inp.is_cuda:
            torch.cuda.synchronize()
        end = time.time()

        if step >= args.num_warmup_iterations:
            results.append_batch(start, end, args.batch_size)

        if inp.is_cuda:
            torch.cuda.synchronize()

        del inp
        del outputs

    LOGGER.info(f"Benchmarking complete. Results:\n{results}")


if __name__ == "__main__":
    args_ = parse_args()
    main(args_)
