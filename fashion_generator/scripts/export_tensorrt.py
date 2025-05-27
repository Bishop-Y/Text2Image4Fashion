#!/usr/bin/env python

import argparse
import time
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit


def parse_args():
    parser = argparse.ArgumentParser(
        description="TensorRT Static-Shape Inference Latency Measurement"
    )
    parser.add_argument(
        "--trt_path", required=True,
        help="Path to the TensorRT engine file (.trt)"
    )
    parser.add_argument(
        "--iters", type=int, default=100,
        help="Number of iterations to measure latency"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(args.trt_path, 'rb') as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine  = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    in0_shape = engine.get_binding_shape(0)
    in1_shape = engine.get_binding_shape(1)
    out_shape = engine.get_binding_shape(2)

    text_np   = np.random.randn(*in0_shape).astype(np.float32)
    noise_np  = np.random.randn(*in1_shape).astype(np.float32)
    output_np = np.empty(out_shape,       dtype=np.float32)

    d_text  = cuda.mem_alloc(text_np.nbytes)
    d_noise = cuda.mem_alloc(noise_np.nbytes)
    d_out   = cuda.mem_alloc(output_np.nbytes)

    bindings = [int(d_text), int(d_noise), int(d_out)]
    stream   = cuda.Stream()

    for _ in range(5):
        cuda.memcpy_htod_async(d_text,  text_np,  stream)
        cuda.memcpy_htod_async(d_noise, noise_np, stream)
        context.execute_v2(bindings)
        cuda.memcpy_dtoh_async(output_np, d_out, stream)
        stream.synchronize()

    times = []
    for _ in range(args.iters):
        start = time.time()
        cuda.memcpy_htod_async(d_text,  text_np,  stream)
        cuda.memcpy_htod_async(d_noise, noise_np, stream)
        context.execute_v2(bindings)
        cuda.memcpy_dtoh_async(output_np, d_out, stream)
        stream.synchronize()
        times.append((time.time() - start) * 1000)

    avg_latency = np.mean(times)
    print(f"TensorRT avg latency: {avg_latency:.2f} ms over {args.iters} runs")

if __name__ == "__main__":
    main()