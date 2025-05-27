import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time
import argparse

def infer_tensorrt(engine_path, batch_size, n_iters, text_dim, z_dim, imsize):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, 'rb') as f:
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

    try:
        context.set_binding_shape(0, (batch_size, text_dim))
        context.set_binding_shape(1, (batch_size, z_dim))
    except Exception as e:
        print(f"Warning: could not set binding shapes: {e}")

    text = np.random.randn(batch_size, text_dim).astype(np.float32)
    noise = np.random.randn(batch_size, z_dim).astype(np.float32)
    output = np.empty((batch_size, 3, imsize, imsize), dtype=np.float32)

    d_text = cuda.mem_alloc(text.nbytes)
    d_noise = cuda.mem_alloc(noise.nbytes)
    d_output = cuda.mem_alloc(output.nbytes)

    bindings = [int(d_text), int(d_noise), int(d_output)]

    for _ in range(5):
        cuda.memcpy_htod(d_text, text)
        cuda.memcpy_htod(d_noise, noise)
        context.execute_v2(bindings)
        cuda.memcpy_dtoh(output, d_output)

    times = []
    for _ in range(n_iters):
        start = time.time()
        cuda.memcpy_htod(d_text, text)
        cuda.memcpy_htod(d_noise, noise)
        context.execute_v2(bindings)
        cuda.memcpy_dtoh(output, d_output)
        times.append(time.time() - start)

    avg_latency = np.mean(times) * 1000
    print(f"TensorRT avg latency: {avg_latency:.2f} ms over {n_iters} runs")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on TensorRT engine')
    parser.add_argument('--engine_path', type=str, default='output/model.trt',
                        help='Path to TensorRT engine file')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference')
    parser.add_argument('--n_iters', type=int, default=100,
                        help='Number of iterations for latency measurement')
    parser.add_argument('--text_dim', type=int, default=1024,
                        help='Dimension of text embedding')
    parser.add_argument('--z_dim', type=int, default=100,
                        help='Dimension of noise vector')
    parser.add_argument('--imsize', type=int, default=64,
                        help='Height/width of generated images')

    args = parser.parse_args()
    infer_tensorrt(
        args.engine_path,
        args.batch_size,
        args.n_iters,
        args.text_dim,
        args.z_dim,
        args.imsize
    )
