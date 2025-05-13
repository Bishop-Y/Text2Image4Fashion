import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import time

TRT_LOGGER = trt.Logger()
with open('output/model.trt', 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
context = engine.create_execution_context()

batch = 16
input_shape = (batch, ) + engine.get_binding_shape(0)[1:]
output_shape = (batch, ) + engine.get_binding_shape(1)[1:]

text = np.random.randn(*input_shape).astype(np.float32)
noise = np.random.randn(*input_shape).astype(np.float32)
output = np.empty(output_shape, dtype=np.float32)

d_input = cuda.mem_alloc(text.nbytes)
 d_noise = cuda.mem_alloc(noise.nbytes)
 d_output = cuda.mem_alloc(output.nbytes)

bindings = [int(d_input), int(d_noise), int(d_output)]
stream = cuda.Stream()

for _ in range(5):
    cuda.memcpy_htod_async(d_input, text, stream)
    cuda.memcpy_htod_async(d_noise, noise, stream)
    context.execute_async_v2(bindings, stream.handle)
    cuda.memcpy_dtoh_async(output, d_output, stream)
    stream.synchronize()

times = []
for _ in range(100):
    start = time.time()
    cuda.memcpy_htod_async(d_input, text, stream)
    cuda.memcpy_htod_async(d_noise, noise, stream)
    context.execute_async_v2(bindings, stream.handle)
    cuda.memcpy_dtoh_async(output, d_output, stream)
    stream.synchronize()
    times.append(time.time() - start)
print(f"TensorRT avg latency: {np.mean(times)*1000:.2f} ms")