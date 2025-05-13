import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

builder = trt.Builder(TRT_LOGGER)
network_flags = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(builder.create_network(network_flags), TRT_LOGGER)

onnx_path = 'output/model.onnx'
with open(onnx_path, 'rb') as f:
    parser.parse(f.read())
builder.max_batch_size = 16
builder.max_workspace_size = 1<<30
engine = builder.build_cuda_engine(parser.network)

with open('output/model.trt', 'wb') as f:
    f.write(engine.serialize())
print("TensorRT engine saved to output/model.trt")