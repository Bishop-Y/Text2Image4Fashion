import onnxruntime as rt
import torch
import time
import numpy as np
from omegaconf import OmegaConf

cfg = OmegaConf.load('config/config.yaml')

sess = rt.InferenceSession('output/model.onnx')

batch = 16
text = np.random.randn(batch, cfg.text.dimension).astype(np.float32)
noise = np.random.randn(batch, cfg.gan.z_dim).astype(np.float32)

for _ in range(5):
    sess.run(None, {'text_emb': text, 'noise': noise})

n_iters = 100
times = []
for _ in range(n_iters):
    start = time.time()
    sess.run(None, {'text_emb': text, 'noise': noise})
    times.append(time.time() - start)
print(f"ONNX avg latency: {np.mean(times)*1000:.2f} ms")