import argparse, os, time, numpy as np, torch, onnxruntime as ort
import clip, torchvision.transforms as T

p = argparse.ArgumentParser(description="ONNX latency + CLIP")
p.add_argument("--onnx_path", required=True)
p.add_argument("--txt_dim",  type=int, required=True)
p.add_argument("--z_dim",    type=int, required=True)
p.add_argument("--batch",    type=int, default=16)
p.add_argument("--iters",    type=int, default=100)
p.add_argument("--use_cuda", action="store_true",
               help="Use CUDAExecutionProvider if available")
args = p.parse_args()

providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] \
           if args.use_cuda else ["CPUExecutionProvider"]
sess = ort.InferenceSession(args.onnx_path, providers=providers)

input_names  = [i.name for i in sess.get_inputs()]
output_name  = sess.get_outputs()[0].name

text_np  = np.random.randn(args.batch, args.txt_dim).astype(np.float32)
noise_np = np.random.randn(args.batch, args.z_dim ).astype(np.float32)

device = "cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu"
clip_model, clip_pre = clip.load("ViT-B/32", device=device)
clip_model.eval()
clip_scores = []

def clip_score(fake):
    img = (fake + 1) / 2
    img = torch.clamp(img, 0, 1)
    img = T.Resize((224, 224))(img)
    prompts = ["dummy prompt"] * fake.size(0)
    tok  = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        imf = clip_model.encode_image(img).float()
        txf = clip_model.encode_text(tok ).float()
        imf = imf / imf.norm(dim=-1, keepdim=True)
        txf = txf / txf.norm(dim=-1, keepdim=True)
    return (imf * txf).sum(dim=-1).mean().item()

sess.run([output_name], {input_names[0]: text_np, input_names[1]: noise_np})

times = []
for _ in range(args.iters):
    t0 = time.time()
    out = sess.run([output_name],
                   {input_names[0]: text_np, input_names[1]: noise_np})[0]
    times.append((time.time() - t0) * 1e3)
    fake_t = torch.from_numpy(out).to(device)
    clip_scores.append(clip_score(fake_t))

print(f"ONNX avg latency: {np.mean(times):.2f} ms over {args.iters} runs")
print(f"Avg CLIP-score  : {np.mean(clip_scores):.4f}")
