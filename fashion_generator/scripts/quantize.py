import torch
import hydra
from omegaconf import DictConfig
from fashion_generator.modules.gan_lit_module import GANLitModule

@hydra.main(config_path="../../config", config_name="config", version_base=None)
def quantize(cfg: DictConfig):
    out_path = 'weights/quantize/quantized_gen.pth'
    dtype = 'qint8'

    lit = GANLitModule(cfg=cfg, output_dir='./tmp_quant')
    lit.eval()

    # Dynamic quantization
    gen = lit.netG
    gen_q = torch.quantization.quantize_dynamic(
        gen,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=getattr(torch, dtype)
    )

    torch.save(gen_q.state_dict(), out_path)
    print(f"Quantized model saved to {out_path}")

if __name__ == '__main__':
    quantize()