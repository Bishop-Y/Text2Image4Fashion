import torch
import hydra
from omegaconf import DictConfig
from fashion_generator.modules.gan_lit_module import GANLitModule

def export_onnx(model: torch.nn.Module, dummy_input, out_path):
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["text_emb", "noise"],
        output_names=["fake_image"],
        dynamic_axes={
            "text_emb": {0: "batch"},
            "noise": {0: "batch"},
            "fake_image": {0: "batch"}
        }
    )
    print(f"ONNX model saved to {out_path}")

@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    ckpt = cfg.finetune.out
    model = GANLitModule.load_from_checkpoint(ckpt, cfg=cfg)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.netG.to(device)

    dummy_txt = torch.randn(cfg.finetune.batch_size, cfg.text.dimension, device=device)
    dummy_noise = torch.randn(cfg.finetune.batch_size, cfg.gan.z_dim, device=device)
    export_onnx(model.netG, (dummy_txt, dummy_noise), cfg.output.base_dir + "/model.onnx")

if __name__ == '__main__':
    main()