import os
import torch
import hydra
from omegaconf import DictConfig

from fashion_generator.modules.gan_lit_module import GANLitModule

def export_onnx(model: torch.nn.Module, dummy_input, out_path):
    model.eval()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
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
    pth_path = cfg.finetune.out
    onnx_path = os.path.join(cfg.output.base_dir, "model.onnx")

    module = GANLitModule(cfg=cfg, output_dir="./tmp_onnx")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    state = torch.load(pth_path, map_location="cpu")
    module.netG.load_state_dict(state)
    module.netG.to(device)

    B = cfg.finetune.batch_size
    dummy_txt   = torch.randn(B, cfg.text.dimension, device=device)
    dummy_noise = torch.randn(B, cfg.gan.z_dim, device=device)

    export_onnx(module.netG, (dummy_txt, dummy_noise), onnx_path)


if __name__ == "__main__":
    main()
