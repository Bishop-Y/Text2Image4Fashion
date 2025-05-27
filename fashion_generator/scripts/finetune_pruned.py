import os
import torch
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from fashion_generator.modules.gan_lit_module import GANLitModule
from fashion_generator.datamodules.datamodule import DeepFashionDataModule


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def finetune_pruned(cfg: DictConfig):
    pruned_ckpt = cfg.prune.out
    out_path = cfg.finetune.out
    lr = cfg.finetune.lr
    epochs = cfg.finetune.epochs
    batch_size = cfg.finetune.batch_size

    module = GANLitModule(cfg=cfg, output_dir="./tmp_finetune")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.netG.to(device).train()

    pruned_state = torch.load(pruned_ckpt, map_location="cpu")
    module.netG.load_state_dict(pruned_state)

    dm = DeepFashionDataModule(
        data_dir=cfg.dataset.data_dir,
        batch_size=batch_size,
        workers=cfg.system.workers,
        max_samples=cfg.finetune.max_samples,
        text_dimension=cfg.text.dimension,
        imsize=cfg.gan.imsize
    )
    dm.setup()
    loader = dm.train_dataloader()

    optimizer = torch.optim.Adam(module.netG.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for sample in loader:
            imgs     = sample.image.to(device)
            txt_emb  = sample.text_embedding.to(device)

            noise = torch.randn(imgs.size(0), cfg.gan.z_dim, device=device)
            _, fake_imgs, _, _ = module.netG(txt_emb, noise)

            loss = loss_fn(fake_imgs, imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch}/{epochs} — avg MSE loss: {avg_loss:.4f}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(module.netG.state_dict(), out_path)
    print(f"Fine-tuned pruned model saved to {out_path}")

if __name__ == "__main__":
    finetune_pruned()

