import torch
import hydra
from omegaconf import DictConfig
from fashion_generator.modules.gan_lit_module import GANLitModule
from fashion_generator.datamodules.datamodule import DeepFashionDataModule
from torch.utils.data import DataLoader

@hydra.main(config_path="../../config", config_name="config", version_base=None)
def finetune_pruned(cfg: DictConfig):
    ckpt = cfg.prune.out
    out_path = cfg.finetune.out
    lr = cfg.finetune.lr
    epochs = cfg.finetune.epochs
    batch_size = cfg.finetune.batch_size

    module = GANLitModule(cfg=cfg, output_dir='./tmp_finetune')
    module.eval()

    state_dict = torch.load(ckpt, map_location="cpu")
    module.netG.load_state_dict(state_dict)
    module.netG.train()

    dm = DeepFashionDataModule(
        data_dir=cfg.dataset.data_dir,
        batch_size=batch_size,
        workers=cfg.system.workers,
        text_dimension=cfg.text.dimension,
        imsize=cfg.gan.imsize
    )
    dm.setup()
    loader = DataLoader(dm.train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(module.netG.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for imgs, txt_emb, _ in loader:
            device = next(module.netG.parameters()).device
            txt_emb = txt_emb.to(device)
            noise = torch.randn(len(txt_emb), cfg.gan.z_dim, device=device)
            s_out_tuple = module.netG(txt_emb, noise)
            s_img = s_out_tuple[1]
            
            loss = loss_fn(s_img, imgs.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, avg loss={epoch_loss/len(loader):.4f}")

    torch.save(module.netG.state_dict(), out_path)
    print(f"Fine‑tuned pruned model saved to {out_path}")

if __name__ == '__main__':
    finetune_pruned()
