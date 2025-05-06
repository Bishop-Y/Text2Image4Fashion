import torch
import hydra
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from fashion_generator.modules.gan_lit_module import GANLitModule
from fashion_generator.datamodules.datamodule import DeepFashionDataModule
import builtins

@hydra.main(config_path="../../config", config_name="config", version_base=None)
def distill(cfg: DictConfig):
    GANLitModule.print = lambda self, *args, **kwargs: builtins.print(*args, **kwargs)
    ckpt = cfg.model.net_g
    epochs = cfg.distill.epochs
    lr = cfg.distill.lr
    batch_size = cfg.distill.batch_size
    out_path = cfg.distill.out

    # Teacher
    teacher = GANLitModule(cfg=cfg, output_dir='./tmp_distill')
    teacher.eval()

    # Student
    student = GANLitModule(cfg=cfg, output_dir='./tmp_distill_student')
    student.train()

    # DataModule
    dm = DeepFashionDataModule(
        data_dir=cfg.dataset.data_dir,
        batch_size=batch_size,
        workers=cfg.system.workers,
        max_samples=15000,
        text_dimension=cfg.text.dimension,
        imsize=cfg.gan.imsize
    )
    dm.setup()
    loader = DataLoader(dm.train_dataset, batch_size=batch_size)

    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        for imgs, txt_emb, _ in loader:
            device = next(teacher.netG.parameters()).device
            txt_emb = txt_emb.to(device)
            noise = torch.randn(len(txt_emb), cfg.gan.z_dim, device=device)

            with torch.no_grad():
                t_out_tuple = teacher.netG(txt_emb, noise)
                t_img = t_out_tuple[1]
            s_out_tuple = student.netG(txt_emb, noise)
            s_img = s_out_tuple[1]

            loss = loss_fn(s_img, t_img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, loss={loss.item():.4f}")

    # Сохранение student.weights
    torch.save(student.state_dict(), out_path)
    print(f"Distilled model saved to {out_path}")

if __name__ == '__main__':
    distill()