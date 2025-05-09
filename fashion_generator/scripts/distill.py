import copy
import torch
import hydra
from omegaconf import DictConfig
from fashion_generator.modules.gan_lit_module import GANLitModule
from fashion_generator.datamodules.datamodule import DeepFashionDataModule
import builtins


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def distill(cfg: DictConfig):
    GANLitModule.print = lambda self, * \
        args, **kwargs: builtins.print(*args, **kwargs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    teacher_module = GANLitModule(cfg=cfg, output_dir='./tmp_distill')
    teacher_gen = teacher_module.netG.eval().to(device)

    # Копируем cfg и уменьшаем в 2 раза базовый размер фильтров
    student_cfg = copy.deepcopy(cfg)
    student_cfg.gan.gf_dim = max(1, cfg.gan.gf_dim // 2)
    student_cfg.model.net_g = ""
    student_module = GANLitModule(
        cfg=student_cfg, output_dir='./tmp_distill_student')
    student_gen = student_module.netG.train().to(device)

    dm = DeepFashionDataModule(
        data_dir=cfg.dataset.data_dir,
        batch_size=cfg.distill.batch_size,
        workers=cfg.system.workers,
        max_samples=15000,
        text_dimension=cfg.text.dimension,
        imsize=cfg.gan.imsize
    )
    dm.setup()
    loader = dm.train_dataloader()

    optimizer = torch.optim.Adam(
        student_gen.parameters(), lr=cfg.distill.lr, betas=(0.5, 0.999))
    loss_fn = torch.nn.MSELoss()
    lam_inter = getattr(cfg.distill, 'intermediate_weight', 1.0)

    # Захват промежуточных фичей
    teacher_feats = {}
    student_feats = {}
    layers_to_match = ['upsample2', 'upsample3']

    def make_hook(name, storage):
        def hook(module, inp, out):
            storage[name] = out
        return hook

    for name in layers_to_match:
        getattr(teacher_gen, name).register_forward_hook(
            make_hook(name, teacher_feats))
        getattr(student_gen, name).register_forward_hook(
            make_hook(name, student_feats))

    epochs = cfg.distill.epochs
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for batch in loader:
            imgs = batch.image.to(device)
            txt_emb = batch.text_embedding.to(device)

            noise = torch.randn(imgs.size(0), cfg.gan.z_dim, device=device)

            # Очищаем предыдущие фичи
            teacher_feats.clear()
            student_feats.clear()

            with torch.no_grad():
                _, t_img, _, _ = teacher_gen(txt_emb, noise)

            _, s_img, _, _ = student_gen(txt_emb, noise)

            # Loss по финальным изображениям
            loss_img = loss_fn(s_img, t_img)

            # Loss по промежуточным картам
            loss_mid = 0.0
            for name in layers_to_match:
                t_map = teacher_feats[name].mean(dim=1, keepdim=True)
                s_map = student_feats[name].mean(dim=1, keepdim=True)
                loss_mid += loss_fn(s_map, t_map)

            # Суммарный loss
            loss = loss_img + lam_inter * loss_mid

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch}/{epochs} — total_loss={avg_loss:.4f} "
              f"(img={loss_img.item():.4f}, mid={loss_mid.item():.4f})")

    torch.save(student_gen.state_dict(), cfg.distill.out)
    print(f"Distilled student saved to {cfg.distill.out}")


if __name__ == '__main__':
    distill()
