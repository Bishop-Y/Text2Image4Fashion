import datetime
import dateutil.tz
import pytorch_lightning as pl
from clearml import Task
import hydra
from omegaconf import DictConfig, OmegaConf

from mlpt.modules.train import GANLitModule
from mlpt.datamodules.datamodule import DeepFashionDataModule


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Выводим конфигурацию
    print(OmegaConf.to_yaml(cfg))

    task = Task.init(project_name="DeepFashion GAN",
                     task_name="Trying ClearML Hydra")

    cfg.gan.imsize = 64 if cfg.gan.stage == 1 else 256

    if not cfg.training.flag:
        cfg.model.net_g = "./output/models/netG_epoch_360.pth"

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = f'./output/DeepFashion_{cfg.dataset.name}_{timestamp}'

    data_module = DeepFashionDataModule(
        data_dir=cfg.dataset.data_dir,
        batch_size=cfg.training.batch_size,
        workers=cfg.system.workers,
        max_samples=15000,
        text_dimension=cfg.text.dimension,
        imsize=cfg.gan.imsize
    )

    gan_module = GANLitModule(cfg=cfg, output_dir=output_dir)

    if cfg.training.flag:
        trainer = pl.Trainer(max_epochs=cfg.training.max_epoch,
                             accelerator="gpu", devices=1)
        trainer.fit(gan_module, datamodule=data_module)
    else:
        data_module.setup(stage='test')
        test_loader = data_module.test_dataloader()
        gan_module.sample(test_loader)


if __name__ == "__main__":
    main()
