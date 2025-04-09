import argparse
import pytorch_lightning as pl
import datetime
import dateutil.tz
from clearml import Task
import mlpt.config.config as cfg
from mlpt.modules.train import GANLitModule
from mlpt.datamodules.datamodule import DeepFashionDataModule


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train GAN network on DeepFashion dataset using PyTorch Lightning")
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--train', dest='train', type=str, default="y")
    parser.add_argument('--stage', dest='stage', type=int, default=1)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    task = Task.init(project_name="DeepFashion GAN",
                     task_name="Trying ClearML")

    if args.gpu_id != '-1':
        cfg.GPU_ID = args.gpu_id
    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir

    if args.stage == 1:
        cfg.IMSIZE = 64
    else:
        cfg.IMSIZE = 256
    cfg.STAGE = args.stage
    if args.train == "y":
        cfg.TRAIN_FLAG = True
    else:
        # Пример пути к сохранённой модели
        cfg.NET_G = "../data/models/netG_epoch_360.pth"
        cfg.TRAIN_FLAG = False

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = f'./output/DeepFashion_{cfg.CONFIG_NAME}_{timestamp}'

    data_module = DeepFashionDataModule(
        data_dir=cfg.DATA_DIR,
        batch_size=cfg.TRAIN_BATCH_SIZE,
        workers=cfg.WORKERS,
        max_samples=15000
    )

    gan_module = GANLitModule(stage=cfg.STAGE, output_dir=output_dir)

    if cfg.TRAIN_FLAG:
        trainer = pl.Trainer(max_epochs=cfg.TRAIN_MAX_EPOCH,
                             accelerator="gpu", devices=1)
        trainer.fit(gan_module, datamodule=data_module)
    else:
        data_module.setup(stage='test')
        test_loader = data_module.test_dataloader()
        gan_module.sample(test_loader)
