import argparse
import torch, datetime, dateutil.tz
import torchvision.transforms as transforms

import config as cfg
from train import GANTrainer
from datasets import DeepFashionCaptionDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train GAN network on DeepFashion dataset")
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='0')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--train', dest='train', type=str, default="y")
    parser.add_argument('--stage', dest='stage', type=int, default=1)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    if args.gpu_id != -1:
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
        cfg.NET_G = "../data/models/netG_epoch_360.pth"
        cfg.TRAIN_FLAG = False

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = './output/DeepFashion_%s_%s' % (cfg.CONFIG_NAME, timestamp)


    image_transform = transforms.Compose([
        transforms.Resize((cfg.IMSIZE, cfg.IMSIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if cfg.TRAIN_FLAG:
        dataset = DeepFashionCaptionDataset(cfg.DATA_DIR, split='train', transform=image_transform, max_samples=15000)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.TRAIN_BATCH_SIZE,
            drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS)
        )
        algo = GANTrainer(output_dir)
        algo.train(data_loader, cfg.STAGE)
    else:
        image_transform = transforms.Compose([
            transforms.Resize((cfg.IMSIZE, cfg.IMSIZE)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = DeepFashionCaptionDataset(cfg.DATA_DIR, split='test', transform=image_transform, max_samples=15000)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=cfg.TRAIN_BATCH_SIZE,
            drop_last=True, shuffle=False, num_workers=int(cfg.WORKERS)
        )
        algo = GANTrainer(output_dir)
        algo.sample(dataloader, cfg.STAGE)
