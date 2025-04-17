import os
import torch
import torchvision.utils as vutils


def mkdir_p(path):
    os.makedirs(path, exist_ok=True)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def save_img_results(data_img, fake, epoch, image_dir, vis_count=64):
    num = vis_count
    fake = fake[0:num]
    if data_img is not None:
        data_img = data_img[0:num]
        vutils.save_image(
            data_img, f'{image_dir}/real_samples.png',
            normalize=True)
        vutils.save_image(
            fake.data, f'{image_dir}/fake_samples_epoch_{epoch:03d}.png',
            normalize=True)
    else:
        vutils.save_image(
            fake.data, f'{image_dir}/lr_fake_samples_epoch_{epoch:03d}.png',
            normalize=True)


def save_model(netG, netD, epoch, model_dir):
    torch.save(
        netG.state_dict(),
        f'{model_dir}/netG_epoch_{epoch}.pth')
    torch.save(
        netD.state_dict(),
        f'{model_dir}/netD_epoch_last.pth')
    print('Save G/D models')
