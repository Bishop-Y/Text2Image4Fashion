import os
import torch
import torch.nn as nn
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


def discriminator_loss(netD, real_imgs, fake_imgs, real_labels, fake_labels, conditions, gpus):
    criterion = nn.BCELoss()
    batch_size = real_imgs.size(0)
    cond = conditions.detach()
    fake = fake_imgs.detach()
    real_features = netD(real_imgs)
    fake_features = netD(fake)
    # real пары
    inputs = (real_features, cond)
    real_logits = netD.get_cond_logits(*inputs)
    errD_real = criterion(real_logits, real_labels)
    # wrong пары
    inputs = (real_features[:(batch_size - 1)], cond[1:])
    wrong_logits = netD.get_cond_logits(*inputs)
    errD_wrong = criterion(wrong_logits, fake_labels[1:])
    # fake пары
    inputs = (fake_features, cond)
    fake_logits = netD.get_cond_logits(*inputs)
    errD_fake = criterion(fake_logits, fake_labels)

    if netD.get_uncond_logits is not None:
        real_logits = netD.get_uncond_logits(real_features)
        fake_logits = netD.get_uncond_logits(fake_features)
        uncond_errD_real = criterion(real_logits, real_labels)
        uncond_errD_fake = criterion(fake_logits, fake_labels)
        errD = ((errD_real + uncond_errD_real) / 2. +
                (errD_fake + errD_wrong + uncond_errD_fake) / 3.)
        errD_real = (errD_real + uncond_errD_real) / 2.
        errD_fake = (errD_fake + uncond_errD_fake) / 2.
    else:
        errD = errD_real + (errD_fake + errD_wrong) * 0.5
    return errD, errD_real.data, errD_wrong.data, errD_fake.data


def generator_loss(netD, fake_imgs, real_labels, conditions, gpus):
    criterion = nn.BCELoss()
    cond = conditions.detach()
    fake_features = netD(fake_imgs)
    inputs = (fake_features, cond)
    fake_logits = netD.get_cond_logits(*inputs)
    errD_fake = criterion(fake_logits, real_labels)
    if netD.get_uncond_logits is not None:
        fake_logits = netD.get_uncond_logits(fake_features)
        uncond_errD_fake = criterion(fake_logits, real_labels)
        errD_fake += uncond_errD_fake
    return errD_fake


def KL_loss(mu, logvar):
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


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
    
