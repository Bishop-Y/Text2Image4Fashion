import os
import torch
import time
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn as nn
import torchfile
import numpy as np
import pickle
import PIL
from PIL import Image
from utils import mkdir_p
from torch.autograd import Variable
from torchvision.models.inception import inception_v3
from scipy.stats import entropy
from torch.nn import functional as F
import torchvision.transforms as transforms
import json
import tqdm

import config as cfg
from model import Stage1_G, Stage1_D
from utils import weights_init, discriminator_loss, generator_loss, KL_loss, save_img_results, save_model
import clip

class GANTrainer(object):
    def __init__(self, output_dir):
        if cfg.TRAIN_FLAG:
            self.model_dir = os.path.join(output_dir, 'Model')
            self.image_dir = os.path.join(output_dir, 'Image')
            self.log_dir = os.path.join(output_dir, 'Log')
            mkdir_p(self.model_dir)
            mkdir_p(self.image_dir)
            mkdir_p(self.log_dir)

        self.gpus = [0]
        self.max_epoch = cfg.TRAIN_MAX_EPOCH
        self.snapshot_interval = cfg.TRAIN_SNAPSHOT_INTERVAL
        self.batch_size = cfg.TRAIN_BATCH_SIZE
        torch.cuda.device(0)
        cudnn.benchmark = True

    def get_imgs(self):
        ...

    def load_network_stageI(self):
        netG = Stage1_G()
        netG.apply(weights_init)
        print(netG)
        netD = Stage1_D()
        netD.apply(weights_init)
        print(netD)

        if cfg.NET_G != '':
            state_dict = torch.load(cfg.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_G)
        if cfg.NET_D != '':
            state_dict = torch.load(cfg.NET_D, map_location=lambda storage, loc: storage)
            netD.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_D)

        if cfg.CUDA:
            netG.cuda()
            netD.cuda()
        return netG, netD

    def load_network_stageII(self):
        from model import Stage1_G, Stage2_G, Stage2_D
        Stage1_G = Stage1_G()
        netG = Stage2_G(Stage1_G)
        netG.apply(weights_init)
        print(netG)
        if cfg.NET_G != '':
            state_dict = torch.load(cfg.NET_G, map_location=lambda storage, loc: storage)
            netG.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_G)
        elif cfg.STAGE1_G != '':
            state_dict = torch.load(cfg.STAGE1_G, map_location=lambda storage, loc: storage)
            netG.Stage1_G.load_state_dict(state_dict)
            print('Load from: ', cfg.STAGE1_G)
        else:
            print("Please give the Stage1_G path")
            return

        netD = Stage2_D()
        netD.apply(weights_init)
        if cfg.NET_D != '':
            state_dict = torch.load(cfg.NET_D, map_location=lambda storage, loc: storage)
            netD.load_state_dict(state_dict)
            print('Load from: ', cfg.NET_D)
        print(netD)

        if cfg.CUDA:
            netG.cuda()
            netD.cuda()
        return netG, netD

    def train(self, data_loader, stage=1):
        device = torch.device("cuda" if cfg.CUDA else "cpu")
        if stage == 1:
            netG, netD = self.load_network_stageI()
        else:
            netG, netD = self.load_network_stageII()

        # Инициализируем CLIP-модель для вычисления метрики
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
        clip_model.eval()

        nz = cfg.Z_DIM
        batch_size = self.batch_size
        noise = Variable(torch.FloatTensor(batch_size, nz))
        fixed_noise = Variable(torch.FloatTensor(batch_size, nz).normal_(0, 1), requires_grad=True)
        real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
        fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))

        if cfg.CUDA:
            noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
            real_labels, fake_labels = real_labels.cuda(), fake_labels.cuda()

        generator_lr = cfg.TRAIN_GENERATOR_LR
        discriminator_lr = cfg.TRAIN_DISCRIMINATOR_LR
        lr_decay_step = cfg.TRAIN_LR_DECAY_EPOCH
        optimizerD = optim.Adam(netD.parameters(), lr=cfg.TRAIN_DISCRIMINATOR_LR, betas=(0.5, 0.999))

        netG_para = []
        for p in netG.parameters():
            if p.requires_grad:
                netG_para.append(p)
        optimizerG = optim.Adam(netG_para, lr=cfg.TRAIN_GENERATOR_LR, betas=(0.5, 0.999))

        clip_log_file = os.path.join(self.log_dir, "clip_scores.txt")
        count = 0
        for epoch in range(self.max_epoch):
            start_t = time.time()
            epoch_clip_scores = []
            if epoch % lr_decay_step == 0 and epoch > 0:
                generator_lr *= 0.5
                for param_group in optimizerG.param_groups:
                    param_group['lr'] = generator_lr
                discriminator_lr *= 0.5
                for param_group in optimizerD.param_groups:
                    param_group['lr'] = discriminator_lr
            for i, data in enumerate(data_loader, 0):
                real_img_cpu, txt_embedding, prompts = data
                real_imgs = Variable(real_img_cpu)
                txt_embedding = Variable(txt_embedding)
                if cfg.CUDA:
                    real_imgs = real_imgs.cuda()
                    txt_embedding = txt_embedding.cuda()
                noise.data.normal_(0, 1)
                inputs = (txt_embedding, noise)
                _, fake_imgs, mu, logvar = nn.parallel.data_parallel(netG, inputs, self.gpus)

                netD.zero_grad()
                errD, errD_real, errD_wrong, errD_fake = discriminator_loss(netD, real_imgs, fake_imgs,
                                                                            real_labels, fake_labels, mu,
                                                                            self.gpus)
                errD.backward()
                optimizerD.step()

                netG.zero_grad()
                errG = generator_loss(netD, fake_imgs, real_labels, mu, self.gpus)
                kl_loss = KL_loss(mu, logvar)
                errG_total = errG + kl_loss * cfg.TRAIN_COEFF_KL
                errG_total.backward()
                optimizerG.step()

                count = count + 1

                if i % 100 == 0:
                    inputs = (txt_embedding, fixed_noise)
                    lr_fake, fake, _, _ = nn.parallel.data_parallel(netG, inputs, self.gpus)
                    save_img_results(real_img_cpu, fake, epoch, self.image_dir)
                    if lr_fake is not None:
                        save_img_results(None, lr_fake, epoch, self.image_dir)
                    num = cfg.VIS_COUNT
                    prompts_to_save = min(len(prompts), num)
                    prompts_filename = f"{self.image_dir}/fake_samples_epoch_{epoch:03d}_prompts.txt"
                    with open(prompts_filename, 'w', encoding='utf-8') as f:
                        for j in range(prompts_to_save):
                            f.write(f"[{j}] {prompts[j]}\n")

                    fake_img_sample = fake_imgs[0].detach().cpu()
                    fake_img_sample = (fake_img_sample + 1.0) / 2.0  # перевод в диапазон [0,1]
                    pil_img = transforms.ToPILImage()(fake_img_sample)
                    prompt_text = prompts[0][:75]
                    text_tokens = clip.tokenize([prompt_text]).to(device)
                    with torch.no_grad():
                        clip_input = clip_preprocess(pil_img).unsqueeze(0).to(device)
                        image_features = clip_model.encode_image(clip_input)
                        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                        text_features = clip_model.encode_text(text_tokens)
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                        clip_score = F.cosine_similarity(image_features, text_features, dim=-1)
                        clip_score_val = clip_score.item()
                    print(f"Epoch {epoch}, Iteration {i} - CLIP Score: {clip_score_val:.4f}")
                    epoch_clip_scores.append(clip_score_val)
                    # -----------------------------------------------------------------------------------

                end_t = time.time()
                print('''[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f Loss_KL: %.4f
                                     Loss_real: %.4f Loss_wrong:%.4f Loss_fake: %.4f
                                     Total Time: %.2fsec
                                  '''
                      % (epoch, self.max_epoch, i, len(data_loader),
                         errD.data, errG.data, kl_loss.data,
                         errD_real, errD_wrong, errD_fake, (end_t - start_t)))
                if epoch % self.snapshot_interval == 0:
                    save_model(netG, netD, epoch, self.model_dir)
            save_model(netG, netD, self.max_epoch, self.model_dir)
            
            avg_clip = sum(epoch_clip_scores) / len(epoch_clip_scores) if epoch_clip_scores else 0.0
            with open(clip_log_file, 'a', encoding='utf-8') as log_f:
                log_f.write(f"Epoch {epoch}: Average CLIP Score = {avg_clip:.4f}\n")
            print(f"Epoch {epoch} - Average CLIP Score logged: {avg_clip:.4f}")

    def inception_score(self, dataloader, cuda=True, batch_size=32, resize=False, splits=1):
        """Computes the inception score of the generated images.
        imgs -- Torch dataset (list) of (3 x H x W) numpy images normalized in the range [0, 1]
        cuda -- whether or not to run on GPU
        batch_size -- batch size for feeding into Inception v3
        splits -- number of splits
        """
        imgs = self.get_imgs()
        N = len(imgs)
        if cuda:
            dtype = torch.cuda.FloatTensor
        else:
            if torch.cuda.is_available():
                print("WARNING: You have a CUDA device, so you should probably set cuda=True")
            dtype = torch.FloatTensor

        dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)
        inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
        inception_model.eval()
        up = nn.Upsample(size=(299, 299), mode='bilinear', align_corners=True).type(dtype)

        def get_pred(x):
            if resize:
                x = up(x)
            x = inception_model(x)
            return F.softmax(x, dim=0).data.cpu().numpy()

        preds = np.zeros((N, 1000))
        start_idx = 0
        for batch in tqdm.tqdm(dataloader):
            batch = batch.type(dtype)
            batch_size_i = batch.size(0)
            preds[start_idx:start_idx+batch_size_i] = get_pred(batch)
            start_idx += batch_size_i

        split_scores = []
        for k in tqdm.tqdm(range(splits)):
            part = preds[k * (N // splits): (k + 1) * (N // splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))
        return np.mean(split_scores), np.std(split_scores)

    def sample(self, data_loader, stage=1):
        if stage == 1:
            netG, _ = self.load_network_stageI()
        else:
            netG, _ = self.load_network_stageII()
        netG.eval()

        save_dir = cfg.NET_G[:cfg.NET_G.find('.pth')]
        if not os.path.isdir(save_dir):
            mkdir_p(save_dir)
        
        nz = cfg.Z_DIM
        batch_size = self.batch_size
        noise = Variable(torch.FloatTensor(batch_size, nz))
        
        count = 0
        for i, data in enumerate(data_loader, 0):
            real_img_cpu, txt_embedding, prompts = data
            txt_embedding = Variable(txt_embedding)
            if cfg.CUDA:
                txt_embedding = txt_embedding.cuda()
            noise.data.normal_(0, 1)
            inputs = (txt_embedding, noise)
            _, fake_imgs, mu, logvar = nn.parallel.data_parallel(netG, inputs, self.gpus)
            for j in range(batch_size):
                save_name = '%s/%d.png' % (save_dir, count + j)
                im = fake_imgs[j].data.cpu().numpy()
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                im = Image.fromarray(im)
                im.save(save_name)
                prompt_save_name = '%s/%d.txt' % (save_dir, count + j)
                with open(prompt_save_name, 'w', encoding='utf-8') as f:
                    f.write(prompts[j])
            count += batch_size

