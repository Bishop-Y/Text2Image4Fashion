import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

import pytorch_lightning as pl
import clip
import torchvision.transforms as transforms
from clearml import Task


from fashion_generator.models.model import Stage1_G
from fashion_generator.models.model import Stage1_D
from fashion_generator.models.model import Stage2_G
from fashion_generator.models.model import Stage2_D
from fashion_generator.utils.utils import mkdir_p
from fashion_generator.utils.utils import weights_init
from fashion_generator.utils.utils import save_model
from fashion_generator.utils.utils import save_img_results
from fashion_generator.utils.losses import discriminator_loss
from fashion_generator.utils.losses import generator_loss
from fashion_generator.utils.losses import KL_loss
from fashion_generator.datamodules.datasets import DeepFashionSample


class GANLitModule(pl.LightningModule):
    def __init__(self, cfg, output_dir='./output', **kwargs):
        super(GANLitModule, self).__init__()
        self.automatic_optimization = False
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg
        self.output_dir = output_dir
        self.log_dir = os.path.join(output_dir, 'Log')
        self.model_dir = os.path.join(output_dir, 'Model')
        self.image_dir = os.path.join(output_dir, 'Image')
        mkdir_p(self.log_dir)
        mkdir_p(self.model_dir)
        mkdir_p(self.image_dir)
        task = Task.current_task()
        if task is None:
            # Offline или нет инициализации ClearML — просто заглушка
            self.cl_task = None
            self.cl_logger = None
        else:
            self.cl_task = task
            self.cl_logger = task.get_logger()



        self.batch_size = self.cfg.training.batch_size
        self.snapshot_interval = self.cfg.training.snapshot_interval
        self.gpus = [0]

        if self.cfg.gan.stage == 1:
            self.netG, self.netD = self.load_network_stageI()
        else:
            self.netG, self.netD = self.load_network_stageII()

        self.clip_model, self.clip_preprocess = clip.load(
            "ViT-B/32", device=self.device)
        self.clip_model.eval()

        self.register_buffer("fixed_noise", torch.randn(
            self.batch_size, self.cfg.gan.z_dim))

        self.epoch_clip_scores = []
        self.epoch_losses_g    = []
        self.epoch_losses_d    = []
        self.epoch_kl_losses   = []
        self.epoch_losses_g_tot= []

    def load_network_stageI(self):
        netG = Stage1_G(self.cfg.gan, self.cfg.text)
        netG.apply(weights_init)
        netD = Stage1_D(self.cfg.gan)
        netD.apply(weights_init)

        if self.cfg.model.net_g != "":
            state_dict = torch.load(self.cfg.model.net_g, map_location="cpu")
            netG.load_state_dict(state_dict)
            self.print("Load StageI_G from: " + self.cfg.model.net_g)
        if self.cfg.model.net_d != "":
            state_dict = torch.load(self.cfg.model.net_d, map_location="cpu")
            netD.load_state_dict(state_dict)
            self.print("Load StageI_D from: " + self.cfg.model.net_d)

        if self.cfg.system.cuda:
            netG.cuda()
            netD.cuda()
        return netG, netD

    def load_network_stageII(self):
        base_stage1 = Stage1_G(self.cfg.gan, self.cfg.text)
        netG = Stage2_G(base_stage1, self.cfg.gan, self.cfg.text)
        netG.apply(weights_init)
        if self.cfg.model.net_g != "":
            state_dict = torch.load(self.cfg.model.net_g, map_location="cpu")
            netG.load_state_dict(state_dict)
            self.print("Load StageII_G from: " + self.cfg.model.net_g)
        elif self.cfg.model.stage1_g != "":
            state_dict = torch.load(
                self.cfg.model.stage1_g, map_location="cpu")
            netG.Stage1_G.load_state_dict(state_dict)
            self.print("Load Stage1_G from: " + self.cfg.model.stage1_g)
        else:
            self.print("Please provide the Stage1_G path")

        netD = Stage2_D(self.cfg.gan)
        netD.apply(weights_init)
        if self.cfg.model.net_d != "":
            state_dict = torch.load(self.cfg.model.net_d, map_location="cpu")
            netD.load_state_dict(state_dict)
            self.print("Load StageII_D from: " + self.cfg.model.net_d)

        if self.cfg.system.cuda:
            netG.cuda()
            netD.cuda()
        return netG, netD

    def configure_optimizers(self):
        optimizerD = torch.optim.Adam(self.netD.parameters(),
                                      lr=self.cfg.training.discriminator_lr,
                                      betas=(0.5, 0.999))
        netG_params = [p for p in self.netG.parameters() if p.requires_grad]
        optimizerG = torch.optim.Adam(netG_params,
                                      lr=self.cfg.training.generator_lr,
                                      betas=(0.5, 0.999))

        schedulerD = torch.optim.lr_scheduler.StepLR(
            optimizerD,
            step_size=self.cfg.training.lr_decay_epoch,
            gamma=0.5
        )
        schedulerG = torch.optim.lr_scheduler.StepLR(
            optimizerG,
            step_size=self.cfg.training.lr_decay_epoch,
            gamma=0.5
        )
        return [optimizerD, optimizerG], [schedulerD, schedulerG]

    def on_train_epoch_start(self):
        self.epoch_clip_scores = []

    def training_step(self, batch: DeepFashionSample, batch_idx):
        device = self.device
        real_imgs = batch.image.to(device)
        txt_embedding = batch.text_embedding.to(device)
        prompts = batch.prompt
        batch_size = real_imgs.size(0)

        noise = torch.randn(batch_size, self.cfg.gan.z_dim, device=device)
        real_labels = torch.ones(batch_size, device=device)
        fake_labels = torch.zeros(batch_size, device=device)

        opt_d, opt_g = self.optimizers()

        opt_d.zero_grad()
        # Generate fake images, detaching gradients for the discriminator
        _, fake_imgs, mu, logvar = self.netG(txt_embedding, noise)
        errD, errD_real, errD_wrong, errD_fake = discriminator_loss(
            self.netD, real_imgs, fake_imgs.detach(),
            real_labels, fake_labels, mu, self.gpus
        )
        self.manual_backward(errD)
        opt_d.step()

        opt_g.zero_grad()
        _, fake_imgs, mu, logvar = self.netG(txt_embedding, noise)
        errG = generator_loss(self.netD, fake_imgs, real_labels, mu, self.gpus)
        kl_loss = KL_loss(mu, logvar)
        errG_total = errG + kl_loss * self.cfg.training.coeff_kl
        self.manual_backward(errG_total)
        opt_g.step()

        self.epoch_losses_d.append(errD.item())
        self.epoch_losses_g.append(errG.item())
        self.epoch_kl_losses.append(kl_loss.item())
        self.epoch_losses_g_tot.append(errG_total.item())


        self.log("Loss_D_step",      errD,      on_step=True,  on_epoch=False, prog_bar=True)
        self.log("Loss_G_step",      errG,      on_step=True,  on_epoch=False, prog_bar=True)
        self.log("KL_loss_step",     kl_loss,   on_step=True,  on_epoch=False, prog_bar=True)
        self.log("Loss_G_total_step", errG_total, on_step=True,  on_epoch=False, prog_bar=True)


        self.cl_logger.report_scalar("Loss_D_step",      "value", errD.item(),      self.global_step)
        self.cl_logger.report_scalar("Loss_G_step",      "value", errG.item(),      self.global_step)
        self.cl_logger.report_scalar("KL_loss_step",     "value", kl_loss.item(),   self.global_step)
        self.cl_logger.report_scalar("Loss_G_total_step","value", errG_total.item(), self.global_step)
        if batch_idx % 100 == 0:
            with torch.no_grad():
                lr_fake, fake_fixed, _, _ = self.netG(
                    txt_embedding, self.fixed_noise[:batch_size])

            save_img_results(real_imgs.detach().cpu(), fake_fixed, self.current_epoch, self.image_dir,
                             vis_count=self.cfg.gan.vis_count)
            if lr_fake is not None:
                save_img_results(None, lr_fake, self.current_epoch, self.image_dir,
                                 vis_count=self.cfg.gan.vis_count)

            num = self.cfg.gan.vis_count
            prompts_to_save = min(len(prompts), num)
            prompts_filename = os.path.join(
                self.image_dir,
                f"fake_samples_epoch_{self.current_epoch:03d}_prompts.txt"
            )
            with open(prompts_filename, 'w', encoding='utf-8') as f:
                for j in range(prompts_to_save):
                    f.write(f"[{j}] {prompts[j]}\n")

            fake_img_sample = fake_fixed[0].detach().cpu()
            fake_img_sample = (fake_img_sample + 1.0) / 2.0
            pil_img = transforms.ToPILImage()(fake_img_sample)
            prompt_text = prompts[0][:75]
            text_tokens = clip.tokenize([prompt_text]).to(device)
            fake_img_processed = self.clip_preprocess(
                pil_img).unsqueeze(0).to(device)

            image_features = self.clip_model.encode_image(fake_img_processed)
            image_features = image_features / \
                image_features.norm(dim=-1, keepdim=True)
            text_features = self.clip_model.encode_text(text_tokens)
            text_features = text_features / \
                text_features.norm(dim=-1, keepdim=True)
            clip_score = F.cosine_similarity(
                image_features, text_features, dim=-1)
            clip_score_val = clip_score.item()

            self.log("CLIP_score_step", clip_score_val, on_step=True, on_epoch=False, prog_bar=True)
            self.cl_logger.report_scalar("CLIP_score_step","value", clip_score_val, self.global_step)
            self.epoch_clip_scores.append(clip_score_val)

            self.epoch_clip_scores.append(clip_score_val)
            self.print(
                f"Epoch {self.current_epoch}, batch {batch_idx} - CLIP Score: {clip_score_val:.4f}")

        return errG_total

    def on_train_epoch_end(self):
        epoch = self.current_epoch
        if epoch % self.snapshot_interval == 0:
            save_model(self.netG, self.netD, epoch, self.model_dir)

        # Calculating mean CLIP‑score per epoch
        if len(self.epoch_clip_scores) > 0:
            avg_clip = sum(self.epoch_clip_scores) / \
                len(self.epoch_clip_scores)
        else:
            avg_clip = 0.0

        clip_log_file = os.path.join(self.log_dir, "clip_scores.txt")
        with open(clip_log_file, 'a', encoding='utf-8') as log_f:
            log_f.write(
                f"Epoch {epoch}: Snapshot saved, Average CLIP Score = {avg_clip:.4f}\n")

        self.print(
            f"Epoch {epoch} - Model snapshot saved. Average CLIP Score: {avg_clip:.4f}")

        # --- эпоховые логи в PL ---
        epoch = self.current_epoch

        # средние значения
        avg_loss_d    = np.mean(self.epoch_losses_d)
        avg_loss_g    = np.mean(self.epoch_losses_g)
        avg_kl        = np.mean(self.epoch_kl_losses)
        avg_loss_g_tot= np.mean(self.epoch_losses_g_tot)
        avg_clip      = np.mean(self.epoch_clip_scores) if self.epoch_clip_scores else 0.0

        # PL-лог
        self.log("Loss_D_epoch",      avg_loss_d,      on_epoch=True)
        self.log("Loss_G_epoch",      avg_loss_g,      on_epoch=True)
        self.log("KL_loss_epoch",     avg_kl,          on_epoch=True)
        self.log("Loss_G_total_epoch",avg_loss_g_tot,  on_epoch=True)
        self.log("CLIP_score_epoch",  avg_clip,        on_epoch=True)

        # ClearML-лог
        self.cl_logger.report_scalar("Loss_D_epoch",      "value", avg_loss_d,      epoch)
        self.cl_logger.report_scalar("Loss_G_epoch",      "value", avg_loss_g,      epoch)
        self.cl_logger.report_scalar("KL_loss_epoch",     "value", avg_kl,          epoch)
        self.cl_logger.report_scalar("Loss_G_total_epoch","value", avg_loss_g_tot,  epoch)
        self.cl_logger.report_scalar("CLIP_score_epoch",  "value", avg_clip,        epoch)


    def sample(self, data_loader):
        self.netG.eval()
        if (".pth" in self.cfg.model.net_g) and (self.cfg.model.net_g.find(".pth") != -1):
            save_dir = self.cfg.model.net_g[:self.cfg.model.net_g.find(".pth")]
        else:
            save_dir = "./sample_output"

        if not os.path.isdir(save_dir):
            mkdir_p(save_dir)

        device = self.device
        nz = self.cfg.gan.z_dim
        count = 0
        for sample in data_loader:
            txt_embedding = sample.text_embedding.to(device)
            prompts = sample.prompt
            batch_size = txt_embedding.size(0)
            noise = torch.randn(batch_size, nz, device=device)

            _, fake_imgs, mu, logvar = self.netG(txt_embedding, noise)
            for j in range(batch_size):
                save_name = os.path.join(save_dir, f"{count + j}.png")
                im = fake_imgs[j].detach().cpu().numpy()
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                im = Image.fromarray(im)
                im.save(save_name)

                prompt_save_name = os.path.join(save_dir, f"{count + j}.txt")
                with open(prompt_save_name, 'w', encoding='utf-8') as f:
                    f.write(prompts[j])

            count += batch_size

        self.print("Sampling completed.")
