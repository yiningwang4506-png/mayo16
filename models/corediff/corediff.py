import os.path as osp
from torch.nn import functional as F
import torch
import torch.nn as nn
import torchvision
import argparse
import tqdm
import copy
import numpy as np
from utils.measure import *
from utils.loss_function import PerceptualLoss
from utils.ema import EMA

from models.basic_template import TrainTask
from .corediff_wrapper import Network, WeightNet
from .diffusion_modules import Diffusion

import wandb


class corediff(TrainTask):
    @staticmethod
    def build_options():
        parser = argparse.ArgumentParser('Private arguments for training of different methods')
        parser.add_argument("--in_channels", default=3, type=int)
        parser.add_argument("--out_channels", default=1, type=int)
        parser.add_argument("--init_lr", default=2e-4, type=float)

        parser.add_argument('--update_ema_iter', default=10, type=int)
        parser.add_argument('--start_ema_iter', default=2000, type=int)
        parser.add_argument('--ema_decay', default=0.995, type=float)

        parser.add_argument('--T', default=10, type=int)

        parser.add_argument('--sampling_routine', default='ddim', type=str)
        parser.add_argument('--only_adjust_two_step', action='store_true')
        parser.add_argument('--start_adjust_iter', default=1, type=int)
        
        # DRL parameters
        parser.add_argument('--reg_max', default=18, type=int,
                          help='Number of discrete bins for DRL (default: 18, total 19 bins)')
        parser.add_argument('--y_0', default=-160.0, type=float,
                          help='Minimum HU value for DRL range')
        parser.add_argument('--y_n', default=200.0, type=float,
                          help='Maximum HU value for DRL range')
        parser.add_argument('--norm_range_max', default=3072.0, type=float,
                          help='Maximum value for normalization')
        parser.add_argument('--norm_range_min', default=-1024.0, type=float,
                          help='Minimum value for normalization')
        parser.add_argument('--use_dfl_loss', action='store_true',
                          help='Use Distribution Focal Loss')
        parser.add_argument('--dfl_weight', default=0.2, type=float,
                          help='Weight for DFL loss')
        
        return parser

    def set_model(self):
        opt = self.opt
        self.ema = EMA(opt.ema_decay)
        self.update_ema_iter = opt.update_ema_iter
        self.start_ema_iter = opt.start_ema_iter
        self.dose = opt.dose
        self.T = opt.T
        self.sampling_routine = opt.sampling_routine
        self.context = opt.context
        
        # 🔥 使用 dose_emb_dim 而不是 text_emb_dim
        denoise_fn = Network(
            in_channels=opt.in_channels, 
            context=opt.context,
            reg_max=opt.reg_max,
            y_0=opt.y_0,
            y_n=opt.y_n,
            norm_range_max=opt.norm_range_max,
            norm_range_min=opt.norm_range_min,
            dose_emb_dim=256  # Dose embedding 维度
        )

        model = Diffusion(
            denoise_fn=denoise_fn,
            image_size=512,
            timesteps=opt.T,
            context=opt.context
        ).cuda()
    
        optimizer = torch.optim.Adam(model.parameters(), opt.init_lr)
        ema_model = copy.deepcopy(model)

        self.logger.modules = [model, ema_model, optimizer]
        self.model = model
        self.optimizer = optimizer
        self.ema_model = ema_model

        self.lossfn = nn.MSELoss()
        self.lossfn_sub1 = nn.MSELoss()
        
        # 🔥 检查是否使用 dose condition
        self.use_dose_condition = getattr(opt, 'use_text_condition', False)
        if self.use_dose_condition:
            print("✅ Dose Embedding conditioning ENABLED")
        else:
            print("✅ Standard training mode (no dose condition)")
        
        # DRL parameters for DFL loss
        self.use_dfl_loss = opt.use_dfl_loss
        self.dfl_weight = opt.dfl_weight
        self.reg_max = opt.reg_max
        self.hu_interval = denoise_fn.hu_interval
        self.y_0 = opt.y_0
        self.norm_range_max = opt.norm_range_max
        self.norm_range_min = opt.norm_range_min

        self.reset_parameters()
    

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())


    def step_ema(self, n_iter):
        if n_iter < self.start_ema_iter:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    
    def compute_dfl_loss(self, out_dist, y, x):
        """
        计算Distribution Focal Loss (完全保持不变)
        """
        x_HU = x * 4096.0 - 1024.0
        y_HU = y * 4096.0 - 1024.0
        residual_HU = y_HU - x_HU
        
        bin_idx_raw = (residual_HU - self.y_0) / self.hu_interval
        bin_idx = torch.clamp(bin_idx_raw, 0, self.reg_max)
        
        B, C, H, W = bin_idx.shape
        bin_idx_flat = bin_idx.reshape(-1)
        
        left = bin_idx_flat.floor().long()
        right = torch.clamp(left + 1, max=self.reg_max)
        weight_right = bin_idx_flat - left.float()
        
        target_dist_flat = torch.zeros(bin_idx_flat.shape[0], self.reg_max + 1, device=bin_idx.device)
        target_dist_flat.scatter_(1, left.unsqueeze(1), (1 - weight_right).unsqueeze(1))
        target_dist_flat.scatter_(1, right.unsqueeze(1), weight_right.unsqueeze(1))
        
        pred_dist = torch.softmax(out_dist, dim=1)
        pred_dist_flat = pred_dist.permute(0, 2, 3, 1).reshape(-1, self.reg_max + 1)
        
        B_out, C_out, H_out, W_out = out_dist.shape
        out_dist_flat = out_dist.permute(0, 2, 3, 1).reshape(-1, self.reg_max + 1)
        
        loss = -(target_dist_flat * torch.log_softmax(out_dist_flat, dim=-1)).sum(-1).mean()
        loss = loss * 0.0001
        
        return loss

    def train(self, inputs, n_iter):
        opt = self.opt
        self.model.train()
        self.ema_model.train()
        
        # 🔥 兼容新旧数据格式
        if isinstance(inputs, dict):
            # 新格式（带dose的dict）
            low_dose = inputs['input'].cuda()
            full_dose = inputs['target'].cuda()
            
            # 🔥 获取 dose 值
            if self.use_dose_condition and 'dose' in inputs:
                dose = inputs['dose'].float().cuda()  # [B]
            else:
                dose = None
        else:
            # 旧格式（向后兼容）
            low_dose, full_dose = inputs
            low_dose, full_dose = low_dose.cuda(), full_dose.cuda()
            dose = None

        # 🔥 传入 dose 参数
        gen_full_dose, x_mix, gen_full_dose_sub1, x_mix_sub1, out_dist_1, out_dist_2 = self.model(
            low_dose, full_dose, n_iter,
            only_adjust_two_step=opt.only_adjust_two_step,
            start_adjust_iter=opt.start_adjust_iter,
            dose=dose
        )

        # MSE loss
        loss_mse = 0.5 * self.lossfn(gen_full_dose, full_dose) + \
                   0.5 * self.lossfn_sub1(gen_full_dose_sub1, full_dose)
        
        loss = loss_mse
        
        # DFL loss (optional)
        if self.use_dfl_loss:
            if self.context:
                x_middle = low_dose[:, 1].unsqueeze(1)
            else:
                x_middle = low_dose
        
            loss_dfl_1 = self.compute_dfl_loss(out_dist_1, full_dose, x_middle)
            loss_dfl_2 = self.compute_dfl_loss(out_dist_2, full_dose, x_middle)
            loss_dfl = 0.5 * loss_dfl_1 + 0.5 * loss_dfl_2
            
            loss = loss_mse + self.dfl_weight * loss_dfl
            
            # 调试信息
            if n_iter % 500 == 1:
                print(f"\n[Iter {n_iter}] loss_mse: {loss_mse.item():.6f}, loss_dfl: {loss_dfl.item():.6f}")
                if dose is not None:
                    print(f"  dose values: {dose.cpu().numpy()}")

        loss.backward()

        if opt.wandb:
            if n_iter == opt.resume_iter + 1:
                wandb.init(project="your wandb project name")

        self.optimizer.step()
        self.optimizer.zero_grad()

        lr = self.optimizer.param_groups[0]['lr']
        loss_value = loss.item()
        
        if self.use_dfl_loss:
            self.logger.msg([loss_value, loss_mse.item(), loss_dfl.item(), lr], n_iter)
        else:
            self.logger.msg([loss_value, lr], n_iter)

        if opt.wandb:
            log_dict = {'epoch': n_iter, 'loss': loss_value, 'loss_mse': loss_mse.item()}
            if self.use_dfl_loss:
                log_dict['loss_dfl'] = loss_dfl.item()
            wandb.log(log_dict)

        if n_iter % self.update_ema_iter == 0:
            self.step_ema(n_iter)


    @torch.no_grad()
    def test(self, n_iter):
        opt = self.opt
        self.ema_model.eval()

        psnr, ssim, rmse = 0., 0., 0.
        for inputs in tqdm.tqdm(self.test_loader, desc='test'):
            # 🔥 兼容新旧格式
            if isinstance(inputs, dict):
                low_dose = inputs['input'].cuda()
                full_dose = inputs['target'].cuda()
                
                if self.use_dose_condition and 'dose' in inputs:
                    dose = inputs['dose'].float().cuda()
                else:
                    dose = None
            else:
                low_dose, full_dose = inputs
                low_dose, full_dose = low_dose.cuda(), full_dose.cuda()
                dose = None

            # 🔥 传入 dose 参数
            gen_full_dose, direct_recons, imstep_imgs = self.ema_model.sample(
                batch_size = low_dose.shape[0],
                img = low_dose,
                t = self.T,
                sampling_routine = self.sampling_routine,
                n_iter=n_iter,
                start_adjust_iter=opt.start_adjust_iter,
                dose=dose
            )

            full_dose = self.transfer_calculate_window(full_dose)
            gen_full_dose = self.transfer_calculate_window(gen_full_dose)
        
            data_range = full_dose.max() - full_dose.min()
            psnr_score, ssim_score, rmse_score = compute_measure(full_dose, gen_full_dose, data_range)
            psnr += psnr_score / len(self.test_loader)
            ssim += ssim_score / len(self.test_loader)
            rmse += rmse_score / len(self.test_loader)

        self.logger.msg([psnr, ssim, rmse], n_iter)

        if opt.wandb:
            wandb.log({'epoch': n_iter, 'PSNR': psnr, 'SSIM': ssim, 'RMSE': rmse})


    @torch.no_grad()
    def generate_images(self, n_iter):
        opt = self.opt
        self.ema_model.eval()
        low_dose, full_dose = self.test_images

        # 🔥 为可视化生成 dose 条件
        if self.use_dose_condition:
            # 获取第一个 dose 值
            dose_value = opt.dose if isinstance(opt.dose, int) else int(opt.dose.split(',')[0])
            dose = torch.full((low_dose.shape[0],), dose_value, dtype=torch.float32, device=low_dose.device)
        else:
            dose = None

        gen_full_dose, direct_recons, imstep_imgs = self.ema_model.sample(
                batch_size = low_dose.shape[0],
                img = low_dose,
                t = self.T,
                sampling_routine = self.sampling_routine,
                n_iter=n_iter,
                start_adjust_iter=opt.start_adjust_iter,
                dose=dose
            )

        if self.context:
            low_dose = low_dose[:, 1].unsqueeze(1)

        b, c, w, h = low_dose.size()
        fake_imgs = torch.stack([low_dose, full_dose, gen_full_dose])
        fake_imgs = self.transfer_display_window(fake_imgs)
        fake_imgs = fake_imgs.transpose(1, 0).reshape((-1, c, w, h))
        self.logger.save_image(torchvision.utils.make_grid(fake_imgs, nrow=3),
                               n_iter, 'test_{}_{}'.format(self.dose, self.sampling_routine) + '_' + opt.test_dataset)


    def train_osl_framework(self, test_iter):
        opt = self.opt
        self.ema_model.eval()

        ''' Initialize WeightNet '''
        weightnet = WeightNet(weight_num=10).cuda()
        optimizer_w = torch.optim.Adam(weightnet.parameters(), opt.init_lr*10)
        lossfn = PerceptualLoss()

        ''' get imstep images of diffusion '''
        for i in range(len(self.test_dataset)-2):
            if i == opt.index:
                if opt.unpair:
                    inputs_low = self.test_dataset[i]
                    inputs_full = self.test_dataset[i+2]
                else:
                    inputs_low = self.test_dataset[i]
                    inputs_full = inputs_low
                
                # 🔥 兼容新旧格式
                if isinstance(inputs_low, dict):
                    low_dose = inputs_low['input']
                    full_dose = inputs_full['target']
                else:
                    low_dose, _ = inputs_low
                    _, full_dose = inputs_full
                    
        low_dose, full_dose = torch.from_numpy(low_dose).unsqueeze(0).cuda(), torch.from_numpy(full_dose).unsqueeze(0).cuda()

        # OSL framework 不需要 dose 条件
        gen_full_dose, direct_recons, imstep_imgs = self.ema_model.sample(
            batch_size=low_dose.shape[0],
            img=low_dose,
            t=self.T,
            sampling_routine=self.sampling_routine,
            start_adjust_iter=opt.start_adjust_iter,
            dose=None
        )

        inputs = imstep_imgs.transpose(0, 2).squeeze(0)
        targets = full_dose

        ''' train WeightNet '''
        input_patches, target_patches = self.get_patch(inputs, targets, patch_size=opt.patch_size, stride=32)
        input_patches, target_patches = input_patches.detach(), target_patches.detach()

        for n_iter in tqdm.trange(1, opt.osl_max_iter):
            weightnet.train()
            batch_ids = torch.from_numpy(np.random.randint(0, input_patches.shape[0], opt.osl_batch_size)).cuda()
            input = input_patches.index_select(dim = 0, index = batch_ids).detach()
            target = target_patches.index_select(dim = 0, index = batch_ids).detach()

            out, weights = weightnet(input)
            loss = lossfn(out, target)
            loss.backward()

            optimizer_w.step()
            optimizer_w.zero_grad()
            lr = optimizer_w.param_groups[0]['lr']
            self.logger.msg([loss, lr], n_iter)
            if opt.wandb:
                wandb.log({'epoch': n_iter, 'loss': loss})
        opt_image = weights * inputs
        opt_image = opt_image.sum(dim=1, keepdim=True)
        print(weights)

        ''' Calculate the quantitative metrics before and after weighting'''
        full_dose_cal = self.transfer_calculate_window(full_dose)
        gen_full_dose_cal = self.transfer_calculate_window(gen_full_dose)
        opt_image_cal = self.transfer_calculate_window(opt_image)
        data_range = full_dose_cal.max() - full_dose_cal.min()
        psnr_ori, ssim_ori, rmse_ori = compute_measure(full_dose_cal, gen_full_dose_cal, data_range)
        psnr_opt, ssim_opt, rmse_opt = compute_measure(full_dose_cal, opt_image_cal, data_range)
        self.logger.msg([psnr_ori, ssim_ori, rmse_ori], test_iter)
        self.logger.msg([psnr_opt, ssim_opt, rmse_opt], test_iter)

        fake_imgs = torch.cat((low_dose[:, 1].unsqueeze(1), full_dose, gen_full_dose, opt_image), dim=0)
        fake_imgs = self.transfer_display_window(fake_imgs)
        self.logger.save_image(torchvision.utils.make_grid(fake_imgs, nrow=4), test_iter,
                               'test_opt_' + opt.test_dataset + '_{}_{}'.format(self.dose, opt.index))

        if opt.unpair:
            filename = './weights/unpair_weights_' + opt.test_dataset + '_{}_{}.npy'.format(self.dose, opt.index)
        else:
            filename = './weights/weights_' + opt.test_dataset + '_{}_{}.npy'.format(self.dose, opt.index)
        np.save(filename, weights.detach().cpu().squeeze().numpy())


    def test_osl_framework(self, test_iter):
        opt = self.opt
        self.ema_model.eval()

        if opt.unpair:
            filename = './weights/unpair_weights_' + opt.test_dataset + '_{}_{}.npy'.format(self.dose, opt.index)
        else:
            filename = './weights/weights_' + opt.test_dataset + '_{}_{}.npy'.format(self.dose, opt.index)
        weights = np.load(filename)
        print(weights)
        weights = torch.from_numpy(weights).unsqueeze(1).unsqueeze(2).unsqueeze(0).cuda()

        psnr_ori, ssim_ori, rmse_ori = 0., 0., 0.
        psnr_opt, ssim_opt, rmse_opt = 0., 0., 0.

        for inputs in tqdm.tqdm(self.test_loader, desc='test'):
            # 🔥 兼容新旧格式
            if isinstance(inputs, dict):
                low_dose = inputs['input'].cuda()
                full_dose = inputs['target'].cuda()
            else:
                low_dose, full_dose = inputs
                low_dose, full_dose = low_dose.cuda(), full_dose.cuda()

            gen_full_dose, direct_recons, imstep_imgs = self.ema_model.sample(
                batch_size=low_dose.shape[0],
                img=low_dose,
                t=self.T,
                sampling_routine=self.sampling_routine,
                n_iter=test_iter,
                start_adjust_iter=opt.start_adjust_iter,
                dose=None
            )
            imstep_imgs = imstep_imgs[:self.T]
            inputs = imstep_imgs.squeeze(2).transpose(0, 1)

            opt_image = weights * inputs
            opt_image = opt_image.sum(dim=1, keepdim=True)

            full_dose = self.transfer_calculate_window(full_dose)
            gen_full_dose = self.transfer_calculate_window(gen_full_dose)
            opt_image = self.transfer_calculate_window(opt_image)

            data_range = full_dose.max() - full_dose.min()
            psnr_ori_score, ssim_ori_score, rmse_ori_score = compute_measure(full_dose, gen_full_dose, data_range)
            psnr_opt_score, ssim_opt_score, rmse_opt_score = compute_measure(full_dose, opt_image, data_range)

            psnr_ori += psnr_ori_score / len(self.test_loader)
            ssim_ori += ssim_ori_score / len(self.test_loader)
            rmse_ori += rmse_ori_score / len(self.test_loader)

            psnr_opt += psnr_opt_score / len(self.test_loader)
            ssim_opt += ssim_opt_score / len(self.test_loader)
            rmse_opt += rmse_opt_score / len(self.test_loader)

        self.logger.msg([psnr_ori, ssim_ori, rmse_ori], test_iter)
        self.logger.msg([psnr_opt, ssim_opt, rmse_opt], test_iter)


    def get_patch(self, input_img, target_img, patch_size=256, stride=32):
        input_patches = []
        target_patches = []
        _, c_input, h, w = input_img.shape
        _, c_target, h, w = target_img.shape

        Top = np.arange(0, h - patch_size + 1, stride)
        Left = np.arange(0, w - patch_size + 1, stride)
        for t_idx in range(len(Top)):
            top = Top[t_idx]
            for l_idx in range(len(Left)):
                left = Left[l_idx]
                input_patch = input_img[:, :, top:top + patch_size, left:left + patch_size]
                target_patch = target_img[:, :, top:top + patch_size, left:left + patch_size]
                input_patches.append(input_patch)
                target_patches.append(target_patch)

        input_patches = torch.stack(input_patches).transpose(0, 1).reshape((-1, c_input, patch_size, patch_size))
        target_patches = torch.stack(target_patches).transpose(0, 1).reshape((-1, c_target, patch_size, patch_size))
        return input_patches, target_patches
