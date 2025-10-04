import torch
import torch.nn as nn
import torch.nn.functional as F


class DRL_Loss(nn.Module):
    def __init__(self, n_bins=19, y_min=-160, y_max=200, lambda_df=0.02):
        super().__init__()
        self.lambda_df = lambda_df
        self.n_bins = n_bins
        # 定义bins: [-160, -140, ..., 200]
        self.register_buffer('bins', torch.linspace(y_min, y_max, n_bins))
        
    def forward(self, prediction, probabilities, target):
        """
        prediction: [B, 1, H, W] - 加权求和后的预测值
        probabilities: [B, 19, H, W] - 19个bins的概率分布
        target: [B, 1, H, W] - 真实的full_dose
        """
        # MSE Loss (原来就有的)
        mse_loss = F.mse_loss(prediction, target)
        
        # Distribution Focal Loss (新增的)
        df_loss = self.distribution_focal_loss(probabilities, target)
        
        # 组合
        total_loss = mse_loss + self.lambda_df * df_loss
        return total_loss
    
    def distribution_focal_loss(self, probs, target):
        """
        强制网络将概率集中在target附近的两个bins上
        probs: [B, 19, H, W]
        target: [B, 1, H, W]
        """
        B, C, H, W = probs.shape
        
        # 将target限制在bins范围内
        target_clamped = torch.clamp(target, self.bins[0], self.bins[-1])
        
        # 找到target对应的左右bins索引
        bins_tensor = self.bins.view(1, -1, 1, 1)  # [1, 19, 1, 1]
        
        # 找到小于等于target的最大bin（左侧bin）
        # 对每个bin，判断是否 <= target
        mask_left = (bins_tensor <= target_clamped).float()  # [B, 19, H, W]
        idx_left = torch.sum(mask_left, dim=1, keepdim=True) - 1  # [B, 1, H, W]
        idx_left = torch.clamp(idx_left.long(), 0, C-2)
        
        idx_right = idx_left + 1
        
        # 获取左右bins的值
        bins_expanded = self.bins.view(1, -1, 1, 1).expand(B, -1, H, W)
        bin_left = torch.gather(bins_expanded, 1, idx_left)    # [B, 1, H, W]
        bin_right = torch.gather(bins_expanded, 1, idx_right)  # [B, 1, H, W]
        
        # 计算权重（线性插值）
        weight_right = (target_clamped - bin_left) / (bin_right - bin_left + 1e-8)
        weight_left = 1.0 - weight_right
        
        # 获取对应概率
        prob_left = torch.gather(probs, 1, idx_left)   # [B, 1, H, W]
        prob_right = torch.gather(probs, 1, idx_right) # [B, 1, H, W]
        
        # 计算focal loss（交叉熵形式）
        loss = -(weight_left * torch.log(prob_left + 1e-8) + 
                 weight_right * torch.log(prob_right + 1e-8))
        
        return loss.mean()