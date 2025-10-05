# 在 distribution_focal_loss 开头添加调试信息
import re

with open('utils/drl_loss.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 修改 distribution_focal_loss 方法，添加调试
new_method = '''    def distribution_focal_loss(self, probs, target):
        """
        Distribution Focal Loss
        Args:
            probs: [B, n_bins, H, W] - 概率分布
            target: [B, 1, H, W] - 归一化后的目标值
        """
        # 🔍 调试：打印形状
        print(f"[DEBUG] probs.shape: {probs.shape}")
        print(f"[DEBUG] target.shape: {target.shape}")
        print(f"[DEBUG] bins.shape: {self.bins.shape}")
        
        device = target.device
        bins = self.bins.to(device)
        
        B, n_bins, H, W = probs.shape  # 直接从probs获取
        
        # Flatten
        target_flat = target.view(-1)  # [B*H*W]
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, n_bins)  # [B*H*W, n_bins]
        
        # 找到target对应的左右bins索引
        idx_left = torch.searchsorted(bins, target_flat) - 1
        idx_left = idx_left.clamp(0, n_bins - 2).long().to(device)
        idx_right = (idx_left + 1).to(device)
        
        # Expand bins for gathering
        bins_expanded = bins.unsqueeze(0).expand(B * H * W, -1)  # [B*H*W, n_bins]
        
        # 获取左右bins的值
        bin_left = torch.gather(bins_expanded, 1, idx_left.unsqueeze(1))
        bin_right = torch.gather(bins_expanded, 1, idx_right.unsqueeze(1))
        
        # 计算权重
        weight_right = (target_flat.unsqueeze(1) - bin_left) / (bin_right - bin_left + 1e-8)
        weight_left = 1 - weight_right
        
        # 获取对应bins的概率
        prob_left = torch.gather(probs_flat, 1, idx_left.unsqueeze(1))
        prob_right = torch.gather(probs_flat, 1, idx_right.unsqueeze(1))
        
        # 交叉熵损失（带权重）
        loss = -weight_left * torch.log(prob_left + 1e-8) - weight_right * torch.log(prob_right + 1e-8)
        
        return loss.mean()'''

pattern = r'    def distribution_focal_loss\(self, probs, target\):.*?(?=\n    def |\n\nclass |\Z)'
content = re.sub(pattern, new_method, content, flags=re.DOTALL)

with open('utils/drl_loss.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ 已添加调试信息")
print("运行: bash train.sh")
print("查看 probs 和 target 的实际形状")
