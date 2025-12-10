
"""
CoreDiff 最优条件包装器
直接替换原来的 corediff_wrapper_fixed.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class OptimalConditionEncoder(nn.Module):
    """可学习的条件编码器"""
    def __init__(self, num_devices=3, num_doses=2, emb_dim=256):
        super().__init__()
        
        # 6 种组合，每种独立学习
        num_combinations = num_devices * num_doses
        self.condition_emb = nn.Embedding(num_combinations, emb_dim)
        
        # 正交初始化
        nn.init.orthogonal_(self.condition_emb.weight)
        self.condition_emb.weight.data *= 1.5
        
        # MLP 增加表达能力
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.GELU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        
        self.num_doses = num_doses
        
    def forward(self, device_idx, dose_idx):
        combo_idx = device_idx * self.num_doses + dose_idx
        emb = self.condition_emb(combo_idx)
        return self.mlp(emb) + emb


class AdaptiveFiLM(nn.Module):
    """自适应 FiLM 层"""
    def __init__(self, feature_dim, condition_dim=256, init_mix=0.3):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(condition_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim * 2),
        )
        
        # 初始化
        nn.init.normal_(self.fc[0].weight, std=0.02)
        nn.init.zeros_(self.fc[0].bias)
        nn.init.normal_(self.fc[2].weight, std=0.02)
        self.fc[2].bias.data[:feature_dim] = 1.0   # gamma = 1
        self.fc[2].bias.data[feature_dim:] = 0.0   # beta = 0
        
        # 混合权重
        import math
        self.mix_logit = nn.Parameter(torch.tensor(math.log(init_mix / (1 - init_mix))))
        
    def forward(self, x, condition):
        B, C, H, W = x.shape
        
        params = self.fc(condition)
        gamma = params[:, :C].view(B, C, 1, 1)
        beta = params[:, C:].view(B, C, 1, 1)
        
        x_film = gamma * x + beta
        
        mix = torch.sigmoid(self.mix_logit)
        return (1 - mix) * x + mix * x_film
    
    def get_mix_ratio(self):
        return torch.sigmoid(self.mix_logit).item()


class CoreDiffOptimalWrapper(nn.Module):
    """
    CoreDiff 最优条件包装器
    
    关键改进：
    1. 使用正交初始化的可学习 embedding（而非 BERT）
    2. 多层 FiLM 注入
    3. 自适应混合权重（非零初始化）
    """
    def __init__(self, 
                 original_corediff,
                 num_devices=3,
                 num_doses=2,
                 condition_dim=256):
        super().__init__()
        
        self.corediff = original_corediff
        self.condition_dim = condition_dim
        
        # 条件编码器
        self.condition_encoder = OptimalConditionEncoder(
            num_devices=num_devices,
            num_doses=num_doses,
            emb_dim=condition_dim
        )
        
        # 获取 denoise_fn
        self.denoise_fn = original_corediff.denoise_fn
        
        # 获取各层通道数
        channel_list = self._get_channels()
        
        # 创建 FiLM 层（在关键位置注入）
        # 策略：enc1, bottleneck, dec2 三个位置
        self.film_enc1 = AdaptiveFiLM(channel_list[0], condition_dim, init_mix=0.2)
        self.film_bottleneck = AdaptiveFiLM(channel_list[1], condition_dim, init_mix=0.4)
        self.film_dec2 = AdaptiveFiLM(channel_list[2], condition_dim, init_mix=0.3)
        
        print(f"✅ CoreDiffOptimalWrapper initialized")
        print(f"   Condition dim: {condition_dim}")
        print(f"   FiLM channels: enc1={channel_list[0]}, bottleneck={channel_list[1]}, dec2={channel_list[2]}")
        
        # 保存原始 forward 方法
        self._patch_denoise_fn()
    
    def _get_channels(self):
        """获取各层通道数"""
        df = self.denoise_fn
        
        # 尝试从模型获取
        try:
            enc1_ch = df.enc1[-1].out_channels if hasattr(df.enc1[-1], 'out_channels') else 64
        except:
            enc1_ch = 64
            
        try:
            bottleneck_ch = df.bottleneck[-1].out_channels if hasattr(df.bottleneck[-1], 'out_channels') else 256
        except:
            bottleneck_ch = 256
            
        try:
            dec2_ch = df.dec2[-1].out_channels if hasattr(df.dec2[-1], 'out_channels') else 128
        except:
            dec2_ch = 128
        
        return [enc1_ch, bottleneck_ch, dec2_ch]
    
    def _patch_denoise_fn(self):
        """打补丁到 denoise_fn 的 forward"""
        original_forward = self.denoise_fn.forward
        wrapper = self
        
        def new_forward(self_df, x, t, context=None, condition=None, **kwargs):
            # 如果没有条件，使用原始 forward
            if condition is None:
                return original_forward(x, t, context=context, **kwargs)
            
            # 时间嵌入
            t_emb = self_df.time_embedding(t)
            
            # Encoder
            e1 = self_df.enc1(torch.cat([x, context], dim=1) if context is not None else x)
            e1 = wrapper.film_enc1(e1, condition)  # FiLM 注入
            
            e2 = self_df.enc2(e1)
            e3 = self_df.enc3(e2)
            e4 = self_df.enc4(e3)
            
            # Bottleneck
            b = self_df.bottleneck(e4 + t_emb.unsqueeze(-1).unsqueeze(-1).expand_as(e4))
            b = wrapper.film_bottleneck(b, condition)  # FiLM 注入
            
            # Decoder
            d1 = self_df.dec1(torch.cat([b, e4], dim=1))
            d2 = self_df.dec2(torch.cat([d1, e3], dim=1))
            d2 = wrapper.film_dec2(d2, condition)  # FiLM 注入
            
            d3 = self_df.dec3(torch.cat([d2, e2], dim=1))
            d4 = self_df.dec4(torch.cat([d3, e1], dim=1))
            
            out = self_df.out(d4)
            return out
        
        # 替换方法
        import types
        self.denoise_fn.forward = types.MethodType(new_forward, self.denoise_fn)
    
    def forward(self, x, t, context=None, device_idx=None, dose_idx=None, **kwargs):
        """
        前向传播
        
        Args:
            x: [B, C, H, W] 输入
            t: [B] 时间步
            context: [B, C, H, W] 上下文（相邻切片）
            device_idx: [B] 设备索引 (0-2)
            dose_idx: [B] 剂量索引 (0-1)
        """
        # 获取条件 embedding
        if device_idx is not None and dose_idx is not None:
            condition = self.condition_encoder(device_idx, dose_idx)
        else:
            condition = None
        
        # 调用修改后的 forward
        return self.denoise_fn(x, t, context=context, condition=condition, **kwargs)
    
    def get_film_stats(self):
        """获取 FiLM 层状态"""
        return {
            'enc1_mix': self.film_enc1.get_mix_ratio(),
            'bottleneck_mix': self.film_bottleneck.get_mix_ratio(),
            'dec2_mix': self.film_dec2.get_mix_ratio(),
        }
    
    # 代理其他属性
    def __getattr__(self, name):
        if name in ['corediff', 'denoise_fn', 'condition_encoder', 
                    'film_enc1', 'film_bottleneck', 'film_dec2', 'condition_dim']:
            return super().__getattr__(name)
        return getattr(self.corediff, name)


def create_optimal_wrapper(corediff_model, num_devices=3, num_doses=2, condition_dim=256):
    """创建最优条件包装器的工厂函数"""
    return CoreDiffOptimalWrapper(
        corediff_model,
        num_devices=num_devices,
        num_doses=num_doses,
        condition_dim=condition_dim
    )


# 测试
if __name__ == '__main__':
    print("⚠️ 需要实际的 CoreDiff 模型来测试")
    print("使用方法:")
    print("  from corediff_optimal_wrapper import create_optimal_wrapper")
    print("  wrapped_model = create_optimal_wrapper(your_corediff_model)")
