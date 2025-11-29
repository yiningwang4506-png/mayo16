import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from einops import rearrange
import sys
sys.path.append('.')
from FCB import FCB


# ============================================================
# 🔥 Dose Embedding 模块
# ============================================================
class DoseEmbedding(nn.Module):
    """
    将剂量数值编码为embedding向量
    比文本描述更直接、区分度更高
    
    支持的剂量: 5, 10, 25, 50, 100 等
    """
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 方式1：可学习的离散剂量embedding
        # 索引: 0-100 对应 0%-100% 剂量
        self.dose_embed = nn.Embedding(101, embed_dim)
        
        # 方式2：连续剂量的MLP编码（更灵活，处理任意剂量值）
        self.dose_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )
        
        # 融合两种编码
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    def forward(self, dose):
        """
        Args:
            dose: [B] 或 [B, 1] - 剂量值 (25, 50等)
        Returns:
            [B, embed_dim] - 剂量embedding
        """
        if dose.dim() == 1:
            dose = dose.unsqueeze(1)  # [B, 1]
        
        # 离散embedding
        dose_int = dose.squeeze(1).long().clamp(0, 100)
        discrete_emb = self.dose_embed(dose_int)  # [B, embed_dim]
        
        # 连续MLP编码（归一化到0-1）
        dose_norm = dose.float() / 100.0
        continuous_emb = self.dose_mlp(dose_norm)  # [B, embed_dim]
        
        # 融合
        combined = torch.cat([discrete_emb, continuous_emb], dim=1)
        output = self.fusion(combined)
        
        return output


# ============================================================
# 🔥 FiLM 层 (带稳定初始化)
# ============================================================
class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation
    使用残差连接和小值初始化，确保训练稳定
    """
    def __init__(self, cond_dim, feature_dim):
        super(FiLMLayer, self).__init__()
        
        # 两层MLP生成gamma和beta
        self.fc = nn.Sequential(
            nn.Linear(cond_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim * 2)
        )
        
        # 🔥 关键：小值初始化，确保起步接近baseline
        nn.init.normal_(self.fc[2].weight, mean=0, std=0.01)
        nn.init.zeros_(self.fc[2].bias)
        
        self.feature_dim = feature_dim

    def forward(self, x, cond):
        """
        Args:
            x: [B, C, H, W] - 特征图
            cond: [B, cond_dim] - 条件向量 (dose embedding)
        """
        if cond is None:
            return x
        
        B, C, H, W = x.shape
        
        params = self.fc(cond)  # [B, C*2]
        gamma, beta = params.chunk(2, dim=1)
        
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)
        
        # 标准FiLM: (1 + gamma) * x + beta
        # gamma初始化接近0，所以起步时接近identity
        return (1 + gamma) * x + beta


# ============================================================
# 原有模块 (保持不变)
# ============================================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class single_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class up(nn.Module):
    def __init__(self, in_ch):
        super(up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = x2 + x1
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class adjust_net(nn.Module):
    def __init__(self, out_channels=64, middle_channels=32):
        super(adjust_net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(2, middle_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(middle_channels, middle_channels*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(middle_channels*2, middle_channels*4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),
            nn.Conv2d(middle_channels*4, out_channels*2, 1, padding=0)
        )

    def forward(self, x):
        out = self.model(x)
        out = F.adaptive_avg_pool2d(out, (1,1))
        out1 = out[:, :out.shape[1]//2]
        out2 = out[:, out.shape[1]//2:]
        return out1, out2


# ============================================================
# UNet (添加 Dose Embedding)
# ============================================================
class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1,
                 reg_max=18,
                 y_0=-160.0,
                 y_n=200.0,
                 norm_range_max=3072.0,
                 norm_range_min=-1024.0,
                 dose_emb_dim=256):
        super(UNet, self).__init__()

        # DRL parameters (完全保持不变)
        self.reg_max = reg_max
        self.y_0 = y_0
        self.y_n = y_n
        self.norm_range_max = norm_range_max
        self.norm_range_min = norm_range_min
        
        # 🔥 Dose Embedding 模块
        self.dose_embedding = DoseEmbedding(embed_dim=dose_emb_dim)

        dim = 32
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        self.inc = nn.Sequential(
            single_conv(in_channels, 64),
            single_conv(64, 64)
        )

        self.down1 = nn.AvgPool2d(2)
        self.mlp1 = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, 64)
        )
        self.adjust1 = adjust_net(64)
        self.conv1 = nn.Sequential(
            single_conv(64, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.down2 = nn.AvgPool2d(2)
        self.mlp2 = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, 128)
        )
        self.adjust2 = adjust_net(128)
        
        # === FCB Integration (完全保持不变) ===
        self.conv2_spatial = nn.Sequential(
            single_conv(128, 256),
            single_conv(256, 256),
            single_conv(256, 256)
        )
        
        self.conv2_freq = FCB(
            input_chs=128,
            output_chs=128,
            num_rows=128,
            num_cols=128,
            stride=1,
            init='he'
        )
        
        self.conv2_fusion = nn.Sequential(
            single_conv(384, 256),  # 256 + 128 = 384
            single_conv(256, 256),
            single_conv(256, 256)
        )
        # === End FCB Integration ===
        
        # 🔥 FiLM层 - 在bottleneck和decoder注入dose condition
        self.film_bottleneck = FiLMLayer(dose_emb_dim, 256)
        self.film_decoder = FiLMLayer(dose_emb_dim, 128)

        self.up1 = up(256)
        self.mlp3 = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, 128)
        )
        self.adjust3 = adjust_net(128)
        self.conv3 = nn.Sequential(
            single_conv(128, 128),
            single_conv(128, 128),
            single_conv(128, 128)
        )

        self.up2 = up(128)
        self.mlp4 = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, 64)
        )
        self.adjust4 = adjust_net(64)
        self.conv4 = nn.Sequential(
            single_conv(64, 64),
            single_conv(64, 64)
        )

        # DRL output layer (完全保持不变)
        self.outc = outconv(64, self.reg_max + 1)
        self.outc.conv.bias.data[:] = 1.0
        
        proj = torch.linspace(y_0, y_n, self.reg_max + 1, dtype=torch.float)
        proj = proj / (norm_range_max - norm_range_min)
        self.register_buffer('proj', proj, persistent=False)
        
        self.hu_interval = (y_n - y_0) / self.reg_max

    def forward(self, x, t, x_adjust, adjust, dose=None):
        """
        Args:
            x: 输入图像
            t: 时间步
            x_adjust: adjust网络的输入
            adjust: 是否使用adjust
            dose: [B] - 剂量值 (25, 50等)，如果为None则不使用dose condition
        """
        # 🔥 计算 dose embedding
        if dose is not None:
            dose_emb = self.dose_embedding(dose)  # [B, 256]
        else:
            dose_emb = None
        
        inx = self.inc(x)
        time_emb = self.time_mlp(t)
        
        down1 = self.down1(inx)
        condition1 = self.mlp1(time_emb)
        b, c = condition1.shape
        condition1 = rearrange(condition1, 'b c -> b c 1 1')
        if adjust:
            gamma1, beta1 = self.adjust1(x_adjust)
            down1 = down1 + gamma1 * condition1 + beta1
        else:
            down1 = down1 + condition1
        conv1 = self.conv1(down1)

        down2 = self.down2(conv1)
        condition2 = self.mlp2(time_emb)
        b, c = condition2.shape
        condition2 = rearrange(condition2, 'b c -> b c 1 1')
        if adjust:
            gamma2, beta2 = self.adjust2(x_adjust)
            down2 = down2 + gamma2 * condition2 + beta2
        else:
            down2 = down2 + condition2
        
        # === FCB Forward (完全保持不变) ===
        spatial_feat = self.conv2_spatial(down2)  # (B, 256, 128, 128)
        freq_feat = self.conv2_freq(down2)        # (B, 128, 128, 128)
        
        merged = torch.cat([spatial_feat, 0.3 * freq_feat], dim=1) 
        conv2 = self.conv2_fusion(merged)         # (B, 256, 128, 128)
        # === End FCB Forward ===
        
        # 🔥 FiLM注入点1: bottleneck
        conv2 = self.film_bottleneck(conv2, dose_emb)

        up1 = self.up1(conv2, conv1)
        condition3 = self.mlp3(time_emb)
        b, c = condition3.shape
        condition3 = rearrange(condition3, 'b c -> b c 1 1')
        if adjust:
            gamma3, beta3 = self.adjust3(x_adjust)
            up1 = up1 + gamma3 * condition3 + beta3
        else:
            up1 = up1 + condition3
        conv3 = self.conv3(up1)
        
        # 🔥 FiLM注入点2: decoder
        conv3 = self.film_decoder(conv3, dose_emb)

        up2 = self.up2(conv3, inx)
        condition4 = self.mlp4(time_emb)
        b, c = condition4.shape
        condition4 = rearrange(condition4, 'b c -> b c 1 1')
        if adjust:
            gamma4, beta4 = self.adjust4(x_adjust)
            up2 = up2 + gamma4 * condition4 + beta4
        else:
            up2 = up2 + condition4
        conv4 = self.conv4(up2)

        # DRL output (完全保持不变)
        out = self.outc(conv4)
        out_dist = out.permute(0, 2, 3, 1)
        out = out_dist.softmax(3).matmul(self.proj.view([-1, 1]))
        out = out.permute(0, 3, 1, 2)
        
        return out, out_dist


class Network(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, context=True,
                 reg_max=18,
                 y_0=-160.0,
                 y_n=200.0,
                 norm_range_max=3072.0,
                 norm_range_min=-1024.0,
                 dose_emb_dim=256):
        super(Network, self).__init__()
        self.unet = UNet(in_channels=in_channels, 
                        out_channels=out_channels,
                        reg_max=reg_max,
                        y_0=y_0,
                        y_n=y_n,
                        norm_range_max=norm_range_max,
                        norm_range_min=norm_range_min,
                        dose_emb_dim=dose_emb_dim)
        self.context = context
        
        self.reg_max = reg_max
        self.hu_interval = self.unet.hu_interval
        self.y_0 = y_0
        self.norm_range_max = norm_range_max
        self.norm_range_min = norm_range_min

    def forward(self, x, t, y, x_end, adjust=True, dose=None):
        """
        Args:
            x: 输入图像
            t: 时间步
            y: ground truth
            x_end: 噪声端
            adjust: 是否使用adjust
            dose: [B] - 剂量值 (25, 50等)
        """
        if self.context:
            x_middle = x[:, 1].unsqueeze(1)
        else:
            x_middle = x
        
        x_adjust = torch.cat((y, x_end), dim=1)
        out, out_dist = self.unet(x, t, x_adjust, adjust=adjust, dose=dose)
        out = out + x_middle

        return out, out_dist


class WeightNet(nn.Module):
    def __init__(self, weight_num=10):
        super(WeightNet, self).__init__()
        init = torch.ones([1, weight_num, 1, 1]) / weight_num
        self.weights = nn.Parameter(init)

    def forward(self, x):
        weights = F.softmax(self.weights, 1)
        out = weights * x
        out = out.sum(dim=1, keepdim=True)
        return out, weights
