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
# FiLM 层 - 用于注入文本条件
# ============================================================
class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation
    用文本条件调制特征图
    """
    def __init__(self, cond_dim, feature_dim):
        super(FiLMLayer, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(cond_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim * 2)
        )
        
        # 初始化：让 gamma 接近 0，beta 接近 0
        # 这样初始时 FiLM 接近恒等映射
        nn.init.zeros_(self.fc[2].weight)
        nn.init.zeros_(self.fc[2].bias)
        
        self.feature_dim = feature_dim

    def forward(self, x, cond):
        """
        Args:
            x: [B, C, H, W] 特征图
            cond: [B, cond_dim] 文本条件
        """
        if cond is None:
            return x
        
        B, C, H, W = x.shape
        
        params = self.fc(cond)  # [B, C*2]
        gamma, beta = params.chunk(2, dim=1)  # 各 [B, C]
        
        gamma = gamma.view(B, C, 1, 1)
        beta = beta.view(B, C, 1, 1)
        
        # FiLM: (1 + gamma) * x + beta
        return (1 + gamma) * x + beta


# ============================================================
# 基础模块（保持不变）
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
# UNet（支持 Text Condition）
# ============================================================
class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1,
                 reg_max=18,
                 y_0=-160.0,
                 y_n=200.0,
                 norm_range_max=3072.0,
                 norm_range_min=-1024.0,
                 text_emb_dim=None):  # 🔥 新增参数
        super(UNet, self).__init__()

        self.reg_max = reg_max
        self.y_0 = y_0
        self.y_n = y_n
        self.norm_range_max = norm_range_max
        self.norm_range_min = norm_range_min
        self.use_text_condition = text_emb_dim is not None
        
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
        
        # FCB 分支
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
            single_conv(384, 256),
            single_conv(256, 256),
            single_conv(256, 256)
        )
        
        # 🔥 FiLM 层（如果启用 text condition）
        if self.use_text_condition:
            self.film_enc1 = FiLMLayer(text_emb_dim, 128)
            self.film_bottleneck = FiLMLayer(text_emb_dim, 256)
            self.film_dec1 = FiLMLayer(text_emb_dim, 128)
            self.film_dec2 = FiLMLayer(text_emb_dim, 64)

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

        # DRL output
        self.outc = outconv(64, self.reg_max + 1)
        self.outc.conv.bias.data[:] = 1.0
        
        proj = torch.linspace(y_0, y_n, self.reg_max + 1, dtype=torch.float)
        proj = proj / (norm_range_max - norm_range_min)
        self.register_buffer('proj', proj, persistent=False)
        
        self.hu_interval = (y_n - y_0) / self.reg_max

    def forward(self, x, t, x_adjust, adjust, text_emb=None):
        """
        Args:
            x: [B, C, H, W] 输入
            t: [B] 时间步
            x_adjust: [B, 2, H, W] 调整信号
            adjust: bool 是否使用 adjust_net
            text_emb: [B, text_emb_dim] 文本条件（可选）
        """
        inx = self.inc(x)
        time_emb = self.time_mlp(t)
        
        # Encoder
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
        
        # 🔥 FiLM 注入点 1: encoder
        if self.use_text_condition and text_emb is not None:
            conv1 = self.film_enc1(conv1, text_emb)

        down2 = self.down2(conv1)
        condition2 = self.mlp2(time_emb)
        b, c = condition2.shape
        condition2 = rearrange(condition2, 'b c -> b c 1 1')
        if adjust:
            gamma2, beta2 = self.adjust2(x_adjust)
            down2 = down2 + gamma2 * condition2 + beta2
        else:
            down2 = down2 + condition2
        
        # FCB
        spatial_feat = self.conv2_spatial(down2)
        freq_feat = self.conv2_freq(down2)
        merged = torch.cat([spatial_feat, 0.3 * freq_feat], dim=1)
        conv2 = self.conv2_fusion(merged)
        
        # 🔥 FiLM 注入点 2: bottleneck（最重要）
        if self.use_text_condition and text_emb is not None:
            conv2 = self.film_bottleneck(conv2, text_emb)

        # Decoder
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
        
        # 🔥 FiLM 注入点 3: decoder
        if self.use_text_condition and text_emb is not None:
            conv3 = self.film_dec1(conv3, text_emb)

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
        
        # 🔥 FiLM 注入点 4: output
        if self.use_text_condition and text_emb is not None:
            conv4 = self.film_dec2(conv4, text_emb)

        # DRL output
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
                 text_emb_dim=None):  # 🔥 新增
        super(Network, self).__init__()
        self.unet = UNet(
            in_channels=in_channels, 
            out_channels=out_channels,
            reg_max=reg_max,
            y_0=y_0,
            y_n=y_n,
            norm_range_max=norm_range_max,
            norm_range_min=norm_range_min,
            text_emb_dim=text_emb_dim
        )
        self.context = context
        
        self.reg_max = reg_max
        self.hu_interval = self.unet.hu_interval
        self.y_0 = y_0
        self.norm_range_max = norm_range_max
        self.norm_range_min = norm_range_min

    def forward(self, x, t, y, x_end, adjust=True, text_emb=None):
        if self.context:
            x_middle = x[:, 1].unsqueeze(1)
        else:
            x_middle = x
        
        x_adjust = torch.cat((y, x_end), dim=1)
        out, out_dist = self.unet(x, t, x_adjust, adjust=adjust, text_emb=text_emb)
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