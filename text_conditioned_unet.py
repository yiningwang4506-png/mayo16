"""
Text-Conditioned CoreDiff U-Net
在 CoreDiff 的 U-Net 中添加 Cross-Attention 来融合文本条件
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange


class CrossAttentionBlock(nn.Module):
    """
    Cross-Attention模块:图像特征 attend to 文本embedding
    """
    
    def __init__(self, dim, text_dim=256, num_heads=8):
        """
        Args:
            dim: 图像特征维度
            text_dim: 文本embedding维度
            num_heads: 注意力头数
        """
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        # Query: 来自图像特征
        self.to_q = nn.Linear(dim, dim, bias=False)
        
        # Key, Value: 来自文本embedding
        self.to_k = nn.Linear(text_dim, dim, bias=False)
        self.to_v = nn.Linear(text_dim, dim, bias=False)
        
        # 输出投影
        self.proj = nn.Linear(dim, dim)
        
        # Layer norm
        self.norm_img = nn.LayerNorm(dim)
        self.norm_text = nn.LayerNorm(text_dim)
    
    def forward(self, x, text_emb):
        """
        Args:
            x: [B, C, H, W] - 图像特征
            text_emb: [B, text_dim] - 文本embedding
            
        Returns:
            [B, C, H, W] - 经过cross-attention增强的特征
        """
        B, C, H, W = x.shape
        
        # Reshape: [B, C, H, W] -> [B, H*W, C]
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        
        # Normalize
        x_flat = self.norm_img(x_flat)
        text_emb = self.norm_text(text_emb.unsqueeze(1))  # [B, 1, text_dim]
        
        # Compute Q, K, V
        q = self.to_q(x_flat)  # [B, H*W, C]
        k = self.to_k(text_emb)  # [B, 1, C]
        v = self.to_v(text_emb)  # [B, 1, C]
        
        # Multi-head attention
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        
        # Attention scores
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        # Weighted sum
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # Project and reshape back
        out = self.proj(out)
        out = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)
        
        # Residual connection
        return x + out


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


class TextConditionedUNet(nn.Module):
    """
    带文本条件的 CoreDiff U-Net
    在每个下采样和上采样阶段注入文本条件
    """
    
    def __init__(self, in_channels=2, out_channels=1, text_dim=256):
        super(TextConditionedUNet, self).__init__()

        dim = 32
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        # ========== Encoder ==========
        self.inc = nn.Sequential(
            single_conv(in_channels, 64),
            single_conv(64, 64)
        )
        # Text condition after inc
        self.text_attn_inc = CrossAttentionBlock(64, text_dim)

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
        # Text condition after conv1
        self.text_attn_conv1 = CrossAttentionBlock(128, text_dim)

        self.down2 = nn.AvgPool2d(2)
        self.mlp2 = nn.Sequential(
            nn.GELU(),
            nn.Linear(dim, 128)
        )
        self.adjust2 = adjust_net(128)
        self.conv2 = nn.Sequential(
            single_conv(128, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256),
            single_conv(256, 256)
        )
        # Text condition at bottleneck
        self.text_attn_bottleneck = CrossAttentionBlock(256, text_dim)

        # ========== Decoder ==========
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
        # Text condition after conv3
        self.text_attn_conv3 = CrossAttentionBlock(128, text_dim)

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
        # Text condition after conv4
        self.text_attn_conv4 = CrossAttentionBlock(64, text_dim)

        self.outc = outconv(64, out_channels)

    def forward(self, x, t, x_adjust, text_emb, adjust=True):
        """
        Args:
            x: [B, C, H, W] - 输入图像
            t: [B] - time step
            x_adjust: [B, 2, H, W] - adjust信号
            text_emb: [B, text_dim] - 文本条件
            adjust: bool - 是否使用adjust_net
            
        Returns:
            [B, 1, H, W] - 预测的干净图像
        """
        # Initial conv
        inx = self.inc(x)
        inx = self.text_attn_inc(inx, text_emb)  # ← 注入文本条件
        
        # Time embedding
        time_emb = self.time_mlp(t)
        
        # ========== Down path ==========
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
        conv1 = self.text_attn_conv1(conv1, text_emb)  # ← 注入文本条件

        down2 = self.down2(conv1)
        condition2 = self.mlp2(time_emb)
        b, c = condition2.shape
        condition2 = rearrange(condition2, 'b c -> b c 1 1')
        if adjust:
            gamma2, beta2 = self.adjust2(x_adjust)
            down2 = down2 + gamma2 * condition2 + beta2
        else:
            down2 = down2 + condition2
        conv2 = self.conv2(down2)
        conv2 = self.text_attn_bottleneck(conv2, text_emb)  # ← Bottleneck注入

        # ========== Up path ==========
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
        conv3 = self.text_attn_conv3(conv3, text_emb)  # ← 注入文本条件

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
        conv4 = self.text_attn_conv4(conv4, text_emb)  # ← 注入文本条件

        out = self.outc(conv4)
        return out


class TextConditionedNetwork(nn.Module):
    """
    完整的文本条件CoreDiff网络
    包含U-Net和residual connection
    """
    
    def __init__(self, in_channels=3, out_channels=1, context=True, text_dim=256):
        super(TextConditionedNetwork, self).__init__()
        self.unet = TextConditionedUNet(
            in_channels=in_channels, 
            out_channels=out_channels,
            text_dim=text_dim
        )
        self.context = context

    def forward(self, x, t, y, x_end, text_emb, adjust=True):
        """
        Args:
            x: [B, 3, H, W] - 输入(context模式) 或 [B, 1, H, W]
            t: [B] - time step
            y: [B, 1, H, W] - ground truth (仅训练时)
            x_end: [B, 1, H, W] - 噪声端点
            text_emb: [B, text_dim] - 文本条件
            adjust: bool
            
        Returns:
            [B, 1, H, W] - 预测的干净图像
        """
        if self.context:
            x_middle = x[:, 1].unsqueeze(1)
        else:
            x_middle = x
        
        x_adjust = torch.cat((y, x_end), dim=1)
        out = self.unet(x, t, x_adjust, text_emb, adjust=adjust) + x_middle

        return out


# ============== 示例用法 ==============
if __name__ == '__main__':
    # 创建模型
    model = TextConditionedNetwork(
        in_channels=3,
        context=True,
        text_dim=256
    ).cuda()
    
    # 模拟输入
    B = 2
    x = torch.randn(B, 3, 512, 512).cuda()
    t = torch.randint(0, 10, (B,)).cuda()
    y = torch.randn(B, 1, 512, 512).cuda()
    x_end = torch.randn(B, 1, 512, 512).cuda()
    text_emb = torch.randn(B, 256).cuda()
    
    # 前向传播
    out = model(x, t, y, x_end, text_emb, adjust=True)
    
    print(f"输入: {x.shape}")
    print(f"文本条件: {text_emb.shape}")
    print(f"输出: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")