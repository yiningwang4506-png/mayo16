import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("="*60)
print("dose_embed 诊断脚本 (GPU版本)")
print("="*60)

# ============================================================
# 简化测试:直接测试关键组件,不需要完整初始化模型
# ============================================================

# 测试1: 检查 UNet 是否有 dose_embed
print("\n[测试1] 检查 UNet 是否有 dose_embed")
print("-"*60)

try:
    from models.corediff.corediff_wrapper import UNet
    
    # 创建一个UNet实例
    unet = UNet(in_channels=3, out_channels=1).cuda()
    
    # 检查dose_embed是否存在
    if hasattr(unet, 'dose_embed'):
        print("✓ dose_embed 存在")
        print(f"  结构: {unet.dose_embed}")
    else:
        print("✗ dose_embed 不存在!")
        print("  → 请检查 corediff_wrapper.py 的 UNet.__init__")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# 测试2: 检查dose_embed的输入输出
print("\n[测试2] 检查 dose_embed 的输入输出")
print("-"*60)

with torch.no_grad():
    dose_10 = torch.tensor([[0.1]]).cuda()
    dose_25 = torch.tensor([[0.25]]).cuda()
    dose_90 = torch.tensor([[0.9]]).cuda()
    
    try:
        emb_10 = unet.dose_embed(dose_10)
        emb_25 = unet.dose_embed(dose_25)
        emb_90 = unet.dose_embed(dose_90)
        
        print(f"✓ dose_embed 可以正常运行")
        print(f"  输入shape: {dose_10.shape}")
        print(f"  输出shape: {emb_10.shape}")
        
        # 检查差异
        diff_10_25 = (emb_10 - emb_25).abs().mean().item()
        diff_10_90 = (emb_10 - emb_90).abs().mean().item()
        diff_25_90 = (emb_25 - emb_90).abs().mean().item()
        
        print(f"\n  嵌入向量差异:")
        print(f"    |emb(0.1) - emb(0.25)|: {diff_10_25:.6f}")
        print(f"    |emb(0.1) - emb(0.9)|:  {diff_10_90:.6f}")
        print(f"    |emb(0.25) - emb(0.9)|: {diff_25_90:.6f}")
        
        if diff_10_90 < 1e-6:
            print("\n  ⚠️ 警告: 嵌入向量完全相同(这在初始化时是正常的)")
        elif diff_10_90 < 0.01:
            print("\n  ⚠️ 差异较小")
        else:
            print("\n  ✓ 嵌入向量有差异")
            
    except Exception as e:
        print(f"✗ dose_embed 运行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


# 测试3: 检查forward参数
print("\n[测试3] 检查 UNet.forward 参数")
print("-"*60)

import inspect
forward_sig = inspect.signature(unet.forward)
params = list(forward_sig.parameters.keys())

print(f"UNet.forward 的参数: {params}")

if 'dose_value' in params:
    print("✓ forward 有 dose_value 参数")
else:
    print("✗ forward 没有 dose_value 参数!")
    print("  → 请修改 corediff_wrapper.py 的 UNet.forward")
    sys.exit(1)


# 测试4: 完整forward传播测试
print("\n[测试4] 测试完整的 forward 传播")
print("-"*60)

try:
    batch_size = 2
    # 使用完整尺寸512x512
    x = torch.randn(batch_size, 3, 512, 512).cuda()
    t = torch.tensor([5, 5]).cuda()
    x_adjust = torch.randn(batch_size, 2, 512, 512).cuda()
    adjust = False
    
    dose_low = torch.tensor([0.1, 0.1]).cuda()
    dose_high = torch.tensor([0.9, 0.9]).cuda()
    
    with torch.no_grad():
        print("  正在运行 forward(dose=0.1)...")
        out_low, dist_low = unet(x, t, x_adjust, adjust, dose_low)
        
        print("  正在运行 forward(dose=0.9)...")
        out_high, dist_high = unet(x, t, x_adjust, adjust, dose_high)
        
        print(f"✓ forward 传播成功")
        print(f"  输出shape: {out_low.shape}")
        
        # 检查输出差异
        output_diff = (out_low - out_high).abs().mean().item()
        print(f"\n  不同剂量的输出差异: {output_diff:.6f}")
        
        if output_diff < 1e-7:
            print("\n  ✗✗✗ 严重问题: 输出完全相同!")
            print("      dose_value 在 forward 中没有被使用!")
            print("\n  请检查 UNet.forward 内部:")
            print("    1. 是否调用了 self.dose_embed(dose_value)?")
            print("    2. 是否把 dose_emb 加到了 down1/down2/up1/up2?")
            print("\n  示例代码:")
            print("    dose_emb = self.dose_embed(dose_value.view(-1, 1))")
            print("    down1 = down1 + condition1 + dose_emb[:, :, None, None]")
        elif output_diff < 1e-4:
            print("\n  ⚠️⚠️ 输出差异很小!")
            print("      dose_value 可能作用太弱")
            print("\n  建议:")
            print("    1. 增加 dose_embed 的系数")
            print("    2. 增大 dose_embed 网络的容量")
        else:
            print("\n  ✓✓✓ 输出有明显差异!")
            print("      dose_value 正在起作用!")
            
except Exception as e:
    print(f"✗ forward 失败: {e}")
    import traceback
    traceback.print_exc()


# 测试5: 检查其他文件
print("\n[测试5] 检查其他文件的修改")
print("-"*60)

try:
    from models.corediff.corediff_wrapper import Network
    network_sig = inspect.signature(Network.forward)
    network_params = list(network_sig.parameters.keys())
    
    print(f"Network.forward 的参数: {network_params}")
    if 'dose_value' in network_params:
        print("✓ Network.forward 有 dose_value")
    else:
        print("⚠️ Network.forward 没有 dose_value")
except Exception as e:
    print(f"⚠️ 检查Network失败: {e}")

try:
    from models.corediff.diffusion_modules import Diffusion
    diff_sig = inspect.signature(Diffusion.forward)
    diff_params = list(diff_sig.parameters.keys())
    
    print(f"\nDiffusion.forward 的参数: {diff_params}")
    if 'dose_value' in diff_params:
        print("✓ Diffusion.forward 有 dose_value")
    else:
        print("⚠️ Diffusion.forward 没有 dose_value")
        
    # 检查sample方法
    sample_sig = inspect.signature(Diffusion.sample)
    sample_params = list(sample_sig.parameters.keys())
    print(f"\nDiffusion.sample 的参数: {sample_params}")
    if 'dose_value' in sample_params:
        print("✓ Diffusion.sample 有 dose_value")
    else:
        print("⚠️ Diffusion.sample 没有 dose_value")
        
except Exception as e:
    print(f"⚠️ 检查Diffusion失败: {e}")


# 测试6: 梯度检查
print("\n[测试6] 检查梯度流")
print("-"*60)

try:
    # 创建需要梯度的输入
    x = torch.randn(1, 3, 512, 512, requires_grad=True).cuda()
    t = torch.tensor([5]).cuda()
    x_adjust = torch.randn(1, 2, 512, 512).cuda()
    dose = torch.tensor([0.25]).cuda()
    
    # Forward
    out, dist = unet(x, t, x_adjust, False, dose)
    
    # Backward
    loss = out.mean()
    loss.backward()
    
    # 检查dose_embed的梯度
    dose_embed_weight = unet.dose_embed[0].weight
    if dose_embed_weight.grad is not None:
        grad_norm = dose_embed_weight.grad.norm().item()
        print(f"✓ dose_embed 有梯度")
        print(f"  梯度norm: {grad_norm:.6f}")
        
        if grad_norm < 1e-6:
            print("  ⚠️ 梯度很小,可能学习不到有用信息")
        else:
            print("  ✓ 梯度正常")
    else:
        print("✗ dose_embed 没有梯度!")
        print("  → 说明 dose_embed 没有参与前向传播!")
        
except Exception as e:
    print(f"⚠️ 梯度检查失败: {e}")
    import traceback
    traceback.print_exc()


# 总结
print("\n" + "="*60)
print("诊断总结")
print("="*60)

print("\n关键检查:")
print("  [测试4] 如果'输出完全相同' → dose_value没被使用")
print("  [测试6] 如果'没有梯度' → dose_embed没参与计算")
print("\n如果以上都通过,说明代码修改正确!")
print("\n下一步:")
print("  1. 如果测试通过,开始训练并监控:")
print("     - Loss是否下降")
print("     - 不同剂量的PSNR是否有差异")
print("  2. 训练1000步后,用不同剂量测试看效果区别")