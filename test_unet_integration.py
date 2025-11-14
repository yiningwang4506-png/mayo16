"""
测试 Text-Conditioned U-Net
验证模型能否正确接收图像和文本条件
"""
import torch
import sys
sys.path.append('/root/autodl-tmp/CoreDiff-main')

print("="*60)
print("🧪 Testing Text-Conditioned U-Net")
print("="*60)

# Step 1: 导入U-Net
print("\n🔵 Step 1: Importing U-Net...")
try:
    from text_conditioned_unet import TextConditionedUNet
    print("✅ U-Net imported successfully")
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: 创建模型
print("\n🔵 Step 2: Creating U-Net model...")
try:
    model = TextConditionedUNet(
        in_channels=3,  # 因为你用context,所以是3帧
        out_channels=1,
        text_dim=256  # ⭐ 修正: 参数名是 text_dim 而不是 text_embed_dim
    )
    model.eval()
    print(f"✅ Model created")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"❌ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: 准备测试数据
print("\n🔵 Step 3: Preparing test data...")
batch_size = 2
img = torch.randn(batch_size, 3, 512, 512)  # 模拟CT图像
text_emb = torch.randn(batch_size, 256)     # 模拟文本embedding
timestep = torch.tensor([100, 200])          # 模拟扩散timestep

# ⭐ CoreDiff 需要 x_adjust 参数 (用于 adjust_net)
# x_adjust 通常是 [target, noise] 拼接而成
x_adjust = torch.randn(batch_size, 2, 512, 512)  # [B, 2, H, W]

print(f"✅ Test data prepared")
print(f"   Image shape: {img.shape}")
print(f"   Text embedding shape: {text_emb.shape}")
print(f"   Timestep shape: {timestep.shape}")
print(f"   X_adjust shape: {x_adjust.shape}")

# Step 4: 前向传播
print("\n🔵 Step 4: Running forward pass...")
try:
    with torch.no_grad():
        output = model(img, timestep, x_adjust, text_emb, adjust=True)
    print(f"✅ Forward pass successful!")
    print(f"   Output shape: {output.shape}")
    
    # 验证输出维度
    assert output.shape == (batch_size, 1, 512, 512), "❌ Output shape错误!"
    print(f"✅ Output shape correct: {output.shape}")
    
except Exception as e:
    print(f"❌ Forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: 测试与真实数据集的兼容性
print("\n🔵 Step 5: Testing with real dataset...")
try:
    from text_conditioned_dataset import TextConditionedCTDataset
    from torch.utils.data import DataLoader
    
    # 创建数据集
    dataset = TextConditionedCTDataset(
        dataset='mayo_2016',
        mode='test',
        test_id=9,
        dose=25,
        context=True,
        use_text=True
    )
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    # 获取一个batch
    batch = next(iter(dataloader))
    
    real_img = batch['input']
    real_text_emb = batch['text_embedding']
    fake_timestep = torch.randint(0, 1000, (real_img.shape[0],))
    
    print(f"✅ Real data loaded")
    print(f"   Real image shape: {real_img.shape}")
    print(f"   Real text embedding shape: {real_text_emb.shape}")
    
    # 测试前向传播
    with torch.no_grad():
        # 创建假的 x_adjust (实际训练时会从 target 和 noise 生成)
        fake_x_adjust = torch.cat([real_img[:, :1], real_img[:, :1]], dim=1)  # [B, 2, H, W]
        output = model(real_img, fake_timestep, fake_x_adjust, real_text_emb, adjust=True)
    
    print(f"✅ Forward pass with real data successful!")
    print(f"   Output shape: {output.shape}")
    
except Exception as e:
    print(f"❌ Real data test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("🎉 ALL U-NET TESTS PASSED!")
print("="*60)
print("\n✅ 你的U-Net可以正确接收文本条件了!")
print("✅ 下一步可以修改训练脚本")