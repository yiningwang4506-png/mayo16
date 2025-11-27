import torch
import glob
import sys

print("="*70)
print("🚨 紧急诊断 - PSNR 过低")
print("="*70)

# 找到最新的实验
experiments = glob.glob('./output/corediff_*/save_models/model-2500')
if not experiments:
    print("❌ 没有找到 2500 步的 checkpoint")
    sys.exit(1)

latest = sorted(experiments)[-1]
exp_name = latest.split('/')[2]
print(f"\n📁 实验: {exp_name}")
print(f"📁 Checkpoint: {latest}\n")

# 1. 加载 checkpoint
ckpt = torch.load(latest, map_location='cpu')

# 2. 检查 loss 是否正常
print("[1/5] 检查训练 loss...")
print("-"*70)

# 从日志读取
log_files = glob.glob(f'./output/{exp_name}/logs/*.log')
if log_files:
    with open(log_files[0], 'r') as f:
        lines = f.readlines()[-50:]  # 最后50行
    
    losses = []
    for line in lines:
        if 'loss' in line.lower() and not 'psnr' in line.lower():
            try:
                # 尝试提取 loss 值
                parts = line.split(',')
                for part in parts:
                    if 'loss' in part.lower():
                        val = float(part.split()[-1])
                        losses.append(val)
                        break
            except:
                pass
    
    if losses:
        print(f"  最近的 loss 值:")
        print(f"    最小: {min(losses):.6f}")
        print(f"    最大: {max(losses):.6f}")
        print(f"    平均: {sum(losses)/len(losses):.6f}")
        print(f"    最后: {losses[-1]:.6f}")
        
        if losses[-1] > 0.05:
            print(f"\n  ❌ Loss 太高！{losses[-1]:.6f} > 0.05")
            print(f"     → 模型没有收敛")
        elif losses[-1] < 0.0001:
            print(f"\n  ⚠️  Loss 过小！{losses[-1]:.8f} < 0.0001")
            print(f"     → 可能过拟合或梯度消失")
        else:
            print(f"  ✅ Loss 范围正常")
else:
    print("  ⚠️  未找到日志文件")

# 3. 检查网络参数是否更新
print("\n[2/5] 检查网络参数...")
print("-"*70)

param_stats = []
for key, val in ckpt.items():
    if 'weight' in key and 'denoise_fn' in key and 'conv' in key:
        param_stats.append({
            'name': key,
            'max': val.abs().max().item(),
            'mean': val.abs().mean().item()
        })

if param_stats:
    # 按 max 排序
    param_stats.sort(key=lambda x: x['max'], reverse=True)
    
    print(f"  Top 5 参数范围:")
    for i, stat in enumerate(param_stats[:5]):
        print(f"    {i+1}. max={stat['max']:.4f}, mean={stat['mean']:.4f}")
    
    max_param = param_stats[0]['max']
    if max_param > 10:
        print(f"\n  ❌ 参数爆炸！max={max_param:.2f} > 10")
    elif max_param < 0.001:
        print(f"\n  ❌ 参数消失！max={max_param:.6f} < 0.001")
    else:
        print(f"\n  ✅ 参数范围正常")

# 4. 检查 FiLM 参数
print("\n[3/5] 检查 FiLM 参数...")
print("-"*70)

film_params = []
for key, val in ckpt.items():
    if 'film' in key.lower() and 'fc.2' in key:  # FiLM 输出层
        film_params.append({
            'name': key,
            'val': val,
            'max': val.abs().max().item(),
            'mean': val.abs().mean().item()
        })

if film_params:
    for fp in film_params:
        print(f"  {fp['name'].split('.')[-3]}:")
        print(f"    max={fp['max']:.6f}, mean={fp['mean']:.6f}")
    
    max_film = max([fp['max'] for fp in film_params])
    if max_film < 0.01:
        print(f"\n  ❌ FiLM 参数太小！max={max_film:.6f} < 0.01")
        print(f"     → FiLM 几乎不起作用")
    else:
        print(f"\n  ✅ FiLM 参数正常")
else:
    print("  ⚠️  未找到 FiLM 参数")

# 5. 测试实际前向传播
print("\n[4/5] 测试前向传播...")
print("-"*70)

try:
    from models.corediff.corediff_wrapper import Network
    
    # 创建新模型
    model = Network(in_channels=3, context=True, text_emb_dim=256).cuda()
    
    # 尝试加载 checkpoint（只加载匹配的部分）
    model_dict = model.state_dict()
    
    # 过滤出可以加载的参数
    pretrained_dict = {}
    for k, v in ckpt.items():
        # 移除 'denoise_fn.' 前缀
        new_k = k.replace('denoise_fn.', '')
        if new_k in model_dict and model_dict[new_k].shape == v.shape:
            pretrained_dict[new_k] = v
    
    print(f"  可加载参数: {len(pretrained_dict)}/{len(model_dict)}")
    
    if len(pretrained_dict) > 0:
        model.load_state_dict(pretrained_dict, strict=False)
        model.eval()
        
        # 测试输入
        x = torch.randn(1, 3, 256, 256).cuda()
        t = torch.tensor([5]).cuda()
        y = torch.randn(1, 1, 256, 256).cuda()
        x_end = torch.randn(1, 1, 256, 256).cuda()
        text_emb = torch.randn(1, 256).cuda()
        
        with torch.no_grad():
            out_no_text, _ = model(x, t, y, x_end, adjust=True, text_emb=None)
            out_with_text, _ = model(x, t, y, x_end, adjust=True, text_emb=text_emb)
        
        diff = (out_with_text - out_no_text).abs().mean().item()
        baseline = out_no_text.abs().mean().item()
        
        print(f"  输出范围: [{out_no_text.min().item():.4f}, {out_no_text.max().item():.4f}]")
        print(f"  文本条件影响: {diff:.6f}")
        print(f"  相对影响: {diff/baseline*100:.4f}%")
        
        if baseline < 0.01 or baseline > 10:
            print(f"\n  ❌ 输出范围异常！baseline={baseline:.4f}")
            print(f"     → 应该在 [0, 1] 范围内")
        else:
            print(f"  ✅ 输出范围正常")
            
        if diff/baseline < 0.0001:
            print(f"  ❌ 文本条件几乎无影响")
    else:
        print("  ❌ 无法加载任何参数（架构完全不匹配）")
        
except Exception as e:
    print(f"  ❌ 前向传播失败: {e}")
    import traceback
    traceback.print_exc()

# 6. 检查数据是否正确
print("\n[5/5] 检查数据加载...")
print("-"*70)

try:
    sys.path.append('.')
    from text_conditioned_dataset import TextConditionedCTDataset
    
    dataset = TextConditionedCTDataset(
        dataset='mayo_2016',
        mode='test',
        test_id=9,
        dose=25,
        context=True,
        use_text=True
    )
    
    sample = dataset[0]
    
    print(f"  数据集大小: {len(dataset)}")
    print(f"  输入形状: {sample['input'].shape}")
    print(f"  目标形状: {sample['target'].shape}")
    print(f"  输入范围: [{sample['input'].min():.4f}, {sample['input'].max():.4f}]")
    print(f"  目标范围: [{sample['target'].min():.4f}, {sample['target'].max():.4f}]")
    
    if sample['input'].max() > 2 or sample['input'].min() < -1:
        print(f"\n  ❌ 数据未归一化！范围应该在 [0, 1]")
    else:
        print(f"  ✅ 数据范围正常")
        
except Exception as e:
    print(f"  ❌ 数据加载失败: {e}")

# 终极诊断
print("\n" + "="*70)
print("🎯 可能的问题:")
print("="*70)

print("""
根据上述诊断，最可能的问题是：

1. 学习率问题
   - 如果 loss 不下降 → 学习率太小
   - 如果 loss 震荡或 NaN → 学习率太大

2. 数据问题
   - 如果输入/输出范围异常 → 数据预处理错误
   - 如果数据集很小 → 可能过拟合

3. FiLM 问题
   - 如果 FiLM 参数太小 → 没被训练
   - 如果文本条件无影响 → FiLM 初始化有问题

4. 架构问题
   - 如果无法加载参数 → 代码和 checkpoint 不匹配
   - 如果输出范围异常 → 模型输出层有问题

建议修复顺序：
  → 先不用 text condition，跑纯 baseline
  → 如果 baseline 正常，再加 text condition
  → 如果 baseline 也不行，检查数据和代码
""")

print("\n" + "="*70)
print("💡 下一步:")
print("="*70)
print("1. 查看上述诊断结果")
print("2. 如果是数据/架构问题，先修复")
print("3. 如果只是 FiLM 问题，可以先跑 baseline")
print("\n运行 baseline 测试:")
print("  bash train_baseline_test.sh")