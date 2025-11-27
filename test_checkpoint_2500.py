import torch
import sys
sys.path.append('.')

from models.corediff.corediff_wrapper import Network
from text_conditioned_dataset import TextConditionedCTDataset
from torch.utils.data import DataLoader

print("="*70)
print("🧪 手动测试 checkpoint-2500")
print("="*70)

# 1. 加载模型
print("\n[1/3] 加载模型...")
ckpt = torch.load('./output/corediff_text_FILM1123/save_models/ema_model-2500', map_location='cpu')

# 创建模型
from models.corediff.corediff import corediff
import argparse

# 模拟 args
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='corediff')
parser.add_argument('--in_channels', default=3, type=int)
parser.add_argument('--out_channels', default=1, type=int)
parser.add_argument('--context', action='store_true', default=True)
parser.add_argument('--T', default=10, type=int)
parser.add_argument('--sampling_routine', default='ddim')
parser.add_argument('--test_dataset', default='mayo_2016')
parser.add_argument('--test_id', default=9, type=int)
parser.add_argument('--dose', default='25')
parser.add_argument('--test_batch_size', default=1, type=int)
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--use_text_condition', action='store_true', default=True)

# DRL 参数
parser.add_argument('--reg_max', default=18, type=int)
parser.add_argument('--y_0', default=-160.0, type=float)
parser.add_argument('--y_n', default=240.0, type=float)
parser.add_argument('--norm_range_max', default=3072.0, type=float)
parser.add_argument('--norm_range_min', default=-1024.0, type=float)

opt = parser.parse_args([])

# 创建 Diffusion 模型
from models.corediff.diffusion_modules import Diffusion

denoise_fn = Network(
    in_channels=3,
    context=True,
    text_emb_dim=256,
    reg_max=18,
    y_0=-160,
    y_n=240,
    norm_range_max=3072,
    norm_range_min=-1024
)

model = Diffusion(
    denoise_fn=denoise_fn,
    image_size=512,
    timesteps=10,
    context=True
).cuda()

# 加载 checkpoint
model.load_state_dict(ckpt)
model.eval()

print("✅ 模型加载成功")

# 2. 加载测试数据
print("\n[2/3] 加载测试数据...")
test_dataset = TextConditionedCTDataset(
    dataset='mayo_2016',
    mode='test',
    test_id=9,
    dose=25,
    context=True,
    use_text=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

print(f"✅ 测试集大小: {len(test_dataset)}")

# 3. 测试并计算 PSNR
print("\n[3/3] 计算 PSNR...")

from utils.measure import compute_measure

psnrs = []
ssims = []
rmses = []

# 只测试前 10 个样本（快速）
for i, batch in enumerate(test_loader):
    if i >= 10:
        break
    
    low_dose = batch['input'].cuda()
    full_dose = batch['target'].cuda()
    text_emb = batch['text_embedding'].cuda()
    
    # 推理
    with torch.no_grad():
        gen_full_dose, _, _ = model.sample(
            batch_size=1,
            img=low_dose,
            t=10,
            sampling_routine='ddim',
            n_iter=2500,
            start_adjust_iter=1,
            text_emb=text_emb
        )
    
    # 反归一化
    MIN_B, MAX_B = -1024, 3072
    cut_min, cut_max = -1000, 1000
    
    full_dose = full_dose * (MAX_B - MIN_B) + MIN_B
    full_dose = torch.clamp(full_dose, cut_min, cut_max)
    full_dose = 255 * (full_dose - cut_min) / (cut_max - cut_min)
    
    gen_full_dose = gen_full_dose * (MAX_B - MIN_B) + MIN_B
    gen_full_dose = torch.clamp(gen_full_dose, cut_min, cut_max)
    gen_full_dose = 255 * (gen_full_dose - cut_min) / (cut_max - cut_min)
    
    data_range = full_dose.max() - full_dose.min()
    psnr, ssim, rmse = compute_measure(full_dose, gen_full_dose, data_range)
    
    psnrs.append(psnr)
    ssims.append(ssim)
    rmses.append(rmse)
    
    print(f"  样本 {i+1}: PSNR={psnr:.2f}, SSIM={ssim:.4f}, RMSE={rmse:.2f}")

# 计算平均
print("\n" + "="*70)
print("📊 测试结果 (前10个样本):")
print("="*70)
print(f"  PSNR: {sum(psnrs)/len(psnrs):.2f} dB")
print(f"  SSIM: {sum(ssims)/len(ssims):.4f}")
print(f"  RMSE: {sum(rmses)/len(rmses):.2f}")

print("\n🎯 诊断:")
avg_psnr = sum(psnrs)/len(psnrs)
if avg_psnr < 30:
    print(f"  ❌ PSNR 太低！{avg_psnr:.2f} < 30")
    print("     → 模型基本没学到东西")
elif avg_psnr < 35:
    print(f"  ⚠️  PSNR 偏低: {avg_psnr:.2f}")
    print("     → 训练还不够充分（2500步太早）")
elif avg_psnr < 40:
    print(f"  ✅ PSNR 正常: {avg_psnr:.2f}")
    print("     → 继续训练应该能到 41+")
else:
    print(f"  ✅ PSNR 很好: {avg_psnr:.2f}")