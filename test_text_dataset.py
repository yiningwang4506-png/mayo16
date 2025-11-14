"""
测试 TextConditionedCTDataset 
验证文本条件功能是否正常工作
"""
import torch
from torch.utils.data import DataLoader
import sys

# 确保能导入你的模块
sys.path.append('/root/autodl-tmp/CoreDiff-main')

from text_conditioned_dataset import TextConditionedCTDataset, text_dataset_dict


def test_single_sample():
    """测试1: 加载单个样本"""
    print("\n" + "="*60)
    print("🧪 TEST 1: Single Sample Loading")
    print("="*60)
    
    # 创建数据集 (测试集, dose=25)
    dataset = TextConditionedCTDataset(
        dataset='mayo_2016',
        mode='test',
        test_id=9,
        dose=25,
        context=True,
        use_text=True
    )
    
    print(f"\n📊 Dataset size: {len(dataset)}")
    
    # 加载第一个样本
    sample = dataset[0]
    
    print("\n✅ Sample structure:")
    print(f"  - Input shape: {sample['input'].shape}")  # 应该是 (3, 512, 512)
    print(f"  - Target shape: {sample['target'].shape}")  # 应该是 (1, 512, 512)
    print(f"  - Text embedding shape: {sample['text_embedding'].shape}")  # 应该是 (256,)
    print(f"  - Dose: {sample['dose']}%")
    print(f"  - Patient ID: L{sample['patient_id']:03d}")
    print(f"\n📝 Generated description:")
    print(f"  {sample['description']}")
    
    # 验证维度
    assert sample['input'].shape == (3, 512, 512), "❌ Input shape错误!"
    assert sample['target'].shape == (1, 512, 512), "❌ Target shape错误!"
    assert sample['text_embedding'].shape == (256,), "❌ Text embedding维度错误!"
    assert sample['dose'] in [25, 50, 100], "❌ Dose值异常!"
    
    print("\n✅ TEST 1 PASSED!")
    return dataset


def test_batch_loading(dataset):
    """测试2: 批量加载"""
    print("\n" + "="*60)
    print("🧪 TEST 2: Batch Loading with DataLoader")
    print("="*60)
    
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0  # 设为0避免多进程问题
    )
    
    for i, batch in enumerate(dataloader):
        print(f"\n📦 Batch {i+1}:")
        print(f"  - Input shape: {batch['input'].shape}")  # (4, 3, 512, 512)
        print(f"  - Target shape: {batch['target'].shape}")  # (4, 1, 512, 512)
        print(f"  - Text embedding shape: {batch['text_embedding'].shape}")  # (4, 256)
        print(f"  - Doses: {batch['dose'].tolist()}")
        
        # 验证batch维度
        assert batch['input'].shape[0] == 4, "❌ Batch size不对!"
        assert batch['text_embedding'].shape == (4, 256), "❌ Batch text embedding shape错误!"
        
        # 只测试第一个batch
        if i == 0:
            break
    
    print("\n✅ TEST 2 PASSED!")


def test_multi_dose():
    """测试3: 多剂量训练"""
    print("\n" + "="*60)
    print("🧪 TEST 3: Multi-Dose Training")
    print("="*60)
    
    # 创建多剂量数据集
    dataset = TextConditionedCTDataset(
        dataset='mayo_2016',
        mode='train',
        test_id=9,
        dose=[25, 50],  # 同时使用25%和50%
        context=True,
        use_text=True
    )
    
    print(f"\n📊 Total samples (25% + 50%): {len(dataset)}")
    
    # 统计不同剂量的样本数
    dose_counts = {25: 0, 50: 0}
    descriptions_by_dose = {25: [], 50: []}
    
    # 采样前10个样本检查
    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        dose = sample['dose']
        dose_counts[dose] = dose_counts.get(dose, 0) + 1
        descriptions_by_dose[dose].append(sample['description'])
    
    print(f"\n📊 Dose distribution in first 10 samples:")
    for dose, count in dose_counts.items():
        print(f"  - Dose {dose}%: {count} samples")
    
    print(f"\n📝 Example descriptions:")
    for dose in [25, 50]:
        if descriptions_by_dose[dose]:
            print(f"\n  Dose {dose}%:")
            print(f"  {descriptions_by_dose[dose][0][:200]}...")
    
    print("\n✅ TEST 3 PASSED!")


def test_text_embedding_diversity():
    """测试4: 验证不同dose的text embedding确实不同"""
    print("\n" + "="*60)
    print("🧪 TEST 4: Text Embedding Diversity")
    print("="*60)
    
    # 分别加载25%和50%的数据
    dataset_25 = TextConditionedCTDataset(
        dataset='mayo_2016', mode='test', test_id=9,
        dose=25, context=True, use_text=True
    )
    
    dataset_50 = TextConditionedCTDataset(
        dataset='mayo_2016', mode='test', test_id=9,
        dose=50, context=True, use_text=True
    )
    
    # 获取样本
    sample_25 = dataset_25[0]
    sample_50 = dataset_50[0]
    
    # 计算embedding的余弦相似度
    emb_25 = sample_25['text_embedding']
    emb_50 = sample_50['text_embedding']
    
    cosine_sim = torch.nn.functional.cosine_similarity(
        emb_25.unsqueeze(0),
        emb_50.unsqueeze(0)
    ).item()
    
    print(f"\n📊 Text embedding comparison:")
    print(f"  - Dose 25% embedding norm: {emb_25.norm().item():.4f}")
    print(f"  - Dose 50% embedding norm: {emb_50.norm().item():.4f}")
    print(f"  - Cosine similarity: {cosine_sim:.4f}")
    
    print(f"\n📝 Descriptions:")
    print(f"  25%: {sample_25['description'][:150]}...")
    print(f"  50%: {sample_50['description'][:150]}...")
    
    # 验证embedding确实不同
    assert cosine_sim < 0.99, "❌ 不同dose的embedding太相似!"
    print(f"\n✅ Embeddings are different (similarity={cosine_sim:.4f} < 0.99)")
    
    print("\n✅ TEST 4 PASSED!")


def test_backward_compatibility():
    """测试5: 验证向后兼容性 (use_text=False)"""
    print("\n" + "="*60)
    print("🧪 TEST 5: Backward Compatibility (Original Mode)")
    print("="*60)
    
    # 不使用文本条件
    dataset = TextConditionedCTDataset(
        dataset='mayo_2016',
        mode='test',
        test_id=9,
        dose=25,
        context=True,
        use_text=False  # 关闭文本条件
    )
    
    sample = dataset[0]
    
    # 原始模式应该返回tuple而不是dict
    assert isinstance(sample, tuple), "❌ 原始模式应该返回tuple!"
    assert len(sample) == 2, "❌ 应该返回(input, target)!"
    
    input_img, target_img = sample
    print(f"\n✅ Original mode (no text):")
    print(f"  - Input type: {type(input_img)}")
    print(f"  - Input shape: {input_img.shape}")
    print(f"  - Target shape: {target_img.shape}")
    
    print("\n✅ TEST 5 PASSED!")


def main():
    """运行所有测试"""
    print("\n" + "="*70)
    print("🚀 Starting TextConditionedCTDataset Test Suite")
    print("="*70)
    
    try:
        # 测试1: 单样本加载
        dataset = test_single_sample()
        
        # 测试2: 批量加载
        test_batch_loading(dataset)
        
        # 测试3: 多剂量训练
        test_multi_dose()
        
        # 测试4: embedding多样性
        test_text_embedding_diversity()
        
        # 测试5: 向后兼容
        test_backward_compatibility()
        
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED!")
        print("="*70)
        print("\n✅ 你的 TextConditionedCTDataset 已经可以使用了!")
        print("✅ 下一步可以开始集成到训练流程中")
        
    except Exception as e:
        print("\n" + "="*70)
        print("❌ TEST FAILED!")
        print("="*70)
        print(f"\n错误信息: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)