"""
Multi-Device Text-Conditioned CT Dataset
支持多设备、多剂量的文本条件数据集
"""
import os
import os.path as osp
from glob import glob
from torch.utils.data import Dataset
import numpy as np
import torch
from functools import partial


# ============================================================
# 文本描述生成器
# ============================================================
class MultiDeviceTextGenerator:
    """
    根据设备和剂量生成医学文本描述
    """
    
    # 设备描述
    DEVICE_DESCRIPTIONS = {
        'LZU_PH': 'Philips scanner',
        'ZJU_GE': 'GE scanner', 
        'ZJU_UI': 'United Imaging scanner',
    }
    
    # 剂量描述
    DOSE_DESCRIPTIONS = {
        10: '10% ultra-low-dose with high noise level',
        25: '25% low-dose with moderate noise level',
        50: '50% reduced-dose with mild noise level',
        100: 'full-dose with standard noise level',
    }
    
    @staticmethod
    def generate_description(device, dose):
        """
        生成文本描述
        
        Args:
            device: 'LZU_PH', 'ZJU_GE', 或 'ZJU_UI'
            dose: 10 或 25
            
        Returns:
            str: 文本描述
        """
        device_desc = MultiDeviceTextGenerator.DEVICE_DESCRIPTIONS.get(
            device, f'{device} scanner'
        )
        dose_desc = MultiDeviceTextGenerator.DOSE_DESCRIPTIONS.get(
            dose, f'{dose}% dose'
        )
        
        description = (
            f"This CT scan was acquired using a {device_desc} "
            f"with {dose_desc}."
        )
        
        return description


# ============================================================
# 文本编码器（使用 PubMed-BERT）
# ============================================================
class TextEncoder:
    """
    使用 PubMed-BERT 编码文本
    """
    
    def __init__(self, 
                 model_name='microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
                 output_dim=256,
                 cache_dir='./pretrained_models'):
        
        from transformers import AutoTokenizer, AutoModel
        import os
        
        # 设置离线模式（如果已下载）
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        os.environ['HF_HUB_OFFLINE'] = '1'
        
        print(f"🔄 Loading text encoder: {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=cache_dir,
                local_files_only=True
            )
            self.bert = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=True
            )
            print("✅ Text encoder loaded from local cache")
        except Exception as e:
            print(f"⚠️ Local cache not found, downloading...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=cache_dir,
                local_files_only=False
            )
            self.bert = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                local_files_only=False
            )
            print("✅ Text encoder downloaded and loaded")
        
        self.bert.eval()
        for param in self.bert.parameters():
            param.requires_grad = False
        
        # 投影层
        bert_dim = self.bert.config.hidden_size
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(bert_dim, output_dim * 2),
            torch.nn.GELU(),
            torch.nn.Linear(output_dim * 2, output_dim),
        )
        
        self.output_dim = output_dim
        self._cache = {}  # 缓存编码结果
        
    def encode(self, text):
        """编码单个文本"""
        if text in self._cache:
            return self._cache[text]
        
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=128
            )
            
            outputs = self.bert(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :]  # [1, 768]
            text_emb = self.projection(cls_emb)  # [1, 256]
            text_emb = text_emb.squeeze(0).numpy()  # [256]
        
        self._cache[text] = text_emb
        return text_emb


# ============================================================
# 简化版文本编码器（不需要 BERT，用于快速测试）
# ============================================================
class SimpleTextEncoder:
    """
    简化版文本编码器
    直接用 one-hot 编码设备和剂量
    """
    
    def __init__(self, output_dim=256):
        self.output_dim = output_dim
        
        # 设备编码 (3种)
        self.device_map = {'LZU_PH': 0, 'ZJU_GE': 1, 'ZJU_UI': 2}
        
        # 剂量编码 (2种)
        self.dose_map = {10: 0, 25: 1}
        
        # 创建固定的 embedding
        np.random.seed(42)
        self.device_embeddings = np.random.randn(3, output_dim // 2).astype(np.float32)
        self.dose_embeddings = np.random.randn(2, output_dim // 2).astype(np.float32)
        
        # 归一化
        self.device_embeddings /= np.linalg.norm(self.device_embeddings, axis=1, keepdims=True)
        self.dose_embeddings /= np.linalg.norm(self.dose_embeddings, axis=1, keepdims=True)
        
    def encode(self, device, dose):
        """
        编码设备和剂量
        
        Returns:
            np.ndarray: [output_dim]
        """
        device_idx = self.device_map.get(device, 0)
        dose_idx = self.dose_map.get(dose, 0)
        
        device_emb = self.device_embeddings[device_idx]
        dose_emb = self.dose_embeddings[dose_idx]
        
        # 拼接
        text_emb = np.concatenate([device_emb, dose_emb])
        
        return text_emb


# ============================================================
# 多设备文本条件数据集
# ============================================================
class MultiDeviceTextConditionedDataset(Dataset):
    """
    多设备多剂量文本条件数据集
    """
    
    def __init__(self, 
                 data_root='./data',
                 mode='train',
                 devices=None,
                 doses=None,
                 num_test_patients=4,
                 context=True,
                 use_bert=True,  # 是否使用 BERT 编码
                 text_emb_dim=256):
        
        self.mode = mode
        self.context = context
        self.data_root = data_root
        self.use_bert = use_bert
        
        # 默认使用所有设备和剂量
        if devices is None:
            devices = ['LZU_PH', 'ZJU_GE', 'ZJU_UI']
        if doses is None:
            doses = [10, 25]
        
        print(f"{'='*60}")
        print(f"📊 MultiDeviceTextConditionedDataset 初始化")
        print(f"{'='*60}")
        print(f"  Mode: {mode}")
        print(f"  Devices: {devices}")
        print(f"  Doses: {doses}")
        print(f"  Context: {context}")
        print(f"  Use BERT: {use_bert}")
        
        # 初始化文本编码器
        if use_bert:
            print("🔄 Initializing BERT text encoder...")
            self.text_encoder = TextEncoder(output_dim=text_emb_dim)
        else:
            print("🔄 Using simple text encoder...")
            self.text_encoder = SimpleTextEncoder(output_dim=text_emb_dim)
        
        self.text_generator = MultiDeviceTextGenerator()
        
        self.input_files = []
        self.target_files = []
        self.metadata = []
        
        # 遍历每个设备和剂量
        for device in devices:
            for dose in doses:
                dir_name = f"{device}_dose{dose}"
                data_dir = osp.join(data_root, dir_name, dir_name)
                
                if not osp.exists(data_dir):
                    print(f"  ⚠️ 目录不存在: {data_dir}")
                    continue
                
                all_files = sorted(glob(osp.join(data_dir, f"*_dose{dose}.npy")))
                if not all_files:
                    continue
                
                patient_ids = sorted(list(set([
                    osp.basename(f).split('_')[0] for f in all_files
                ])))
                
                # 划分训练/测试
                test_pids = patient_ids[-num_test_patients:]
                train_pids = patient_ids[:-num_test_patients]
                
                if mode == 'train':
                    selected_pids = train_pids
                else:
                    selected_pids = test_pids
                
                for pid in selected_pids:
                    patient_inputs = sorted(glob(
                        osp.join(data_dir, f"{pid}_*_dose{dose}.npy")
                    ))
                    patient_targets = sorted(glob(
                        osp.join(data_dir, f"{pid}_*_target.npy")
                    ))
                    
                    if len(patient_inputs) != len(patient_targets):
                        continue
                    
                    if len(patient_inputs) < 3:
                        continue
                    
                    if context:
                        for i in range(1, len(patient_inputs) - 1):
                            triple = f"{patient_inputs[i-1]}~{patient_inputs[i]}~{patient_inputs[i+1]}"
                            self.input_files.append(triple)
                            self.target_files.append(patient_targets[i])
                            self.metadata.append({
                                'device': device,
                                'dose': dose,
                                'patient': pid,
                                'slice': i
                            })
                    else:
                        for i in range(1, len(patient_inputs) - 1):
                            self.input_files.append(patient_inputs[i])
                            self.target_files.append(patient_targets[i])
                            self.metadata.append({
                                'device': device,
                                'dose': dose,
                                'patient': pid,
                                'slice': i
                            })
                
                count = len([m for m in self.metadata 
                           if m['device'] == device and m['dose'] == dose])
                print(f"  ✅ {dir_name}: {len(selected_pids)} patients, {count} samples")
        
        print(f"{'='*60}")
        print(f"📊 总计: {len(self.input_files)} 样本")
        print(f"{'='*60}")
        
        # 预计算所有文本 embedding（加速训练）
        print("🔄 Pre-computing text embeddings...")
        self._precompute_text_embeddings()
        print("✅ Text embeddings ready!")
    
    def _precompute_text_embeddings(self):
        """预计算所有设备-剂量组合的文本 embedding"""
        self.text_embeddings = {}
        
        devices = set(m['device'] for m in self.metadata)
        doses = set(m['dose'] for m in self.metadata)
        
        for device in devices:
            for dose in doses:
                if self.use_bert:
                    desc = self.text_generator.generate_description(device, dose)
                    emb = self.text_encoder.encode(desc)
                else:
                    emb = self.text_encoder.encode(device, dose)
                
                key = f"{device}_{dose}"
                self.text_embeddings[key] = emb
                print(f"    {key}: {emb.shape}")
    
    def __getitem__(self, index):
        input_path = self.input_files[index]
        target_path = self.target_files[index]
        meta = self.metadata[index]
        
        # 加载图像
        if self.context:
            paths = input_path.split('~')
            imgs = [np.load(p)[np.newaxis, ...].astype(np.float32) for p in paths]
            input_img = np.concatenate(imgs, axis=0)
        else:
            input_img = np.load(input_path)[np.newaxis, ...].astype(np.float32)
        
        target_img = np.load(target_path)[np.newaxis, ...].astype(np.float32)
        
        # 归一化
        input_img = self.normalize_(input_img)
        target_img = self.normalize_(target_img)
        
        # 获取文本 embedding
        key = f"{meta['device']}_{meta['dose']}"
        text_emb = self.text_embeddings[key]
        
        return {
            'input': input_img.astype(np.float32),
            'target': target_img.astype(np.float32),
            'text_embedding': text_emb.astype(np.float32),
            'device': meta['device'],
            'dose': meta['dose'],
        }
    
    def __len__(self):
        return len(self.input_files)
    
    def normalize_(self, img, MIN_B=0, MAX_B=4000):
        img = np.clip(img, MIN_B, MAX_B)
        return img / MAX_B


# ============================================================
# 测试代码
# ============================================================
if __name__ == '__main__':
    print("\n" + "="*70)
    print("🧪 测试 MultiDeviceTextConditionedDataset")
    print("="*70)
    
    # 先用简单编码器测试（不需要 BERT）
    print("\n【测试简单编码器】")
    dataset = MultiDeviceTextConditionedDataset(
        data_root='./data',
        mode='train',
        devices=['LZU_PH', 'ZJU_GE', 'ZJU_UI'],
        doses=[10, 25],
        num_test_patients=4,
        context=True,
        use_bert=False,  # 不用 BERT，快速测试
        text_emb_dim=256
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    
    # 测试样本
    sample = dataset[0]
    print(f"\n样本内容:")
    print(f"  Input shape: {sample['input'].shape}")
    print(f"  Target shape: {sample['target'].shape}")
    print(f"  Text embedding shape: {sample['text_embedding'].shape}")
    print(f"  Device: {sample['device']}")
    print(f"  Dose: {sample['dose']}%")
    
    # 验证不同设备/剂量的 embedding 不同
    print(f"\n【验证 embedding 区分度】")
    emb_lzu_10 = dataset.text_embeddings['LZU_PH_10']
    emb_lzu_25 = dataset.text_embeddings['LZU_PH_25']
    emb_ge_10 = dataset.text_embeddings['ZJU_GE_10']
    
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    print(f"  LZU_PH 10% vs 25%: {cosine_sim(emb_lzu_10, emb_lzu_25):.4f}")
    print(f"  LZU_PH 10% vs GE 10%: {cosine_sim(emb_lzu_10, emb_ge_10):.4f}")
    
    print("\n✅ 测试完成！")