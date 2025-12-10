"""
最优条件数据集 - 直接替换原来的 BERT 版本
"""
import os
import os.path as osp
from glob import glob
from torch.utils.data import Dataset
import numpy as np
import torch


class MultiDeviceOptimalDataset(Dataset):
    """
    使用可学习 embedding 的多设备数据集
    直接替换 MultiDeviceTextConditionedDataset
    """
    
    # 类别映射
    DEVICE_MAP = {'LZU_PH': 0, 'ZJU_GE': 1, 'ZJU_UI': 2}
    DOSE_MAP = {10: 0, 25: 1}
    
    def __init__(self, 
                 data_root='./data',
                 mode='train',
                 devices=None,
                 doses=None,
                 num_test_patients=4,
                 context=True,
                 text_emb_dim=256,  # 保持接口兼容
                 **kwargs):         # 忽略 use_bert 等参数
        
        self.mode = mode
        self.context = context
        self.data_root = data_root
        self.text_emb_dim = text_emb_dim
        
        if devices is None:
            devices = ['LZU_PH', 'ZJU_GE', 'ZJU_UI']
        if doses is None:
            doses = [10, 25]
        
        self.devices = devices
        self.doses = doses
        
        print(f"{'='*60}")
        print(f"📊 MultiDeviceOptimalDataset 初始化")
        print(f"{'='*60}")
        print(f"  Mode: {mode}")
        print(f"  Devices: {devices}")
        print(f"  Doses: {doses}")
        print(f"  Context: {context}")
        
        self.input_files = []
        self.target_files = []
        self.metadata = []  # 存储 device_idx, dose_idx
        
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
                
                test_pids = patient_ids[-num_test_patients:]
                train_pids = patient_ids[:-num_test_patients]
                
                selected_pids = train_pids if mode == 'train' else test_pids
                
                for pid in selected_pids:
                    patient_inputs = sorted(glob(osp.join(data_dir, f"{pid}_*_dose{dose}.npy")))
                    patient_targets = sorted(glob(osp.join(data_dir, f"{pid}_*_target.npy")))
                    
                    if len(patient_inputs) != len(patient_targets) or len(patient_inputs) < 3:
                        continue
                    
                    if context:
                        for i in range(1, len(patient_inputs) - 1):
                            triple = f"{patient_inputs[i-1]}~{patient_inputs[i]}~{patient_inputs[i+1]}"
                            self.input_files.append(triple)
                            self.target_files.append(patient_targets[i])
                            self.metadata.append({
                                'device': device,
                                'dose': dose,
                                'device_idx': self.DEVICE_MAP[device],
                                'dose_idx': self.DOSE_MAP[dose],
                            })
                    else:
                        for i in range(1, len(patient_inputs) - 1):
                            self.input_files.append(patient_inputs[i])
                            self.target_files.append(patient_targets[i])
                            self.metadata.append({
                                'device': device,
                                'dose': dose,
                                'device_idx': self.DEVICE_MAP[device],
                                'dose_idx': self.DOSE_MAP[dose],
                            })
                
                count = len([m for m in self.metadata 
                           if m['device'] == device and m['dose'] == dose])
                print(f"  ✅ {dir_name}: {len(selected_pids)} patients, {count} samples")
        
        print(f"{'='*60}")
        print(f"📊 总计: {len(self.input_files)} 样本")
        print(f"{'='*60}")
    
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
        
        return {
            'input': input_img.astype(np.float32),
            'target': target_img.astype(np.float32),
            'device_idx': meta['device_idx'],
            'dose_idx': meta['dose_idx'],
            'device': meta['device'],
            'dose': meta['dose'],
        }
    
    def __len__(self):
        return len(self.input_files)
    
    def normalize_(self, img, MIN_B=0, MAX_B=4000):
        img = np.clip(img, MIN_B, MAX_B)
        return img / MAX_B


# 测试
if __name__ == '__main__':
    ds = MultiDeviceOptimalDataset(
        data_root='./data',
        mode='train',
        devices=['LZU_PH', 'ZJU_GE', 'ZJU_UI'],
        doses=[10, 25],
    )
    
    sample = ds[0]
    print(f"\nSample:")
    print(f"  Input: {sample['input'].shape}")
    print(f"  Target: {sample['target'].shape}")
    print(f"  Device idx: {sample['device_idx']}")
    print(f"  Dose idx: {sample['dose_idx']}")
    print(f"  Device: {sample['device']}, Dose: {sample['dose']}")