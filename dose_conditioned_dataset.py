"""
Dose-Conditioned CT Dataset
支持 Dose Embedding 的数据集，返回剂量值用于条件生成
"""
import os
import os.path as osp
from glob import glob
from torch.utils.data import Dataset
import numpy as np
import torch
from functools import partial
import re


class DoseConditionedCTDataset(Dataset):
    """
    支持 Dose Embedding 的CT数据集
    返回 dict 格式，包含 dose 值
    """
    def __init__(self, dataset, mode, test_id=9, dose=5, context=True, use_text=True):
        """
        Args:
            dataset: 数据集名称 ('mayo_2016', 'mayo_2016_sim', etc.)
            mode: 'train' 或 'test'
            test_id: 测试集患者ID索引
            dose: 剂量值，支持 int, str, list
            context: 是否使用上下文 (3帧)
            use_text: 是否启用 dose conditioning (保持参数名兼容)
        """
        self.mode = mode
        self.context = context
        self.use_dose_condition = use_text  # 复用 use_text 参数
        
        print(f"📊 Initializing DoseConditioned dataset: {dataset}, mode: {mode}, dose: {dose}")
        print(f"🎯 Dose conditioning: {'ENABLED' if self.use_dose_condition else 'DISABLED'}")

        # === Mayo 2016 / Sim 数据集处理 ===
        if dataset in ['mayo_2016_sim', 'mayo_2016']:
            if dataset == 'mayo_2016_sim':
                data_root = './data_preprocess/gen_data/mayo_2016_sim_npy'
                self.dataset_name = 'mayo_2016_sim'
            else:
                data_root = '/root/autodl-tmp/CoreDiff-main/data/Mayo16_SM_dose25_and_dose50'
                self.dataset_name = 'mayo_2016'

            patient_ids = [67, 96, 109, 143, 192, 286, 291, 310, 333, 506]

            if mode == 'train':
                patient_ids.pop(test_id)
            else:
                patient_ids = patient_ids[test_id:test_id+1]

            # 统一解析 dose 参数
            if isinstance(dose, (list, tuple)):
                dose_list = dose
            elif isinstance(dose, str):
                dose_list = [int(v.strip()) for v in dose.split(',')]
            else:
                dose_list = [dose]

            print(f"✅ Training with dose levels: {dose_list}")

            patient_input_lists = []
            patient_target_lists = []
            patient_dose_lists = []  # 🔥 记录每个样本的 dose 值

            for pid in patient_ids:
                target_files = sorted(glob(osp.join(data_root, f"L{pid:03d}_*_target.npy")))

                for d in dose_list:
                    input_files = sorted(glob(osp.join(data_root, f"L{pid:03d}_*_dose{d}.npy")))

                    if len(input_files) != len(target_files):
                        print(f"⚠️ Mismatch: patient {pid}, dose {d} → {len(input_files)} vs {len(target_files)}")

                    if context:
                        if len(input_files) > 2:
                            for i in range(1, len(input_files)-1):
                                triple = f"{input_files[i-1]}~{input_files[i]}~{input_files[i+1]}"
                                patient_input_lists.append(triple)
                                patient_target_lists.append(target_files[i])
                                patient_dose_lists.append(d)  # 🔥 记录 dose
                    else:
                        if len(input_files) > 2:
                            for i in range(1, len(input_files)-1):
                                patient_input_lists.append(input_files[i])
                                patient_target_lists.append(target_files[i])
                                patient_dose_lists.append(d)  # 🔥 记录 dose

            self.input = patient_input_lists
            self.target = patient_target_lists
            self.doses = patient_dose_lists  # 🔥 存储 dose 列表

            print(f"✅ Loaded from: {data_root}")
            print(f"✅ Mixed doses: {dose_list}")
            print(f"✅ Total samples: {len(self.input)}")

            if len(self.input) != len(self.target):
                raise ValueError("Input/Target counts mismatch")

        # Mayo 2020
        elif dataset == 'mayo_2020':
            self.dataset_name = 'mayo_2020'
            data_root = './data_preprocess/gen_data/mayo_2020_npy'

            # 统一解析 dose 参数
            if isinstance(dose, (list, tuple)):
                dose_val = dose[0]  # mayo_2020 只支持单剂量
            elif isinstance(dose, str):
                dose_val = int(dose.split(',')[0])
            else:
                dose_val = dose

            if dose_val == 10:
                patient_ids = ['C052', 'C232', 'C016', 'C120', 'C050']
            else:
                patient_ids = ['L077', 'L056', 'L186', 'L006', 'L148']

            base_target = []
            base_input = []
            base_doses = []

            for id in patient_ids:
                plist = sorted(glob(osp.join(data_root, id + '_target_*_img.npy')))
                base_target += plist[1:-1]

            for id in patient_ids:
                plist = sorted(glob(osp.join(data_root, id + f"_{dose_val}_" + '*_img.npy')))
                if context:
                    cat_list = []
                    for i in range(1, len(plist)-1):
                        triple = '~'.join([plist[i+j] for j in [-1,0,1]])
                        cat_list.append(triple)
                        base_doses.append(dose_val)
                    base_input += cat_list
                else:
                    base_input += plist[1:-1]
                    base_doses += [dose_val] * (len(plist) - 2)

            self.input = base_input
            self.target = base_target
            self.doses = base_doses

            print(f"✅ Inputs: {len(self.input)}, Targets: {len(self.target)}")

    def __getitem__(self, index):
        input_path, target_path = self.input[index], self.target[index]
        dose_value = self.doses[index]  # 🔥 获取 dose 值

        # === 加载图像 ===
        if self.context:
            paths = input_path.split('~')
            imgs = [np.load(p)[np.newaxis, ...].astype(np.float32) for p in paths]
            input_img = np.concatenate(imgs, axis=0)  # (3,H,W)
        else:
            input_img = np.load(input_path)[np.newaxis, ...].astype(np.float32)

        target_img = np.load(target_path)[np.newaxis, ...].astype(np.float32)

        # normalize
        input_img = self.normalize_(input_img)
        target_img = self.normalize_(target_img)

        # 🔥 返回 dict 格式，包含 dose
        if self.use_dose_condition:
            return {
                'input': input_img.astype(np.float32),
                'target': target_img.astype(np.float32),
                'dose': dose_value,  # 🔥 int 类型
            }
        else:
            # 向后兼容：返回 tuple
            return input_img, target_img

    def __len__(self):
        return len(self.target)

    def normalize_(self, img, MIN_B=-1024, MAX_B=3072):
        img = img - 1024
        img = np.clip(img, MIN_B, MAX_B)
        return (img - MIN_B) / (MAX_B - MIN_B)


# === 便捷函数 ===
def create_dose_conditioned_dataset(dataset='mayo_2016', mode='train',
                                    test_id=9, dose=25, context=True, use_text=True):
    return DoseConditionedCTDataset(
        dataset=dataset,
        mode=mode,
        test_id=test_id,
        dose=dose,
        context=context,
        use_text=use_text,
    )


# === dict wrapper (兼容 basic_template.py) ===
dose_dataset_dict = {
    'train': partial(create_dose_conditioned_dataset,
                     dataset='mayo_2016', mode='train',
                     test_id=9, dose=[25, 50],
                     context=True, use_text=True),

    'test_25': partial(create_dose_conditioned_dataset,
                       dataset='mayo_2016', mode='test',
                       test_id=9, dose=25,
                       context=True, use_text=True),

    'test_50': partial(create_dose_conditioned_dataset,
                       dataset='mayo_2016', mode='test',
                       test_id=9, dose=50,
                       context=True, use_text=True),
}


# === 测试 ===
if __name__ == '__main__':
    print("="*60)
    print("🧪 Testing DoseConditionedCTDataset")
    print("="*60)
    
    # 测试多剂量数据集
    dataset = DoseConditionedCTDataset(
        dataset='mayo_2016',
        mode='train',
        test_id=9,
        dose=[25, 50],
        context=True,
        use_text=True
    )
    
    print(f"\n📊 Dataset size: {len(dataset)}")
    
    # 获取几个样本
    for i in [0, 100, len(dataset)-1]:
        if i < len(dataset):
            sample = dataset[i]
            print(f"\nSample {i}:")
            print(f"  Input shape: {sample['input'].shape}")
            print(f"  Target shape: {sample['target'].shape}")
            print(f"  Dose: {sample['dose']}%")
    
    # 统计 dose 分布
    dose_counts = {}
    for i in range(min(1000, len(dataset))):
        d = dataset[i]['dose']
        dose_counts[d] = dose_counts.get(d, 0) + 1
    
    print(f"\n📈 Dose distribution (first 1000 samples):")
    for d, count in sorted(dose_counts.items()):
        print(f"  {d}%: {count} samples")
    
    print("\n✅ Test passed!")
