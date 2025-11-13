import os
import os.path as osp
from glob import glob
from torch.utils.data import Dataset
import numpy as np
from functools import partial


class CTDataset(Dataset):
    def __init__(self, dataset, mode, test_id=9, dose=5, context=True):
        self.mode = mode
        self.context = context
        print(f"📊 Initializing dataset: {dataset}, mode: {mode}, dose: {dose}")

        if dataset in ['mayo_2016_sim', 'mayo_2016']:
            if dataset == 'mayo_2016_sim':
                data_root = './data_preprocess/gen_data/mayo_2016_sim_npy'
            elif dataset == 'mayo_2016':
                data_root = '/root/autodl-tmp/CoreDiff-main/data/Mayo16_SM_dose25_and_dose50'

            # 病人列表
            patient_ids = [67, 96, 109, 143, 192, 286, 291, 310, 333, 506]
            if mode == 'train':
                patient_ids.pop(test_id)
            elif mode == 'test':
                patient_ids = patient_ids[test_id:test_id + 1]

            # === 解析 dose 参数 ===
            if isinstance(dose, (list, tuple)):
                dose_list = dose
            elif isinstance(dose, str):
                dose_list = [int(d.strip()) for d in dose.split(',')]
            else:
                dose_list = [dose]
            
            print(f"✅ Training with dose levels: {dose_list}")

            # === 核心修复: 为每个病人单独处理,保持时序连续性 ===
            patient_input_lists = []
            patient_target_lists = []
            
            for patient_id in patient_ids:
                # 加载该病人的所有target文件(不分dose,target是固定的)
                target_files = sorted(glob(osp.join(data_root, f'L{patient_id:03d}_*_target.npy')))
                
                # 对每个dose level分别处理
                for d in dose_list:
                    # 加载该病人在该dose下的所有输入文件
                    input_files = sorted(glob(osp.join(data_root, f'L{patient_id:03d}_*_dose{d}.npy')))
                    
                    # 确保input和target数量匹配
                    if len(input_files) != len(target_files):
                        print(f"⚠️ Patient {patient_id}, dose {d}: input={len(input_files)}, target={len(target_files)}")
                    
                    # 如果使用context,构建三联输入
                    if context:
                        # 需要去掉首尾,因为首尾无法构成三联
                        if len(input_files) > 2:
                            for i in range(1, len(input_files) - 1):
                                # 构建三联: [i-1, i, i+1]
                                triple_path = f"{input_files[i-1]}~{input_files[i]}~{input_files[i+1]}"
                                patient_input_lists.append(triple_path)
                                patient_target_lists.append(target_files[i])
                    else:
                        # 不使用context,直接使用单帧(去掉首尾)
                        if len(input_files) > 2:
                            for i in range(1, len(input_files) - 1):
                                patient_input_lists.append(input_files[i])
                                patient_target_lists.append(target_files[i])

            self.input = patient_input_lists
            self.target = patient_target_lists
            
            print(f"✅ Loaded from: {data_root}")
            print(f"✅ Mixed doses: {dose_list}")
            print(f"✅ Total inputs: {len(self.input)}, Total targets: {len(self.target)}")
            
            # ✅ 检查数量是否匹配
            if len(self.input) != len(self.target):
                print(f"❌ ERROR: Input count ({len(self.input)}) != Target count ({len(self.target)})")
                raise ValueError("Input and target counts must match!")

        # 其他数据集（原样保留）
        elif dataset == 'mayo_2020':
            data_root = './data_preprocess/gen_data/mayo_2020_npy'
            if dose == 10:
                patient_ids = ['C052', 'C232', 'C016', 'C120', 'C050']
            elif dose == 25:
                patient_ids = ['L077', 'L056', 'L186', 'L006', 'L148']

            patient_lists = []
            for id in patient_ids:
                patient_list = sorted(glob(osp.join(data_root, (id + '_target_' + '*_img.npy'))))
                patient_lists += patient_list[1:len(patient_list) - 1]
            base_target = patient_lists

            patient_lists = []
            for id in patient_ids:
                patient_list = sorted(glob(osp.join(data_root, (id + '_{}_'.format(dose) + '*_img.npy'))))
                if context:
                    cat_patient_list = []
                    for i in range(1, len(patient_list) - 1):
                        patient_path = ''
                        for j in range(-1, 2):
                            patient_path = patient_path + '~' + patient_list[i + j]
                        cat_patient_list.append(patient_path)
                    patient_lists += cat_patient_list
                else:
                    patient_list = patient_list[1:len(patient_list) - 1]
                    patient_lists += patient_list
            base_input = patient_lists
            
            self.input = base_input
            self.target = base_target
            print(f"✅ Inputs: {len(self.input)}, Targets: {len(self.target)}")

    def __getitem__(self, index):
        input, target = self.input[index], self.target[index]
        if self.context:
            input = input.split('~')
            inputs = []
            for i in range(len(input)):  # 修改: 从0开始,而不是从1开始
                inputs.append(np.load(input[i])[np.newaxis, ...].astype(np.float32))
            input = np.concatenate(inputs, axis=0)  # (3,512,512)
        else:
            input = np.load(input)[np.newaxis, ...].astype(np.float32)
        target = np.load(target)[np.newaxis, ...].astype(np.float32)
        input = self.normalize_(input)
        target = self.normalize_(target)
        return input, target

    def __len__(self):
        return len(self.target)

    def normalize_(self, img, MIN_B=-1024, MAX_B=3072):
        img = img - 1024
        img[img < MIN_B] = MIN_B
        img[img > MAX_B] = MAX_B
        img = (img - MIN_B) / (MAX_B - MIN_B)
        return img


# 保留原来的 dict 作为后备
dataset_dict = {
    'train': partial(CTDataset, dataset='mayo_2016', mode='train', test_id=9, dose=25, context=True),
    'mayo_2016_sim': partial(CTDataset, dataset='mayo_2016_sim', mode='test', test_id=9, dose=5, context=True),
    'mayo_2016': partial(CTDataset, dataset='mayo_2016', mode='test', test_id=9, dose=25, context=True),
}