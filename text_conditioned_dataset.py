import os
import os.path as osp
from glob import glob
from torch.utils.data import Dataset
import numpy as np
import torch
from functools import partial
import re

# 导入你已经完成的模块
from text_description_generator import TextDescriptionGenerator
from medical_text_encoder import MedicalTextEncoder


class TextConditionedCTDataset(Dataset):
    """
    支持文本条件的CT数据集
    完全兼容 CoreDiff 的 basic_template.py
    （只返回 numpy，不返回 Tensor）
    """
    def __init__(self, dataset, mode, test_id=9, dose=5, context=True, use_text=True):
        self.mode = mode
        self.context = context
        self.use_text = use_text
        
        print(f"📊 Initializing TextConditioned dataset: {dataset}, mode: {mode}, dose: {dose}")
        print(f"🔤 Text conditioning: {'ENABLED' if use_text else 'DISABLED'}")

        # === 初始化文本模块 ===
        if self.use_text:
            print("🔄 Loading text encoder (PubMed-BERT)...")
            self.text_generator = TextDescriptionGenerator()
            self.text_encoder = MedicalTextEncoder()
            print("✅ Text encoder loaded successfully!")

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

            for pid in patient_ids:

                target_files = sorted(glob(osp.join(data_root, f"L{pid:03d}_*_target.npy")))

                for d in dose_list:
                    input_files = sorted(glob(osp.join(data_root, f"L{pid:03d}_*_dose{d}.npy")))

                    if len(input_files) != len(target_files):
                        print(f"⚠️ Mismatch: patient {pid}, dose {d} → {len(input_files)} vs {len(target_files)}")

                    # context 三联逻辑
                    if context:
                        if len(input_files) > 2:
                            for i in range(1, len(input_files)-1):
                                triple = f"{input_files[i-1]}~{input_files[i]}~{input_files[i+1]}"
                                patient_input_lists.append(triple)
                                patient_target_lists.append(target_files[i])
                    else:
                        if len(input_files) > 2:
                            for i in range(1, len(input_files)-1):
                                patient_input_lists.append(input_files[i])
                                patient_target_lists.append(target_files[i])

            self.input = patient_input_lists
            self.target = patient_target_lists

            print(f"✅ Loaded from: {data_root}")
            print(f"✅ Mixed doses: {dose_list}")
            print(f"✅ Total inputs: {len(self.input)}, Total targets: {len(self.target)}")

            if len(self.input) != len(self.target):
                raise ValueError("Input/Target counts mismatch")

        # Mayo 2020（保持原样）
        elif dataset == 'mayo_2020':
            self.dataset_name = 'mayo_2020'
            data_root = './data_preprocess/gen_data/mayo_2020_npy'

            if dose == 10:
                patient_ids = ['C052', 'C232', 'C016', 'C120', 'C050']
            else:
                patient_ids = ['L077', 'L056', 'L186', 'L006', 'L148']

            base_target = []
            for id in patient_ids:
                plist = sorted(glob(osp.join(data_root, id + '_target_*_img.npy')))
                base_target += plist[1:-1]

            base_input = []
            for id in patient_ids:
                plist = sorted(glob(osp.join(data_root, id + f"_{dose}_" + '*_img.npy')))
                if context:
                    cat_list = []
                    for i in range(1, len(plist)-1):
                        triple = '~'.join([plist[i+j] for j in [-1,0,1]])
                        cat_list.append(triple)
                    base_input += cat_list
                else:
                    base_input += plist[1:-1]

            self.input = base_input
            self.target = base_target

            print(f"✅ Inputs: {len(self.input)}, Targets: {len(self.target)}")

    def parse_metadata_from_filename(self, path):
        fname = osp.basename(path)

        dose_match = re.search(r'dose(\d+)', fname)
        dose = int(dose_match.group(1)) if dose_match else 25

        patient_match = re.search(r'L(\d+)', fname)
        pid = int(patient_match.group(1)) if patient_match else 0

        return {
            'dose': dose,
            'site': self.dataset_name,
            'slice_thickness': 1.0,
            'kernel': 'IR70',
            'patient_id': pid
        }

    def __getitem__(self, index):
        input_path, target_path = self.input[index], self.target[index]

        # === 加载图像 ===
        if self.context:
            paths = input_path.split('~')
            imgs = [np.load(p)[np.newaxis, ...].astype(np.float32) for p in paths]
            input_img = np.concatenate(imgs, axis=0)  # (3,H,W)
            main_path = paths[1]
        else:
            input_img = np.load(input_path)[np.newaxis, ...].astype(np.float32)
            main_path = input_path

        target_img = np.load(target_path)[np.newaxis, ...].astype(np.float32)

        # normalize
        input_img = self.normalize_(input_img)
        target_img = self.normalize_(target_img)

        # === 文本条件 ===
        if self.use_text:
            meta = self.parse_metadata_from_filename(main_path)
            desc = self.text_generator.generate_description(
                dose=meta['dose'],
                site=meta['site'],
                slice_thickness=meta['slice_thickness'],
                kernel=meta['kernel']
            )

            text_emb = self.text_encoder.encode(desc)  # Tensor(256)

            # === 关键修复：全部转成 numpy ===
            return {
                'input': input_img.astype(np.float32),
                'target': target_img.astype(np.float32),
                'text_embedding': text_emb.detach().cpu().numpy().astype(np.float32),
                'description': desc,
                'dose': meta['dose'],
                'patient_id': meta['patient_id'],
            }

        # 如果不用 text condition
        return input_img, target_img

    def __len__(self):
        return len(self.target)

    def normalize_(self, img, MIN_B=-1024, MAX_B=3072):
        img = img - 1024
        img = np.clip(img, MIN_B, MAX_B)
        return (img - MIN_B) / (MAX_B - MIN_B)


# === dict wrapper ===
def create_text_conditioned_dataset(dataset='mayo_2016', mode='train',
                                   test_id=9, dose=25, context=True, use_text=True):
    return TextConditionedCTDataset(
        dataset=dataset,
        mode=mode,
        test_id=test_id,
        dose=dose,
        context=context,
        use_text=use_text,
    )


text_dataset_dict = {
    'train': partial(create_text_conditioned_dataset,
                     dataset='mayo_2016', mode='train',
                     test_id=9, dose=[25, 50],
                     context=True, use_text=True),

    'test_25': partial(create_text_conditioned_dataset,
                       dataset='mayo_2016', mode='test',
                       test_id=9, dose=25,
                       context=True, use_text=True),

    'test_50': partial(create_text_conditioned_dataset,
                       dataset='mayo_2016', mode='test',
                       test_id=9, dose=50,
                       context=True, use_text=True),
}
