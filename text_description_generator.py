"""
Text Description Generator for CT Images
根据 CT 图像的元数据(剂量、站点、重建参数等)自动生成医学描述句子
"""

import torch
import torch.nn as nn


class TextDescriptionGenerator:
    """
    根据 CT 图像元数据自动生成医学描述文本
    """
    
    # 剂量级别描述模板
    DOSE_DESCRIPTIONS = {
        100: "full-dose protocol with standard photon flux",
        50: "50% low-dose protocol which results in moderate noise due to reduced photon flux",
        25: "25% low-dose protocol which results in increased noise due to significantly reduced photon flux",
        10: "10% ultra-low-dose protocol which results in high noise due to severely reduced photon flux",
        5: "5% ultra-low-dose protocol which results in extremely high noise due to minimal photon flux"
    }
    
    # 站点/数据集描述
    SITE_DESCRIPTIONS = {
        'mayo_2016': 'Mayo Clinic 2016 dataset',
        'mayo_2016_sim': 'Mayo Clinic 2016 simulated dataset',
        'mayo_2020': 'Mayo Clinic 2020 dataset',
        'piglet': 'Piglet preclinical dataset',
        'phantom': 'Phantom calibration dataset',
        'zhuhai': 'Zhuhai People\'s Hospital',
        'wuhan': 'Wuhan Tongji Hospital',
        'shandong': 'Shandong First Medical University Hospital'
    }
    
    # 重建核描述
    KERNEL_DESCRIPTIONS = {
        'FBP': 'filtered back-projection reconstruction',
        'IR70': 'IR70 iterative reconstruction kernel',
        'IR50': 'IR50 iterative reconstruction kernel',
        'AIDR': 'adaptive iterative dose reduction reconstruction',
        'standard': 'standard reconstruction kernel'
    }
    
    @staticmethod
    def generate_description(dose, site, slice_thickness=None, kernel=None, 
                           reconstruction_method=None, scanner_model=None):
        """
        生成完整的医学描述句子
        
        Args:
            dose: 剂量百分比 (5, 10, 25, 50, 100)
            site: 站点/数据集名称
            slice_thickness: 层厚 (mm)
            kernel: 重建核
            reconstruction_method: 重建方法
            scanner_model: 扫描仪型号
            
        Returns:
            str: 完整的医学描述句子
        """
        
        # 基础剂量描述
        dose_desc = TextDescriptionGenerator.DOSE_DESCRIPTIONS.get(
            dose, f"{dose}% dose protocol"
        )
        
        # 站点描述
        site_desc = TextDescriptionGenerator.SITE_DESCRIPTIONS.get(
            site, f"{site} medical center"
        )
        
        # 构建完整描述
        description = f"This CT scan was acquired using a {dose_desc}. "
        description += f"The image comes from {site_desc}"
        
        # 添加可选参数
        if kernel:
            kernel_desc = TextDescriptionGenerator.KERNEL_DESCRIPTIONS.get(
                kernel, f"{kernel} reconstruction"
            )
            description += f" acquired using {kernel_desc}"
        
        if slice_thickness:
            description += f" with {slice_thickness}mm slice thickness"
        
        if reconstruction_method:
            description += f" using {reconstruction_method} reconstruction method"
        
        if scanner_model:
            description += f" on {scanner_model} scanner"
        
        description += "."
        
        return description
    
    @staticmethod
    def parse_filename_to_description(filename):
        """
        从文件名解析元数据并生成描述
        
        Example filename: 'L067_25_045_img.npy'
        - L067: 患者ID
        - 25: 剂量 (25%)
        - 045: slice编号
        
        Args:
            filename: CT 图像文件名
            
        Returns:
            str: 医学描述句子
        """
        parts = filename.split('_')
        
        # 默认值
        dose = 100
        site = 'unknown'
        
        # 解析剂量
        if len(parts) >= 2:
            try:
                dose_str = parts[1]
                if dose_str == 'target':
                    dose = 100
                else:
                    dose = int(dose_str)
            except:
                pass
        
        # 推断站点 (基于患者ID前缀)
        if parts[0].startswith('L'):
            site = 'mayo_2016'
        elif parts[0].startswith('C'):
            site = 'mayo_2020'
        elif 'piglet' in filename.lower():
            site = 'piglet'
        elif 'xnat' in filename.lower():
            site = 'phantom'
        
        return TextDescriptionGenerator.generate_description(
            dose=dose,
            site=site,
            slice_thickness=1.0,  # Mayo 数据集默认
            kernel='standard'
        )
    
    @staticmethod
    def batch_generate_descriptions(dose_list, site_list, **kwargs):
        """
        批量生成描述 (用于 DataLoader)
        
        Args:
            dose_list: List[int] - 剂量列表
            site_list: List[str] - 站点列表
            **kwargs: 其他可选参数
            
        Returns:
            List[str]: 描述句子列表
        """
        descriptions = []
        for dose, site in zip(dose_list, site_list):
            desc = TextDescriptionGenerator.generate_description(
                dose=dose,
                site=site,
                **kwargs
            )
            descriptions.append(desc)
        return descriptions


# ============== 示例用法 ==============
if __name__ == '__main__':
    gen = TextDescriptionGenerator()
    
    # 示例1: 手动指定参数
    desc1 = gen.generate_description(
        dose=25,
        site='mayo_2016',
        slice_thickness=1.0,
        kernel='IR70'
    )
    print("示例1:", desc1)
    # 输出: "This CT scan was acquired using a 25% low-dose protocol which 
    #        results in increased noise due to significantly reduced photon flux. 
    #        The image comes from Mayo Clinic 2016 dataset acquired using IR70 
    #        iterative reconstruction kernel with 1.0mm slice thickness."
    
    # 示例2: 从文件名解析
    desc2 = gen.parse_filename_to_description('L067_25_045_img.npy')
    print("\n示例2:", desc2)
    
    # 示例3: 批量生成
    doses = [25, 25, 10, 10]
    sites = ['mayo_2016', 'mayo_2016', 'mayo_2020', 'mayo_2020']
    batch_descs = gen.batch_generate_descriptions(doses, sites)
    print("\n示例3 (批量):")
    for i, desc in enumerate(batch_descs):
        print(f"  [{i}] {desc}")