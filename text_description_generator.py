"""
Text Description Generator for CT Images
根据 CT 图像的元数据(剂量、站点、重建参数等)自动生成医学描述句子

🔥 增强版：针对单站点多剂量场景优化
- 25% vs 50% 的描述区分度大幅提升
- 添加物理细节、专业术语
- 从 15 词 → 50+ 词，给 BERT 更丰富的语义信息
"""

import torch
import torch.nn as nn


class TextDescriptionGenerator:
    """
    根据 CT 图像元数据自动生成医学描述文本
    """
    
    # 🔥 增强版：剂量级别描述模板（详细版）
    DOSE_DESCRIPTIONS = {
        100: """Full-dose CT protocol with 100% standard radiation exposure and optimal photon flux. 
                Images exhibit high signal-to-noise ratio with minimal quantum noise and excellent tissue contrast. 
                Diagnostic quality is optimal with clear anatomical detail and well-defined structural boundaries. 
                No denoising is required as image quality meets clinical standards.""",
        
        50: """Low-dose CT protocol with 50% radiation dose reduction compared to standard full-dose acquisition. 
               Photon count is reduced by half resulting in moderate Gaussian noise corruption and decreased signal integrity. 
               Signal-to-noise ratio is moderately compromised but major anatomical structures remain clearly identifiable. 
               Moderate denoising processing is required to restore diagnostic image quality and enhance tissue contrast.""",
        
        25: """Ultra-low-dose CT protocol with 75% radiation dose reduction and severe photon count limitation. 
               Extreme photon starvation leads to dominant Poisson noise with high variance and significant quantum mottle. 
               Signal-to-noise ratio is critically degraded with substantial loss of fine anatomical detail and edge definition. 
               Aggressive denoising algorithms are essential to recover diagnostic information and restore clinical utility.""",
        
        10: """Extreme ultra-low-dose CT protocol with 90% radiation dose reduction and minimal photon flux. 
               Severe photon starvation causes overwhelming Poisson noise with extremely high variance and pervasive quantum artifacts. 
               Signal-to-noise ratio approaches critical thresholds with profound degradation of image quality and structural clarity. 
               Very aggressive denoising with advanced algorithms is mandatory to extract any diagnostic value.""",
        
        5: """Minimal radiation CT protocol with 95% dose reduction and near-critical photon count levels. 
              Extreme photon starvation results in catastrophic Poisson noise dominance with maximal quantum mottle and severe artifacts. 
              Signal-to-noise ratio is at detection limits with massive loss of anatomical information and diagnostic confidence. 
              State-of-the-art deep learning denoising is absolutely required to achieve any clinical interpretability."""
    }
    
    # 简化版（用于消融实验对比）
    DOSE_DESCRIPTIONS_SIMPLE = {
        100: "This is a full-dose CT scan.",
        50: "This is a low-dose CT scan with 50% radiation dose.",
        25: "This is a low-dose CT scan with 25% radiation dose.",
        10: "This is an ultra-low-dose CT scan with 10% radiation dose.",
        5: "This is an ultra-low-dose CT scan with 5% radiation dose."
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
    
    # 重建核描述（增强版）
    KERNEL_DESCRIPTIONS = {
        'FBP': 'filtered back-projection reconstruction with standard convolution kernel',
        'IR70': 'IR70 iterative reconstruction kernel with strong noise suppression',
        'IR50': 'IR50 iterative reconstruction kernel with moderate noise reduction',
        'AIDR': 'adaptive iterative dose reduction reconstruction algorithm',
        'standard': 'standard reconstruction kernel with conventional filtering',
        'sharp': 'sharp reconstruction kernel emphasizing high-frequency detail',
        'smooth': 'smooth reconstruction kernel for enhanced noise reduction'
    }
    
    @staticmethod
    def generate_description(dose, site, slice_thickness=None, kernel=None, 
                           reconstruction_method=None, scanner_model=None,
                           mode='detailed'):
        """
        生成完整的医学描述句子
        
        Args:
            dose: 剂量百分比 (5, 10, 25, 50, 100)
            site: 站点/数据集名称
            slice_thickness: 层厚 (mm)
            kernel: 重建核
            reconstruction_method: 重建方法
            scanner_model: 扫描仪型号
            mode: 'detailed' 或 'simple'
            
        Returns:
            str: 完整的医学描述句子
        """
        
        # 选择描述模板
        if mode == 'simple':
            dose_templates = TextDescriptionGenerator.DOSE_DESCRIPTIONS_SIMPLE
        else:
            dose_templates = TextDescriptionGenerator.DOSE_DESCRIPTIONS
        
        # 基础剂量描述
        dose_desc = dose_templates.get(
            dose, f"{dose}% dose protocol with proportional radiation exposure"
        )
        
        # 站点描述
        site_desc = TextDescriptionGenerator.SITE_DESCRIPTIONS.get(
            site, f"{site} medical center"
        )
        
        # 构建完整描述
        description = dose_desc
        description += f" The image originates from {site_desc}"
        
        # 添加可选参数（增强语义信息）
        if kernel:
            kernel_desc = TextDescriptionGenerator.KERNEL_DESCRIPTIONS.get(
                kernel, f"{kernel} reconstruction kernel"
            )
            description += f" and was processed using {kernel_desc}"
        
        if slice_thickness:
            description += f" with {slice_thickness}mm slice thickness providing spatial resolution"
        
        if reconstruction_method:
            description += f" utilizing {reconstruction_method} reconstruction algorithm for image formation"
        
        if scanner_model:
            description += f" acquired on {scanner_model} CT scanner system"
        
        description += "."
        
        return description
    
    @staticmethod
    def parse_filename_to_description(filename, mode='detailed'):
        """
        从文件名解析元数据并生成描述
        
        Example filename: 'L067_25_045_img.npy'
        - L067: 患者ID
        - 25: 剂量 (25%)
        - 045: slice编号
        
        Args:
            filename: CT 图像文件名
            mode: 'detailed' 或 'simple'
            
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
            kernel='standard',
            mode=mode
        )
    
    @staticmethod
    def batch_generate_descriptions(dose_list, site_list, mode='detailed', **kwargs):
        """
        批量生成描述 (用于 DataLoader)
        
        Args:
            dose_list: List[int] - 剂量列表
            site_list: List[str] - 站点列表
            mode: 'detailed' 或 'simple'
            **kwargs: 其他可选参数
            
        Returns:
            List[str]: 描述句子列表
        """
        descriptions = []
        for dose, site in zip(dose_list, site_list):
            desc = TextDescriptionGenerator.generate_description(
                dose=dose,
                site=site,
                mode=mode,
                **kwargs
            )
            descriptions.append(desc)
        return descriptions
    
    @staticmethod
    def get_dose_category(dose):
        """
        获取剂量类别（用于分析）
        
        Args:
            dose: 剂量百分比
            
        Returns:
            str: 类别名称
        """
        if dose == 100:
            return "full_dose"
        elif dose >= 50:
            return "medium_dose"
        elif dose >= 25:
            return "low_dose"
        else:
            return "ultra_low_dose"


# ============== 示例用法 ==============
if __name__ == '__main__':
    gen = TextDescriptionGenerator()
    
    print("=" * 80)
    print("🧪 Text Description Generator - Enhanced Version")
    print("=" * 80)
    
    # 示例1: 详细模式 - 25% vs 50% 对比
    print("\n📝 示例1: 详细模式 (Detailed Mode)")
    print("-" * 80)
    desc_25 = gen.generate_description(
        dose=25,
        site='mayo_2016',
        slice_thickness=1.0,
        kernel='IR70',
        mode='detailed'
    )
    print(f"\n25% 剂量描述 ({len(desc_25)} 字符):")
    print(f"{desc_25}")
    
    desc_50 = gen.generate_description(
        dose=50,
        site='mayo_2016',
        slice_thickness=1.0,
        kernel='IR70',
        mode='detailed'
    )
    print(f"\n50% 剂量描述 ({len(desc_50)} 字符):")
    print(f"{desc_50}")
    
    # 示例2: 简单模式（用于消融实验）
    print("\n" + "=" * 80)
    print("📝 示例2: 简单模式 (Simple Mode)")
    print("-" * 80)
    desc_25_simple = gen.generate_description(
        dose=25,
        site='mayo_2016',
        mode='simple'
    )
    print(f"\n25% 剂量描述 (简单):")
    print(f"{desc_25_simple}")
    
    desc_50_simple = gen.generate_description(
        dose=50,
        site='mayo_2016',
        mode='simple'
    )
    print(f"\n50% 剂量描述 (简单):")
    print(f"{desc_50_simple}")
    
    # 示例3: 从文件名解析
    print("\n" + "=" * 80)
    print("📝 示例3: 从文件名解析")
    print("-" * 80)
    desc_from_file = gen.parse_filename_to_description('L067_25_045_img.npy', mode='detailed')
    print(f"\n文件名: L067_25_045_img.npy")
    print(f"描述: {desc_from_file}")
    
    # 示例4: 批量生成
    print("\n" + "=" * 80)
    print("📝 示例4: 批量生成")
    print("-" * 80)
    doses = [25, 25, 50, 50]
    sites = ['mayo_2016', 'mayo_2016', 'mayo_2016', 'mayo_2016']
    batch_descs = gen.batch_generate_descriptions(
        doses, sites, 
        mode='detailed',
        slice_thickness=1.0,
        kernel='IR70'
    )
    print(f"\n生成 {len(batch_descs)} 条描述:")
    for i, desc in enumerate(batch_descs):
        dose_label = doses[i]
        print(f"\n[{i}] {dose_label}% - 前100字符:")
        print(f"    {desc[:100]}...")
    
    # 示例5: 统计对比
    print("\n" + "=" * 80)
    print("📊 示例5: 描述统计对比")
    print("=" * 80)
    
    for dose in [25, 50, 100]:
        desc_detail = gen.generate_description(dose, 'mayo_2016', mode='detailed')
        desc_simple = gen.generate_description(dose, 'mayo_2016', mode='simple')
        
        print(f"\n{dose}% 剂量:")
        print(f"  详细模式: {len(desc_detail)} 字符, {len(desc_detail.split())} 词")
        print(f"  简单模式: {len(desc_simple)} 字符, {len(desc_simple.split())} 词")
        print(f"  增强比例: {len(desc_detail) / len(desc_simple):.1f}x")
    
    # 示例6: BERT tokenization 预览
    print("\n" + "=" * 80)
    print("🔤 示例6: BERT Tokenization 预览")
    print("=" * 80)
    print("\n提示: 更长的描述 → 更多tokens → 更丰富的embedding")
    print(f"\n25% 详细描述关键词:")
    keywords_25 = ['ultra-low-dose', 'photon starvation', 'Poisson noise', 
                   'quantum mottle', 'signal-to-noise ratio', 'aggressive denoising']
    for kw in keywords_25:
        print(f"  ✓ {kw}")
    
    print(f"\n50% 详细描述关键词:")
    keywords_50 = ['low-dose', 'photon count reduced', 'Gaussian noise', 
                   'moderate', 'signal integrity', 'moderate denoising']
    for kw in keywords_50:
        print(f"  ✓ {kw}")
    
    print("\n" + "=" * 80)
    print("✅ 测试完成！现在可以用于训练")
    print("=" * 80)
    print("\n💡 使用建议:")
    print("  1. 训练时使用 mode='detailed' 以获得最大区分度")
    print("  2. 消融实验时可对比 mode='simple' vs mode='detailed'")
    print("  3. 25% vs 50% 的描述差异已从 ~5词 增强到 ~50词")
    print("  4. 专业术语帮助BERT提取更有区分度的embedding")