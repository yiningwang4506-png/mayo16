import os, re, os.path as osp, argparse, numpy as np
from glob import glob
from natsort import natsorted
import pydicom

def instance_no(path):
    try:
        d = pydicom.dcmread(path, stop_before_pixels=True)
        return int(getattr(d, "InstanceNumber", 10**9))
    except Exception:
        return 10**9

def find_date_dir(root, pid, prefix):
    """在 root/pid/** 下找第一个以 prefix 开头的目录（例如 '12-23-'），返回该目录路径。"""
    candidates = []
    root_pid = osp.join(root, pid)
    if not osp.isdir(root_pid):
        return None
    for dp, dn, _ in os.walk(root_pid):
        for d in dn:
            if d.startswith(prefix):
                candidates.append(osp.join(dp, d))
    if not candidates:
        return None
    candidates.sort(key=lambda p: len(p))  # 选路径最短的那层（最浅层）
    return candidates[0]

def export_series(date_dir, save_root, pid):
    # 只认含 full/low 的子目录（大小写不敏感）
    subdirs = [d for d in os.listdir(date_dir) if osp.isdir(osp.join(date_dir, d))]
    role_map = {}
    for d in subdirs:
        dl = d.lower()
        if 'full' in dl: role_map[d] = 'target'
        if 'low'  in dl: role_map[d] = '25'
    if not role_map:
        print(f"  WARN: {pid} 在 {date_dir} 下没找到 Full/Low 子目录，跳过")
        return 0

    total = 0
    for d, role in role_map.items():
        src = osp.join(date_dir, d)
        files = []
        for ext in ('*.dcm','*.DCM','*.IMA','*.ima'):
            files += glob(osp.join(src, ext))
        if not files:
            print(f"  WARN: {pid} {d} 无影像文件，跳过")
            continue
        files_sorted = sorted(files, key=instance_no)

        for k, f in enumerate(files_sorted, 1):
            try:
                ds = pydicom.dcmread(f)
                px = np.array(ds.pixel_array, dtype=np.float32)  # === 只保存原始 pixel_array(float32) ===
                name = f"{pid}_{role}_{k:04d}_img.npy"
                np.save(osp.join(save_root, name), px)
                total += 1
            except Exception as e:
                print(f"    skip {f}: {e}")
    return total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_path', required=True, help='包含 Lxxx 的 Mayo2020 根目录')
    ap.add_argument('--save_path',   default='./data_preprocess/gen_data/mayo_2020_npy', help='输出目录（与 dataset.py 对齐）')
    ap.add_argument('--date_prefix', default='12-23-', help='仅处理以该前缀命名的检查目录，例如 12-23-')
    args = ap.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    print("Save to:", osp.abspath(args.save_path))

    # 仅处理 Lxxx 病人
    pids = sorted([d for d in os.listdir(args.dataset_path) if re.fullmatch(r"L\d{3}", d)])
    if not pids:
        print("ERROR: 在 dataset_path 下未发现 Lxxx 病人目录")
        return

    grand = 0
    for pid in pids:
        print(f"[{pid}]")
        date_dir = find_date_dir(args.dataset_path, pid, args.date_prefix)
        if not date_dir:
            print(f"  WARN: 未找到以 {args.date_prefix} 开头的检查目录，跳过")
            continue
        print(f"  Use study: {date_dir}")
        cnt = export_series(date_dir, args.save_path, pid)
        print(f"  Exported: {cnt}")
        grand += cnt

    # 汇总
    from glob import glob
    all_npy = glob(osp.join(args.save_path, '*.npy'))
    tgt = glob(osp.join(args.save_path, '*target*_img.npy'))
    low = glob(osp.join(args.save_path, '*_25_*_img.npy'))
    print("\nDone!")
    print("Total :", len(all_npy))
    print("Target:", len(tgt))
    print("25%  :", len(low))

if __name__ == "__main__":
    main()
