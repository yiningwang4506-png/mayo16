import os
import os.path as osp
import argparse
import numpy as np
from natsort import natsorted
from glob import glob
import pydicom

def save_dataset(args):
    if not osp.exists(args.save_path):
        os.makedirs(args.save_path)
    
    patient_ids = ['L067', 'L096', 'L109', 'L143', 'L192', 'L286', 'L291', 'L310', 'L333', 'L506']
    
    for patient_id in patient_ids:
        print(f"Processing {patient_id} - target")
        patient_path = osp.join(args.data_path, 'full_1mm', patient_id, 'full_1mm')
        data_paths = natsorted(glob(osp.join(patient_path, '*.IMA')))
        print(f"  Found {len(data_paths)} files")
        for slice_idx, data_path in enumerate(data_paths):
            im = pydicom.dcmread(data_path)
            f = np.array(im.pixel_array)
            np.save(osp.join(args.save_path, f'{patient_id}_target_{slice_idx:03d}_img.npy'), f.astype(np.uint16))
    
    for patient_id in patient_ids:
        print(f"Processing {patient_id} - 25")
        patient_path = osp.join(args.data_path, 'quarter_1mm', patient_id, 'quarter_1mm')
        data_paths = natsorted(glob(osp.join(patient_path, '*.IMA')))
        print(f"  Found {len(data_paths)} files")
        for slice_idx, data_path in enumerate(data_paths):
            im = pydicom.dcmread(data_path)
            f = np.array(im.pixel_array)
            np.save(osp.join(args.save_path, f'{patient_id}_25_{slice_idx:03d}_img.npy'), f.astype(np.uint16))
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()
    save_dataset(args)
