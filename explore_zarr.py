#!/usr/bin/env python3

import sys
sys.path.insert(0, '/home/kkingstoun/git/mmpp')
import zarr

zarr_path = "/mnt/storage_2/scratch/pl0095-01/zelent/gleb/simulations/field_dependent/spectrum_withoutanisu.zarr"

print("Exploring zarr structure...")

z = zarr.open(zarr_path, mode='r')

def explore_group(group, path="", max_depth=3, current_depth=0):
    if current_depth >= max_depth:
        return
        
    for key in group.keys():
        full_path = f"{path}/{key}" if path else key
        item = group[key]
        
        if hasattr(item, 'keys'):  # It's a group
            print(f"📁 {full_path}/")
            explore_group(item, full_path, max_depth, current_depth + 1)
        else:  # It's an array
            print(f"📄 {full_path} - shape: {item.shape}, dtype: {item.dtype}")

explore_group(z)

print("\n=== FFT Structure ===")
if 'fft' in z:
    fft_group = z['fft']
    for fft_key in fft_group.keys():
        print(f"FFT dataset: {fft_key}")
        fft_dataset = fft_group[fft_key]
        if hasattr(fft_dataset, 'keys'):
            for sub_key in fft_dataset.keys():
                sub_item = fft_dataset[sub_key]
                print(f"  {sub_key} - shape: {sub_item.shape}, dtype: {sub_item.dtype}")

print("\n=== MODES Structure ===")
if 'modes' in z:
    modes_group = z['modes']
    for modes_key in modes_group.keys():
        print(f"MODES dataset: {modes_key}")
        modes_dataset = modes_group[modes_key]
        if hasattr(modes_dataset, 'keys'):
            for sub_key in modes_dataset.keys():
                sub_item = modes_dataset[sub_key]
                print(f"  {sub_key} - shape: {sub_item.shape}, dtype: {sub_item.dtype}")