#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 14:38:22 2026

@author: mlampert
"""
import h5py
import numpy as np

def explore_hdf5(filepath, max_items=10):
    """
    Recursively prints the internal tree structure, datasets, and attributes 
    of an HDF5 file.
    
    Args:
        filepath (str): Path to the .h5 file.
        max_items (int): Maximum number of items to print in a single group 
                         to prevent console flooding (e.g., if you have 5000 frames).
    """
    print(f"\n{'='*60}")
    print(f" Inspecting HDF5 File: {filepath}")
    print(f"{'='*60}\n")

    def _print_attributes(item, indent):
        """Helper to print metadata attributes attached to groups/datasets."""
        if item.attrs:
            for key, val in item.attrs.items():
                print(f"{indent}  |-- (Attr) {key}: {val}")

    def _traverse_node(name, node, level=0):
        indent = "    " * level
        
        if isinstance(node, h5py.Dataset):
            # It's a data array
            dtype = node.dtype
            shape = node.shape
            print(f"{indent}[Dataset] {name} (Shape: {shape}, Type: {dtype})")
            _print_attributes(node, indent)
            
        elif isinstance(node, h5py.Group):
            # It's a folder/group
            print(f"{indent}[Group] {name}/")
            _print_attributes(node, indent)
            
            # Sort keys numerically if they look like 'frame_1', 'frame_2', etc.
            keys = list(node.keys())
            try:
                keys.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
            except Exception:
                keys.sort()
            
            # Print children, but cap it at max_items to avoid unreadable console spam
            for i, key in enumerate(keys):
                if i >= max_items:
                    print(f"{indent}    ... and {len(keys) - max_items} more items hidden ...")
                    break
                _traverse_node(key, node[key], level + 1)

    try:
        with h5py.File(filepath, 'r') as f:
            # Print root attributes first (like mode, exp_id)
            print("[Root] /")
            _print_attributes(f, "")
            print("-" * 40)
            
            # Start traversing from the base
            for key in f.keys():
                _traverse_node(key, f[key], level=1)
                
    except OSError as e:
        print(f"Error opening file: {e}")
        print("Are you sure the file path is correct and the file is not corrupted?")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    print(f"\n{'='*60}\n")


# --- Example Usage ---
if __name__ == "__main__":
    # Replace with the path to your newly saved tracked or untracked file
    # explore_hdf5("processed_data/your_tracked_dataset.h5")
    pass