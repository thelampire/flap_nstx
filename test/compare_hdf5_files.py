#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 10:59:43 2026

@author: mlampert
"""
import h5py
import numpy as np

def compare_attributes(attrs1, attrs2, path):
    """Compares the attributes of two HDF5 objects, safely handling NaNs."""
    keys1 = set(attrs1.keys())
    keys2 = set(attrs2.keys())
    
    if keys1 != keys2:
        print(f"[!] Attribute key mismatch at {path}:")
        print(f"    Only in file 1: {keys1 - keys2}")
        print(f"    Only in file 2: {keys2 - keys1}")
        return False
        
    all_match = True
    for key in keys1:
        val1 = attrs1[key]
        val2 = attrs2[key]
        
        # Handle numpy arrays in attributes
        if isinstance(val1, np.ndarray) or isinstance(val2, np.ndarray):
            # np.array_equal doesn't handle NaNs, so we use allclose with equal_nan=True
            if not np.allclose(val1, val2, equal_nan=True):
                print(f"[!] Attribute array mismatch at {path} (Key: {key})")
                all_match = False
        else:
            # Check for NaN equality since np.nan != np.nan
            is_nan_match = False
            if isinstance(val1, (float, np.floating)) and isinstance(val2, (float, np.floating)):
                if np.isnan(val1) and np.isnan(val2):
                    is_nan_match = True
            
            if not is_nan_match and val1 != val2:
                print(f"[!] Attribute mismatch at {path} (Key: {key}): {val1} vs {val2}")
                all_match = False
                
    return all_match

def compare_hdf5_nodes(node1, node2, path="/"):
    """Recursively compares two HDF5 nodes (Groups or Datasets)."""
    all_match = True
    
    # 1. Compare Attributes
    if not compare_attributes(node1.attrs, node2.attrs, path):
        all_match = False

    # 2. Compare Group structures
    if isinstance(node1, h5py.Group) and isinstance(node2, h5py.Group):
        keys1 = set(node1.keys())
        keys2 = set(node2.keys())
        
        if keys1 != keys2:
            print(f"[!] Structure mismatch at Group {path}:")
            print(f"    Only in file 1: {keys1 - keys2}")
            print(f"    Only in file 2: {keys2 - keys1}")
            all_match = False
            
        # Recurse into common keys
        common_keys = keys1.intersection(keys2)
        for key in common_keys:
            new_path = f"{path}{key}/" if path == "/" else f"{path}/{key}"
            
            # Check if types match before recursing
            if type(node1[key]) != type(node2[key]):
                print(f"[!] Type mismatch at {new_path}: {type(node1[key])} vs {type(node2[key])}")
                all_match = False
                continue
                
            if not compare_hdf5_nodes(node1[key], node2[key], new_path):
                all_match = False

    # 3. Compare Datasets
    elif isinstance(node1, h5py.Dataset) and isinstance(node2, h5py.Dataset):
        # Check shapes
        if node1.shape != node2.shape:
            print(f"[!] Shape mismatch at Dataset {path}: {node1.shape} vs {node2.shape}")
            return False
            
        # Check data types
        if node1.dtype != node2.dtype:
            print(f"[!] Dtype mismatch at Dataset {path}: {node1.dtype} vs {node2.dtype}")
            return False
            
        # Check actual data
        try:
            data1 = node1[...]
            data2 = node2[...]
            
            # Use allclose for floats to handle minor precision differences and NaNs safely
            if np.issubdtype(node1.dtype, np.number):
                if not np.allclose(data1, data2, equal_nan=True):
                    print(f"[!] Data values mismatch at Dataset {path}")
                    all_match = False
            else:
                # For strings, booleans, or objects
                if not np.array_equal(data1, data2):
                    print(f"[!] Data values mismatch at Dataset {path}")
                    all_match = False
                    
        except Exception as e:
            print(f"[!] Could not load/compare data at {path}. Error: {e}")
            all_match = False
            
    return all_match

def compare_hdf5_files(file1_path, file2_path):
    """Main wrapper to open files and initiate comparison."""
    print(f"Comparing:\n  File 1: {file1_path}\n  File 2: {file2_path}\n")
    
    try:
        with h5py.File(file1_path, 'r') as f1, h5py.File(file2_path, 'r') as f2:
            is_identical = compare_hdf5_nodes(f1, f2)
            
            print("-" * 40)
            if is_identical:
                print("Result: Files are completely identical.")
            else:
                print("Result: Differences were found (see log above).")
                
    except OSError as e:
        print(f"Error opening files: {e}")

# --- Example Usage ---
if __name__ == "__main__":
    # Replace these with your actual file paths
    # compare_hdf5_files("dataset_v1.h5", "dataset_v2.h5")
    pass