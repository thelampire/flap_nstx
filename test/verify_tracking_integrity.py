#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 11:58:05 2026

@author: mlampert
"""
import numpy as np
# Ensure you import your classes here, e.g.:
# from structure_object import StructureDataset '

from flap_nstx.tools import StructureDataset

import numpy as np
# from flap_nstx.tools.structure_object import StructureDataset # Adjust import as needed

def verify_tracking_integrity(untracked_filepath, tracked_filepath=None, min_structure_lifetime=10):
    """
    Verifies that structures inside a tracked HDF5 file perfectly match 
    their original untracked counterparts. 
    
    Supports both legacy Dual Files and new Unified Files.
    """
    # Auto-detect Unified vs Dual mode
    if tracked_filepath is None or tracked_filepath == untracked_filepath:
        print(f"--- RUNNING IN UNIFIED FILE MODE ---")
        print(f"Target File: {untracked_filepath}")
        tracked_filepath = untracked_filepath
    else:
        print(f"--- RUNNING IN DUAL FILE MODE ---")
        print(f"Untracked File: {untracked_filepath}")
        print(f"Tracked File:   {tracked_filepath}")

    # Use the Matchmaker arguments to force the correct load behaviors
    print("\nLoading untracked dataset...")
    untracked_ds = StructureDataset.load_hdf5(untracked_filepath, tracked=False)
    
    print("Loading tracked dataset (and dynamically relinking footprints)...")
    tracked_ds = StructureDataset.load_hdf5(tracked_filepath, tracked=True)
    
    print("-" * 50)
    
    # 1. Build a fast-lookup dictionary for the untracked frames using their timestamps
    untracked_time_map = {
        np.round(time, 6): untracked_ds.frames[i] 
        for i, time in enumerate(untracked_ds.frame_times)
    }

    total_tracked_blobs = len(tracked_ds.tracked_structures)
    total_individual_structures = 0
    mismatches = 0
    orphans_found = 0
    relink_failures = 0
    
    print(f"Verifying {total_tracked_blobs} tracked blobs...")

    for blob in tracked_ds.tracked_structures:
        
        # 2. Verify the orphan filter worked
        lifetime = len(blob.time)
        if lifetime < min_structure_lifetime:
            print(f"[!] Orphan Leak: Blob {blob.label} only lived for {lifetime} frames!")
            orphans_found += 1
            
        # 3. Verify every individual footprint inside the blob
        for step_idx, step_time in enumerate(blob.time):
            total_individual_structures += 1
            
            safe_time = np.round(step_time, 6)
            if safe_time not in untracked_time_map:
                print(f"[!] Frame missing: No untracked frame found for time {step_time}")
                mismatches += 1
                continue
                
            untracked_candidates = untracked_time_map[safe_time]
            
            # 4. Verify that the Matchmaker Loader successfully populated the structures list
            if not hasattr(blob, 'structures') or len(blob.structures) <= step_idx:
                print(f"[!] Relink Failure: Blob {blob.label} at time {step_time} has no physical footprint!")
                relink_failures += 1
                continue
                
            t_struct = blob.structures[step_idx]
            
            # Search the untracked frame for the exact spatial pixel arrays
            match_found = False
            for u_struct in untracked_candidates:
                if np.array_equal(t_struct.x_data, u_struct.x_data) and \
                   np.array_equal(t_struct.y_data, u_struct.y_data):
                    match_found = True
                    break
            
            if not match_found:
                print(f"[!] Phantom Structure: Blob {blob.label} at time {step_time} does not match any untracked footprint!")
                mismatches += 1

    print("-" * 50)
    print("VERIFICATION REPORT:")
    print(f"Total Tracked Blobs Verified:        {total_tracked_blobs}")
    print(f"Total Individual Frames Verified:    {total_individual_structures}")
    print(f"Lifetime Filter Violations (Orphans): {orphans_found}")
    print(f"Relink Failures (RAM mapping failed): {relink_failures}")
    print(f"Data Mismatches / Phantoms:          {mismatches}")
    
    if mismatches == 0 and orphans_found == 0 and relink_failures == 0:
        print("\nSUCCESS: All tracked structures are perfectly mapped to their untracked origins!")
        return True
    else:
        print("\nWARNING: Discrepancies detected between the datasets.")
        return False

# --- Example Usage ---
if __name__ == "__main__":
    # To test a Unified File:
    # verify_tracking_integrity("my_unified_file.h5", min_structure_lifetime=10)
    
    # To test legacy separate files:
    # verify_tracking_integrity("untracked_data.h5", "tracked_data.h5", min_structure_lifetime=10)
    pass