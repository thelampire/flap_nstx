import h5py
import numpy as np

def compare_hdf5_roundtrip(file1_path, file2_path):
    """
    Recursively compares two HDF5 files node-by-node and reports ALL differences.
    """
    print(f"Comparing:\n1. {file1_path}\n2. {file2_path}\n{'-'*50}")

    def compare_attributes(name, attrs1, attrs2):
        keys1, keys2 = set(attrs1.keys()), set(attrs2.keys())
        match = True
        
        if keys1 != keys2:
            print(f"[!] Attribute key mismatch at {name}")
            print(f"    File 1 only: {keys1 - keys2}")
            print(f"    File 2 only: {keys2 - keys1}")
            match = False
            
        # Only compare values for keys that exist in both
        for k in keys1.intersection(keys2):
            v1, v2 = attrs1[k], attrs2[k]
            
            # Handle float and NaN equality
            if isinstance(v1, (float, np.floating)) or isinstance(v2, (float, np.floating)):
                if not (np.isnan(v1) and np.isnan(v2)) and not np.isclose(v1, v2, equal_nan=True):
                    print(f"[!] Attribute mismatch at {name} [{k}]: {v1} != {v2}")
                    match = False
            # Handle NumPy arrays in attributes
            elif isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
                if not np.array_equal(v1, v2):
                    print(f"[!] Attribute array mismatch at {name} [{k}]")
                    match = False
            # Handle strings vs bytes
            else:
                str_v1 = v1.decode('utf-8') if isinstance(v1, bytes) else str(v1)
                str_v2 = v2.decode('utf-8') if isinstance(v2, bytes) else str(v2)
                if str_v1 != str_v2:
                    print(f"[!] Attribute mismatch at {name} [{k}]: '{str_v1}' != '{str_v2}'")
                    match = False
                    
        return match

    def traverse_and_compare(name, node1, node2):
        match = True
        
        # 1. Check Node Type
        if type(node1) != type(node2):
            print(f"[!] Structure type mismatch at {name}: {type(node1)} vs {type(node2)}")
            return False # Must return immediately here, cannot dive into mismatched types

        # 2. Compare Attributes
        if not compare_attributes(name, node1.attrs, node2.attrs):
            match = False

        # 3. Compare Groups (Folders)
        if isinstance(node1, h5py.Group):
            keys1, keys2 = set(node1.keys()), set(node2.keys())
            if keys1 != keys2:
                print(f"[!] Group content mismatch at {name}")
                print(f"    File 1 only: {keys1 - keys2}")
                print(f"    File 2 only: {keys2 - keys1}")
                match = False
                
            # Traverse all shared children
            for key in keys1.intersection(keys2):
                if not traverse_and_compare(f"{name}/{key}", node1[key], node2[key]):
                    match = False

        # 4. Compare Datasets (Arrays)
        elif isinstance(node1, h5py.Dataset):
            if node1.shape != node2.shape:
                print(f"[!] Dataset shape mismatch at {name}: {node1.shape} vs {node2.shape}")
                match = False
            else:
                d1, d2 = node1[()], node2[()]
                
                # Handle numerical floats and NaNs
                if d1.dtype.kind in ['f', 'c']:
                    if not np.allclose(d1, d2, equal_nan=True):
                        print(f"[!] Dataset float value mismatch at {name}")
                        match = False
                # Handle strings or object arrays safely
                elif d1.dtype.kind in ['S', 'U', 'O']:
                    if not np.array_equal(d1.astype(str), d2.astype(str)):
                        print(f"[!] Dataset string/object mismatch at {name}")
                        match = False
                # Handle integers and booleans
                else:
                    if not np.array_equal(d1, d2):
                        print(f"[!] Dataset value mismatch at {name}")
                        match = False
                        
        return match

    try:
        with h5py.File(file1_path, 'r') as f1, h5py.File(file2_path, 'r') as f2:
            is_identical = traverse_and_compare("/", f1, f2)
            
            print("-" * 50)
            if is_identical:
                print("VERDICT: SUCCESS! The files contain mathematically identical data.")
            else:
                print("VERDICT: FAILED. Differences were found (see log above).")
            return is_identical
            
    except Exception as e:
        print(f"Comparison crashed: {e}")
        return False
    
import h5py
import numpy as np

def compare_hdf5_roundtrip_exhaustive(file1_path, file2_path):
    """
    Recursively compares two HDF5 files node-by-node and reports ALL differences.
    """
    print(f"Comparing:\n1. {file1_path}\n2. {file2_path}\n{'-'*50}")

    def compare_attributes(name, attrs1, attrs2):
        keys1, keys2 = set(attrs1.keys()), set(attrs2.keys())
        match = True
        
        if keys1 != keys2:
            print(f"[!] Attribute key mismatch at {name}")
            print(f"    File 1 only: {keys1 - keys2}")
            print(f"    File 2 only: {keys2 - keys1}")
            match = False
            
        # Only compare values for keys that exist in both
        for k in keys1.intersection(keys2):
            v1, v2 = attrs1[k], attrs2[k]
            
            # Handle float and NaN equality
            if isinstance(v1, (float, np.floating)) or isinstance(v2, (float, np.floating)):
                if not (np.isnan(v1) and np.isnan(v2)) and not np.isclose(v1, v2, equal_nan=True):
                    print(f"[!] Attribute mismatch at {name} [{k}]: {v1} != {v2}")
                    match = False
            # Handle NumPy arrays in attributes
            elif isinstance(v1, np.ndarray) or isinstance(v2, np.ndarray):
                if not np.array_equal(v1, v2):
                    print(f"[!] Attribute array mismatch at {name} [{k}]")
                    match = False
            # Handle strings vs bytes
            else:
                str_v1 = v1.decode('utf-8') if isinstance(v1, bytes) else str(v1)
                str_v2 = v2.decode('utf-8') if isinstance(v2, bytes) else str(v2)
                if str_v1 != str_v2:
                    print(f"[!] Attribute mismatch at {name} [{k}]: '{str_v1}' != '{str_v2}'")
                    match = False
                    
        return match

    def traverse_and_compare(name, node1, node2):
        match = True
        
        # 1. Check Node Type
        if type(node1) != type(node2):
            print(f"[!] Structure type mismatch at {name}: {type(node1)} vs {type(node2)}")
            return False # Must return immediately here, cannot dive into mismatched types

        # 2. Compare Attributes
        if not compare_attributes(name, node1.attrs, node2.attrs):
            match = False

        # 3. Compare Groups (Folders)
        if isinstance(node1, h5py.Group):
            keys1, keys2 = set(node1.keys()), set(node2.keys())
            if keys1 != keys2:
                print(f"[!] Group content mismatch at {name}")
                print(f"    File 1 only: {keys1 - keys2}")
                print(f"    File 2 only: {keys2 - keys1}")
                match = False
                
            # Traverse all shared children
            for key in keys1.intersection(keys2):
                if not traverse_and_compare(f"{name}/{key}", node1[key], node2[key]):
                    match = False

        # 4. Compare Datasets (Arrays)
        elif isinstance(node1, h5py.Dataset):
            if node1.shape != node2.shape:
                print(f"[!] Dataset shape mismatch at {name}: {node1.shape} vs {node2.shape}")
                match = False
            else:
                d1, d2 = node1[()], node2[()]
                
                # Handle numerical floats and NaNs
                if d1.dtype.kind in ['f', 'c']:
                    if not np.allclose(d1, d2, equal_nan=True):
                        print(f"[!] Dataset float value mismatch at {name}")
                        match = False
                # Handle strings or object arrays safely
                elif d1.dtype.kind in ['S', 'U', 'O']:
                    if not np.array_equal(d1.astype(str), d2.astype(str)):
                        print(f"[!] Dataset string/object mismatch at {name}")
                        match = False
                # Handle integers and booleans
                else:
                    if not np.array_equal(d1, d2):
                        print(f"[!] Dataset value mismatch at {name}")
                        match = False
                        
        return match

    try:
        with h5py.File(file1_path, 'r') as f1, h5py.File(file2_path, 'r') as f2:
            is_identical = traverse_and_compare("/", f1, f2)
            
            print("-" * 50)
            if is_identical:
                print("VERDICT: SUCCESS! The files contain mathematically identical data.")
            else:
                print("VERDICT: FAILED. Differences were found (see log above).")
            return is_identical
            
    except Exception as e:
        print(f"Comparison crashed: {e}")
        return False
    