#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 10:07:14 2026

@author: mlampert
"""
from flap_nstx.tools import MetricArray, FitShape, Polygon
import numpy as np
import h5py

class ParameterDict(dict):
    """
    A strict dictionary that allows dot-notation access to keys (translating 
    underscores to spaces). Now stores RAW FLOATS or MetricArrays depending 
    on the class that uses it.
    """
    def __getattr__(self, item):
        # 1. Try exact dictionary keys
        if item in self:
            return self[item]
            
        # 2. Try replacing underscores with spaces (e.g., .Size_radial_fit)
        spaced_item = item.replace('_', ' ')
        if spaced_item in self:
            return self[spaced_item]
            
        raise AttributeError(f"No dictionary key found for '{item}'")

    def __setattr__(self, key, value):
        if key.startswith('_'):
            self.__dict__[key] = value
        else:
            self[key.replace('_', ' ')] = value


class PlasmaStructure(Polygon, FitShape):
    
    # 1. Seamlessly merge both parent metadata dictionaries into one class-level master dict!
    METADATA = {**Polygon.METADATA, **FitShape.METADATA}
    
    # 2. Create a flat lookup registry keyed by 'dict_label' for downstream trackers to query
    FLAT_METADATA = {}
    for _, meta in METADATA.items():
        if isinstance(meta, dict):
            FLAT_METADATA[meta['dict_label']] = meta
        elif isinstance(meta, list):
            for sub_meta in meta:
                FLAT_METADATA[sub_meta['dict_label']] = sub_meta
    
    def __init__(self, x=None, y=None, x_data=None, y_data=None, x_data_pix=None, 
                 y_data_pix=None, data=None, fit_shape='ellipse', ellipse_method='linalg', 
                 path_order=3, test=False, distance_unit='m', elongation_base='size', 
                 verbose=False, smooth=0, **tracking_kwargs):
        
        Polygon.__init__(self, x=x, y=y, smooth=smooth, x_data=x_data, y_data=y_data, 
                         x_data_pix=x_data_pix, y_data_pix=y_data_pix, data=data, 
                         path_order=path_order, test=test, distance_unit=distance_unit)
        
        FitShape.__init__(self, fitting=fit_shape, x=self.x, y=self.y, 
                          x_data=self.x_data, y_data=self.y_data, data=self.data, 
                          method=ellipse_method, elongation_base=elongation_base, 
                          distance_unit=distance_unit, verbose=verbose, test=test)
        
        self.attribute_map={'Label': 'label', 'Born': 'born', 'Died': 'died', 
                            'Splits': 'splits', 'Merges': 'merges', 
                            'Parent': 'parents', 'Child': 'children'}
        
        # Tracking Metadata
        self.label = tracking_kwargs.get('label', None)
        self.parents = tracking_kwargs.get('parents', [])
        self.children = tracking_kwargs.get('children', [])
        self.born = tracking_kwargs.get('born', False)
        self.died = tracking_kwargs.get('died', False)
        self.splits = tracking_kwargs.get('splits', False)
        self.merges = tracking_kwargs.get('merges', False)

        # 3. These now store purely raw floats! Maximum execution speed.
        self.differential_parameters = ParameterDict()
        self.regular_parameters = ParameterDict()
        
        self.update_regular_parameters()

    def get_meta(self, dict_label):
        """Helper for downstream trackers to fetch LaTeX labels and units based on a key."""
        meta = self.FLAT_METADATA.get(dict_label, None)
        if meta is None:
            # Safe fallback if a custom key is injected
            return {'plot_label': dict_label, 'unit': ''}
            
        return {
            'plot_label': meta['plot_label'],
            'unit': '$'+meta['unit'].replace('DU', self._distance_unit)+'$' if 'DU' in meta['unit'] else meta['unit'],
        }

    def update_regular_parameters(self):
        """
        Dynamically executes methods defined in the METADATA registry and
        extracts their raw float values directly into the dictionary. NO METRICS!
        """
        self.regular_parameters.clear()
        
        for method_name, meta in self.METADATA.items():
            try:
                # 1. Fetch the raw math result natively by calling the method or property
                method_or_prop = getattr(self, method_name)
                
                if callable(method_or_prop):
                    raw_value = method_or_prop()
                else:
                    raw_value = method_or_prop
                
                # 2. Map the raw floats straight into the dictionary
                if isinstance(meta, dict):
                    # Single-value properties (e.g. area)
                    self.regular_parameters[meta['dict_label']] = raw_value
                elif isinstance(meta, list):
                    # Multi-value properties (e.g. centroid -> [R, Z])
                    for i, sub_meta in enumerate(meta):
                        self.regular_parameters[sub_meta['dict_label']] = raw_value[i]
            except Exception:
                pass # Fail silently for missing data (e.g. centroid if no intensity data)

    def smooth(self, refinements=1):
        """Overrides Polygon.smooth to ensure metrics update if boundaries change."""
        super().smooth(refinements=refinements)
        
        if self.fitting_type == 'ellipse':
            if hasattr(self, '_method'):
                if self._method == 'linalg': self._fit_ellipse_linalg(self.x, self.y)
                elif self._method == 'skimage': self._fit_ellipse_skimage(self.x, self.y)
                elif self._method == 'leastsquare': self._fit_ellipse_leastsq(self.x, self.y)
        
        self.update_regular_parameters()

    def save_hdf5(self, group):
        """Dynamically saves attributes, now handling empty lists, np.bool_, and UI fallbacks."""
        for key, value in self.__dict__.items():
            if key.startswith('__'): 
                continue

            if value is None:
                group.attrs[key] = "NONE_TYPE_FLAG"
            # BUG FIX 1: Add np.generic to catch np.bool_, np.int64, etc.
            elif isinstance(value, (int, float, str, bool, np.generic)):
                group.attrs[key] = value
            elif isinstance(value, np.ndarray):
                group.create_dataset(key, data=value)
            elif isinstance(value, (list, tuple)):
                # BUG FIX 2: Explicitly flag empty lists so HDF5 doesn't crash on np.array([])
                if len(value) == 0:
                    group.attrs[key] = "EMPTY_LIST_FLAG"
                else:
                    try:
                        group.create_dataset(key, data=np.array(value))
                    except Exception:
                        pass
            elif isinstance(value, dict):
                dict_group = group.create_group(key)
                for k, v in value.items():
                    if isinstance(v, (int, float, str, bool, np.generic)):
                        dict_group.attrs[k] = v
                    elif isinstance(v, np.ndarray):
                        dict_group.create_dataset(k, data=v)
            elif type(value).__name__ == 'Polygon':
                if hasattr(value, 'exterior'):
                    group.create_dataset(f"{key}_poly_coords", data=np.array(value.exterior.coords))
                else:
                    # BUG FIX 3: Initialize skipped UI polygons to None on load
                    group.attrs[key] = "NONE_TYPE_FLAG" 

    @classmethod
    def load_hdf5(cls, group):
        """Dynamically recreates the PlasmaStructure, resilient to missing/changed features."""
        obj = cls.__new__(cls)
        obj.__dict__ = {} 
        
        # 1. Load Attributes
        for key, val in group.attrs.items():
            if val == "NONE_TYPE_FLAG":
                setattr(obj, key, None)
            elif val == "EMPTY_LIST_FLAG":
                setattr(obj, key, [])
            else:
                setattr(obj, key, val)
                
        # 2. Load Datasets and Groups
        for key in group:
            item = group[key]
            
            if isinstance(item, h5py.Dataset):
                if key.endswith('_poly_coords'):
                    import shapely.geometry
                    orig_key = key.replace('_poly_coords', '')
                    setattr(obj, orig_key, shapely.geometry.Polygon(item[:]))
                else:
                    val = item[:]
                    if isinstance(val, np.ndarray) and val.dtype.kind in ['S', 'U', 'O']:
                        val = val.astype(str).tolist() if len(val.shape) > 0 else str(val)
                    else:
                        val = val.tolist() if len(val.shape) == 1 else val 
                    setattr(obj, key, val)
                    
            elif isinstance(item, h5py.Group):
                d = {}
                for k, v in item.attrs.items(): d[k] = v
                for k in item: d[k] = item[k][:]
                setattr(obj, key, d)
                
        return obj

    # --- Dictionary Backwards Compatibility ---
    def __getitem__(self, key):
        if key == 'Polygon': return self
        if key in ['Ellipse', 'Gaussian']: return self
        if key == 'Regular parameters': return self.regular_parameters
        if key == 'Differential parameters': return self.differential_parameters
        if key == 'Half path': return getattr(self, 'half_path', None)
        
        # Fallback for explicit numpy arrays
        if key == 'X coord': return self.x_data
        if key == 'Y coord': return self.y_data
        if key == 'Data': return self.data
        
        if key in self.attribute_map:
            return getattr(self, self.attribute_map[key])
        raise KeyError(f"Key '{key}' not found in PlasmaStructure mapping.")

    def __setitem__(self, key, value):
        if key in self.attribute_map:
            setattr(self, self.attribute_map[key], value)
        elif key == 'Differential parameters':
            self.differential_parameters = value
        elif key == 'Regular parameters':
            self.regular_parameters = value
        else:
            raise KeyError(f"Cannot set dictionary key '{key}' on PlasmaStructure.")
            

class TrackedPlasmaStructure:
    """A time-resolved plasma structure spanning multiple frames."""
    
    def __init__(self, label: int, start_time: float):
        self.label = label
        self.start_time = start_time
        self.time = []  
        
        self.regular_parameters = ParameterDict()
        self.differential_parameters = ParameterDict()

    def add_step(self, struct: PlasmaStructure, current_time: float):
        """Ingests raw floats and packages them into time-series MetricArrays."""
        self.time.append(current_time)
        
        # 1. Harvest Regular Parameters (Raw Floats -> MetricArray)
        for key, val in struct.regular_parameters.items():
            if key not in self.regular_parameters:
                # Lookup the LaTeX label and units from the single-frame struct!
                meta = struct.get_meta(key)
                
                self.regular_parameters[key] = MetricArray(
                    value=np.array([val], dtype=float),
                    dict_label=key,
                    plot_label=meta['plot_label'],
                    unit=meta['unit'],
                )
            else:
                current_array = self.regular_parameters[key].value
                self.regular_parameters[key].value = np.append(current_array, val)
                
                
    def save_hdf5(self, group):
        """Recursively saves all data, perfectly mirroring the object's exact state."""
        def _save_node(grp, name, val):
            if val is None:
                grp.attrs[name] = "NONE_TYPE_FLAG"
            elif isinstance(val, (int, float, str, bool, np.generic)):
                grp.attrs[name] = val
            elif isinstance(val, np.ndarray):
                grp.create_dataset(name, data=val)
            elif isinstance(val, list):
                if len(val) == 0:
                    grp.attrs[name] = "EMPTY_LIST_FLAG"
                else:
                    l_grp = grp.create_group(name)
                    l_grp.attrs['__is_list__'] = True
                    for i, item in enumerate(val):
                        _save_node(l_grp, f"item_{i}", item)
            elif isinstance(val, dict):
                d_grp = grp.create_group(name)
                d_grp.attrs['__is_dict__'] = True
                for k, v in val.items():
                    _save_node(d_grp, str(k), v)
            elif hasattr(val, 'save_hdf5'):
                obj_grp = grp.create_group(name)
                val.save_hdf5(obj_grp)
            elif type(val).__name__ == 'Polygon':
                if hasattr(val, 'exterior'):
                    grp.create_dataset(f"{name}_poly_coords", data=np.array(val.exterior.coords))
                else:
                    grp.attrs[name] = "NONE_TYPE_FLAG"
            elif hasattr(val, '__dict__'):
                obj_grp = grp.create_group(name)
                obj_grp.attrs['__is_custom_obj__'] = True
                for k, v in val.__dict__.items():
                    if not k.startswith('__'):
                        _save_node(obj_grp, k, v)
                        
        for key, value in self.__dict__.items():
            if not key.startswith('__'):
                _save_node(group, key, value)

    @classmethod
    def load_hdf5(cls, group):
        """Recursively recreates the tracked structure without missing a single edge case."""
        def _load_node(grp, name):
            # Intercept attributes first
            if name in grp.attrs:
                val = grp.attrs[name]
                if val == "NONE_TYPE_FLAG": return None
                if val == "EMPTY_LIST_FLAG": return []
                return val
                
            item = grp[name]
            if isinstance(item, h5py.Dataset):
                if name.endswith('_poly_coords'):
                    import shapely.geometry
                    return shapely.geometry.Polygon(item[:])
                val = item[:]
                if isinstance(val, np.ndarray) and val.dtype.kind in ['S', 'U', 'O']:
                    val = val.astype(str).tolist() if len(val.shape) > 0 else str(val)
                else:
                    val = val.tolist() if len(val.shape) == 1 else val
                return val
                
            elif isinstance(item, h5py.Group):
                if item.attrs.get('__is_list__', False):
                    items = {}
                    for k in item.attrs:
                        if k.startswith('item_'): 
                            idx = int(k.replace('_poly_coords', '').split('_')[1])
                            items[idx] = _load_node(item, k)
                    for k in item:
                        idx = int(k.replace('_poly_coords', '').split('_')[1])
                        items[idx] = _load_node(item, k)
                    return [items[i] for i in sorted(items.keys())]
                    
                elif item.attrs.get('__is_dict__', False):
                    res = {}
                    for k in item.attrs:
                        if not k.startswith('__'): 
                            res[k.replace('_poly_coords', '')] = _load_node(item, k)
                    for k in item:
                        res[k.replace('_poly_coords', '')] = _load_node(item, k)
                    return res
                    
                elif item.attrs.get('__is_custom_obj__', False):
                    class DynamicParameter: pass
                    obj = DynamicParameter()
                    for k in item.attrs:
                        if not k.startswith('__'): 
                            setattr(obj, k.replace('_poly_coords', ''), _load_node(item, k))
                    for k in item:
                        setattr(obj, k.replace('_poly_coords', ''), _load_node(item, k))
                    return obj
                    
                else:
                    try:
                        return PlasmaStructure.load_hdf5(item)
                    except Exception:
                        pass
                        
        obj = cls.__new__(cls)
        obj.__dict__ = {}
        
        for k in group.attrs:
            if not k.startswith('__'): 
                setattr(obj, k.replace('_poly_coords', ''), _load_node(group, k))
        for k in group:
            setattr(obj, k.replace('_poly_coords', ''), _load_node(group, k))
            
        return obj


class StructureDataset:
    """
    A unified container that holds EITHER frame-by-frame untracked structures 
    OR fully tracked time-series structures.
    """
    def __init__(self, mode='untracked', exp_id=None):
        self.mode = mode.lower()
        self.exp_id = exp_id
        
        if self.mode not in ['untracked', 'tracked']:
            raise ValueError("StructureDataset mode must be 'untracked' or 'tracked'.")
            
        self.frames = []       
        self.frame_times = []  
        self.tracked_structures = [] 

    def add_frame(self, structures: list, time: float):
        if self.mode != 'untracked':
            raise RuntimeError("Cannot add raw frames to a dataset in 'tracked' mode.")
        self.frames.append(structures)
        self.frame_times.append(time)
        
    def add_tracked_structure(self, tracked_struct):
        if self.mode != 'tracked':
            raise RuntimeError("Cannot add tracked structures to an 'untracked' dataset.")
        self.tracked_structures.append(tracked_struct)

    def get_structure_by_label(self, label: int):
        if self.mode != 'tracked':
            raise RuntimeError("Cannot search by label in an 'untracked' dataset.")
        for struct in self.tracked_structures:
            if struct.label == label:
                return struct
        raise KeyError(f"Structure with label {label} not found.")

    def __getattr__(self, attr_name):
        """
        Magically intercepts attribute requests (like dataset.area)
        and packages the raw floats into MetricArrays using PlasmaStructure.METADATA.
        """
        if attr_name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr_name}'")
            
        if self.mode == 'untracked':
            structs = [s for frame in self.frames for s in frame]
        else:
            structs = self.tracked_structures
            
        if not structs:
            return np.array([])
            
        first_struct = structs[0]
        if not hasattr(first_struct, attr_name):
            raise AttributeError(f"Contained structures have no attribute '{attr_name}'")
            
        harvested = [getattr(s, attr_name) for s in structs]
        
        # --- Smart Packaging via METADATA ---
        if attr_name in PlasmaStructure.METADATA:
            meta = PlasmaStructure.METADATA[attr_name]
            dist_unit = getattr(first_struct, '_distance_unit', 'DU')
            
            def _get_unit(u):
                return u.replace('DU', dist_unit) if 'DU' in u else u
            
            # Case A: It's a single metric mapped by a dict
            if isinstance(meta, dict):
                return MetricArray(
                    value=np.array(harvested, dtype=float),
                    dict_label=meta['dict_label'],
                    plot_label=meta['plot_label'],
                    unit=_get_unit(meta['unit']),
                )
                
            # Case B: It's a list of metrics mapped by a list of dicts
            elif isinstance(meta, list):
                transposed = list(zip(*harvested))
                metric_arrays = []
                for i, sub_meta in enumerate(meta):
                    metric_arrays.append(MetricArray(
                        value=np.array(transposed[i], dtype=float),
                        dict_label=sub_meta['dict_label'],
                        plot_label=sub_meta['plot_label'],
                        unit=_get_unit(sub_meta['unit']),
                    ))
                return metric_arrays
                
        # Case C: It's a standard variable (e.g., .label) missing from METADATA
        return np.array(harvested)
    
    def save_hdf5(self, filename):
        """Master save function to write the entire dataset to an HDF5 file."""
        with h5py.File(filename, 'w') as f:
            f.attrs['mode'] = self.mode
            f.attrs['exp_id'] = self.exp_id if self.exp_id is not None else "NONE_TYPE_FLAG"
            f.create_dataset('frame_times', data=np.array(self.frame_times))

            if self.mode == 'untracked':
                frames_grp = f.create_group('frames')
                for i_frame, frame_structures in enumerate(self.frames):
                    frame_grp = frames_grp.create_group(f"frame_{i_frame}")
                    for j_str, struct in enumerate(frame_structures):
                        str_grp = frame_grp.create_group(f"struct_{j_str}")
                        struct.save_hdf5(str_grp)

            elif self.mode == 'tracked':
                tracked_grp = f.create_group('tracked_structures')
                # BUG FIX: Iterate over the list, not a dictionary!
                for i, tracked_blob in enumerate(self.tracked_structures):
                    # Extract the label from the object (fallback to the index 'i' just in case)
                    label = getattr(tracked_blob, 'label', i)
                    blob_grp = tracked_grp.create_group(f"blob_{label}")
                    tracked_blob.save_hdf5(blob_grp)

    @classmethod
    def load_hdf5(cls, filename):
        """Master load function to reconstruct the dataset from an HDF5 file."""
        with h5py.File(filename, 'r') as f:
            mode = f.attrs['mode']
            exp_id = f.attrs['exp_id']
            if exp_id == "NONE_TYPE_FLAG": exp_id = None
            
            dataset = cls(mode=mode, exp_id=exp_id)
            if 'frame_times' in f:
                dataset.frame_times = f['frame_times'][:].tolist()

            if mode == 'untracked' and 'frames' in f:
                frames_grp = f['frames']
                
                # Ensure we load frames in the correct integer order
                frame_keys = sorted(frames_grp.keys(), key=lambda x: int(x.split('_')[1]))
                for frame_key in frame_keys:
                    frame_grp = frames_grp[frame_key]
                    current_frame_structures = []
                    
                    for str_key in frame_grp:
                        str_grp = frame_grp[str_key]
                        reconstructed_struct = PlasmaStructure.load_hdf5(str_grp)
                        current_frame_structures.append(reconstructed_struct)
                        
                    dataset.frames.append(current_frame_structures)

            elif mode == 'tracked' and 'tracked_structures' in f:
                tracked_grp = f['tracked_structures']
                for blob_key in tracked_grp:
                    blob_grp = tracked_grp[blob_key]
                    reconstructed_blob = TrackedPlasmaStructure.load_hdf5(blob_grp)
                    dataset.add_tracked_structure(reconstructed_blob)

        return dataset