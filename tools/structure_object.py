#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 10:07:14 2026

@author: mlampert
"""
from flap_nstx.tools import MetricArray, FitShape, Polygon
import numpy as np

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