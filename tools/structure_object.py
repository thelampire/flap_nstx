#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 10:07:14 2026

@author: mlampert
"""
from flap_nstx.tools import Metric, MetricArray, FitShape, Polygon
import numpy as np

class PlasmaStructure(Polygon, FitShape):
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
        
        # Initialize explicit, static dictionaries
        self.differential_parameters = ParameterDict()
        self.regular_parameters = ParameterDict()
        
        # Populate the regular_parameters physically!
        self.update_regular_parameters()

    def update_regular_parameters(self):
        """Dynamically harvests Metric objects and explicitly locks them into the dictionary."""
        self.regular_parameters.clear()
        
        for attr_name in dir(self):
            if attr_name.startswith('_'): 
                continue
            try: 
                val = getattr(self, attr_name)
            except Exception: 
                continue
            
            if callable(val): 
                continue
            
            # STRICT REQUIREMENT: Only save it if it is a Metric!
            if isinstance(val, Metric) and val.dict_label:
                self.regular_parameters[val.dict_label] = val
            elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], Metric):
                for m in val:
                    if m.dict_label:
                        self.regular_parameters[m.dict_label] = m

    def smooth(self, refinements=1):
        """Overrides Polygon.smooth to ensure metrics update if boundaries change."""
        # Smooth the polygon boundaries
        super().smooth(refinements=refinements)
        
        # Re-evaluate mathematical fits if the boundary changed (only applies to ellipses)
        if self.fitting_type == 'ellipse':
            if hasattr(self, '_method'):
                if self._method == 'linalg': self._fit_ellipse_linalg(self.x, self.y)
                elif self._method == 'skimage': self._fit_ellipse_skimage(self.x, self.y)
                elif self._method == 'leastsquare': self._fit_ellipse_leastsq(self.x, self.y)
        
        # Refresh the dictionary with the newly calculated Metrics!
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
        self.time = []  # Raw list of time steps (can easily be cast to np.array)
        
        # We reuse your awesome hybrid dictionaries to hold MetricArrays!
        self.regular_parameters = ParameterDict()
        self.differential_parameters = ParameterDict()

    def add_step(self, struct: PlasmaStructure, current_time: float):
        """Ingests a single-frame structure and appends its data to the time series."""
        self.time.append(current_time)
        
        # 1. Harvest Regular Parameters
        for key, metric in struct.regular_parameters.items():
            if key not in self.regular_parameters:
                # First time seeing this metric! Initialize a MetricArray.
                self.regular_parameters[key] = MetricArray(
                    value=[metric.value],
                    dict_label=metric.dict_label,
                    plot_label=metric.plot_label,
                    unit=metric.unit,
                    multiplier=metric.multiplier
                )
            else:
                # We already have an array for this, just append the new Metric!
                self.regular_parameters[key].append(metric)
                
        # 2. Harvest Differential Parameters
        for key, metric in struct.differential_parameters.items():
            if key not in self.differential_parameters:
                self.differential_parameters[key] = MetricArray(
                    value=[metric.value],
                    dict_label=metric.dict_label,
                    plot_label=metric.plot_label,
                    unit=metric.unit,
                    multiplier=metric.multiplier
                )
            else:
                self.differential_parameters[key].append(metric)

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
            
        # --- UNTRACKED STORAGE (Frame-Centric) ---
        self.frames = []       # List of lists containing PlasmaStructures
        self.frame_times = []  # 1D List of time floats corresponding to each frame
        
        # --- TRACKED STORAGE (Structure-Centric) ---
        self.tracked_structures = [] # List of TrackedPlasmaStructure objects

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

    # ==========================================
    # THE VECTORIZATION ENGINE
    # ==========================================
    def __getattr__(self, attr_name):
        """
        Magically intercepts attribute requests (like dataset.area or dataset.fit_size)
        and broadcasts them across all contained structures.
        """
        # 1. Prevent infinite recursion for missing internal Python methods
        if attr_name.startswith('_'):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr_name}'")
            
        # 2. Flatten the structures we want to query
        if self.mode == 'untracked':
            # Flatten all frames into one giant list
            structs = [s for frame in self.frames for s in frame]
        else:
            structs = self.tracked_structures
            
        if not structs:
            return np.array([])
            
        # 3. Ensure the attribute actually exists on the structures
        first_struct = structs[0]
        if not hasattr(first_struct, attr_name):
            raise AttributeError(f"Contained structures have no attribute '{attr_name}'")
            
        # 4. Harvest the data!
        harvested = [getattr(s, attr_name) for s in structs]
        first_val = harvested[0]
        
        # --- Smart Packaging ---
        
        # Case A: It's a single Metric (e.g., .area, .fit_angle)
        if isinstance(first_val, Metric):
            return MetricArray(
                value=[m.value for m in harvested],
                dict_label=first_val.dict_label,
                plot_label=first_val.plot_label,
                unit=first_val.unit,
                multiplier=first_val.multiplier
            )
            
        # Case B: It's a list of Metrics (e.g., .fit_size -> [Metric(R), Metric(Z)])
        elif isinstance(first_val, list) and len(first_val) > 0 and isinstance(first_val[0], Metric):
            # Transpose the list of lists: [[R1, Z1], [R2, Z2]] -> [[R1, R2], [Z1, Z2]]
            transposed = list(zip(*harvested))
            metric_arrays = []
            for dim_metrics in transposed:
                ref = dim_metrics[0]
                metric_arrays.append(MetricArray(
                    value=[m.value for m in dim_metrics],
                    dict_label=ref.dict_label,
                    plot_label=ref.plot_label,
                    unit=ref.unit,
                    multiplier=ref.multiplier
                ))
            return metric_arrays
            
        # Case C: It's a standard variable (e.g., .label, .born)
        else:
            return np.array(harvested)

class ParameterDict(dict):
    """
    A strict dictionary that allows dot-notation access to keys (translating 
    underscores to spaces) and ONLY stores Metric objects.
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