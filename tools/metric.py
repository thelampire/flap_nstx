#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 15:49:19 2026

@author: mlampert
"""
from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass
class MetricArray:
    value: Any
    dict_label: str
    plot_label: str
    unit: str

    def __post_init__(self):
        # Guarantee it is always at least a 1D NumPy array
        self.value = np.atleast_1d(np.asarray(self.value))

    # --- Helpers ---
    def _get_val(self, other):
        return other.value if isinstance(other, MetricArray) else other
        
    def _get_unit(self, other):
        return other.unit if isinstance(other, MetricArray) else ''
        
    def _get_plot_label(self, other):
        return other.plot_label if isinstance(other, MetricArray) else ''
    
    def _get_dict_label(self, other):
        return other.dict_label if isinstance(other, MetricArray) else ''

    def _check_unit_compatibility(self, other):
        """Helper to prevent comparing or adding apples to oranges."""
        other_unit = self._get_unit(other)
        if other_unit != '' and self.unit != '' and other_unit != self.unit:
            raise ValueError(f"Incompatible units: '{self.unit}' and '{other_unit}'.")

    # ==========================================
    # MATH OPERATORS
    # ==========================================

    # --- Unary Operators ---
    def __neg__(self):
        """Allows placing a minus sign directly in front of the MetricArray"""
        return self.__class__(
            value=-self.value,
            dict_label=self.dict_label,
            plot_label=self.plot_label,
            unit=self.unit,
            
        )

    def __pos__(self):
        """Allows placing a plus sign directly in front of the MetricArray"""
        return self.__class__(
            value=+self.value,
            dict_label=self.dict_label,
            plot_label=self.plot_label,
            unit=self.unit,
            
        )

    def __abs__(self):
        """Allows using the built-in abs() function on the MetricArray"""
        return self.__class__(
            value=np.abs(self.value),
            dict_label=self.dict_label,
            plot_label=self.plot_label,
            unit=self.unit,
            
        )

    # --- Addition (+) ---
    def __add__(self, other):
        self._check_unit_compatibility(other)
        return self.__class__(
            value=self.value + self._get_val(other),
            dict_label=self.dict_label,
            plot_label=self.plot_label,
            unit=self.unit,
            
        )

    def __radd__(self, other):
        self._check_unit_compatibility(other)
        return self.__class__(
            value=self._get_val(other) + self.value,
            dict_label=self.dict_label,
            plot_label=self.plot_label,
            unit=self.unit,
            
        )

    # --- Subtraction (-) ---
    def __sub__(self, other):
        self._check_unit_compatibility(other)
            
        if self.plot_label == self._get_plot_label(other):
            new_plot_label = r"$\Delta " + f"{self.plot_label.replace('$','')}$"
            new_dict_label = self.dict_label + ' diff'
        else:
            new_plot_label = self.plot_label
            new_dict_label = self.dict_label + ' minus ' + self._get_dict_label(other)
            
        return self.__class__(
            value=self.value - self._get_val(other),
            dict_label=new_dict_label,
            plot_label=new_plot_label,
            unit=self.unit,
            
        )

    def __rsub__(self, other):
        self._check_unit_compatibility(other)
        other_plot_label = self._get_plot_label(other)
        
        if self.plot_label == other_plot_label:
            new_plot_label = r"$\Delta " + f"{self.plot_label.replace('$','')}$"
            new_dict_label = self.dict_label + ' diff'
        else:
            new_plot_label = other_plot_label + ' - ' + self.plot_label
            new_dict_label = f"{self._get_dict_label(other)} minus {self.dict_label}"
            
        return self.__class__(
            value=self._get_val(other) - self.value,
            dict_label=new_dict_label,
            plot_label=new_plot_label,
            unit=self.unit,
            
        )

    # --- Multiplication (*) ---
    def __mul__(self, other):
        other_plot_label = self._get_plot_label(other)
        
        if other_plot_label == self.plot_label:
            new_plot_label = f"${self.plot_label.replace('$', '')}^2$"
            new_dict_label = f"{self.dict_label} square"
        elif other_plot_label != '':
            new_plot_label = f"${self.plot_label.replace('$', '')} \cdot {other_plot_label.replace('$', '')}$"
            new_dict_label = f"{self.dict_label} * {self._get_dict_label(other)}"
        else:
            new_plot_label = self.plot_label
            new_dict_label = self.dict_label
            
        other_unit = self._get_unit(other)
        if other_unit == self.unit:
            new_unit = f"${self.unit.replace('$', '')}^2$"
        elif other_unit != '':
            new_unit = f"${self.unit.replace('$', '')} \cdot {other_unit.replace('$', '')}$"
        else:
            new_unit = self.unit
            
        return self.__class__(
            value=self.value * self._get_val(other),
            dict_label=new_dict_label,
            plot_label=new_plot_label,
            unit=new_unit,
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    # --- True Division (/) ---
    def __truediv__(self, other):
        other_label = self._get_plot_label(other)
        
        if other_label == self.plot_label:
            new_label = ''
            new_dict_label = f"{self.dict_label} per {self._get_dict_label(other)}"
        elif other_label != '' and self.plot_label != '':
            new_label = f"${self.plot_label.replace('$', '')}/{other_label.replace('$', '')}$"
            new_dict_label = f"{self.dict_label} per {self._get_dict_label(other)}"
        else:
            new_label = self.plot_label
            new_dict_label = self.dict_label
            
        other_unit = self._get_unit(other)
        if other_unit == self.unit:
            new_unit = ''
        elif other_unit != '' and self.unit != '':
            new_unit = f"${self.unit.replace('$', '')}/{other_unit.replace('$', '')}$"
        elif other_unit != '' and self.unit == '':
            new_unit = f"${other_unit.replace('$', '')}^"+r"{-1}$"
        else:
            new_unit = self.unit
            
        return self.__class__(
            value=self.value / self._get_val(other),
            dict_label=new_dict_label,
            plot_label=new_label,
            unit=new_unit,
        )

    def __rtruediv__(self, other):
        other_label = self._get_plot_label(other)
        if other_label == self.plot_label:
            new_label = ''
            new_dict_label = f"{self._get_dict_label(other)} per {self.dict_label}"
        elif other_label != '':
            new_label = f"${other_label.replace('$', '')}/{self.plot_label.replace('$', '')}$"
            new_dict_label = f"{self._get_dict_label(other)} per {self.dict_label}"
        else:
            new_label = f"$1/{self.plot_label.replace('$', '')}$" if self.plot_label else ''
            new_dict_label = f"{self.dict_label} inv"
            
        other_unit = self._get_unit(other)
        if other_unit == self.unit:
            new_unit = ''
        elif other_unit != '':
            new_unit = f"${other_unit.replace('$', '')}/{self.unit.replace('$', '')}$"
        else:
            new_unit = f"${self.unit.replace('$', '')}^"+r"{-1}$" if self.unit else ''
            if self.unit.startswith('$'):
                new_unit = f"${new_unit}$" # Preserve LaTeX wrapper if it had one
            
        return self.__class__(
            value=self._get_val(other) / self.value,
            dict_label=new_dict_label,
            plot_label=new_label,
            unit=new_unit,
        )

    # --- Power/Exponents (**) ---
    def __pow__(self, other):
        if not isinstance(other, (int, float, complex, np.number)):
            raise ValueError('MetricArray classes can only be raised to the power of a number.')
            
        val = self._get_val(other)
        return self.__class__(
            value=self.value ** val,
            dict_label=f"{self.dict_label} power {val}",
            plot_label=f"${self.plot_label.replace('$', '')}^{{{val}}}$",
            unit=f"${self.unit.replace('$', '')}^{{{val}}}$",             
        )
    
    # --- Modulo / Remainder (%) ---
    def __mod__(self, other):
        return self.__class__(
            value=self.value % self._get_val(other),
            dict_label=self.dict_label,
            plot_label=self.plot_label,
            unit=self.unit,
            
        )

    def __rmod__(self, other):
        return self.__class__(
            value=self._get_val(other) % self.value,
            dict_label=self.dict_label,
            plot_label=self.plot_label,
            unit=self.unit,
            
        )

    # --- Rich Comparisons ---
    def __lt__(self, other):
        self._check_unit_compatibility(other)
        return self.value < self._get_val(other)

    def __le__(self, other):
        self._check_unit_compatibility(other)
        return self.value <= self._get_val(other)

    def __gt__(self, other):
        self._check_unit_compatibility(other)
        return self.value > self._get_val(other)

    def __ge__(self, other):
        self._check_unit_compatibility(other)
        return self.value >= self._get_val(other)

    def __eq__(self, other):
        other_unit = self._get_unit(other)
        if other_unit != '' and self.unit != '' and other_unit != self.unit:
            return False
        return self.value == self._get_val(other)

    def __ne__(self, other):
        return not self.__eq__(other)
    
    # --- Type casting ---
    def __array__(self, dtype=None):
        """Allows NumPy functions (like np.isnan) to unpack the MetricArray automatically."""
        return np.asarray(self.value, dtype=dtype)
    
    def __float__(self):
        """Allows Python and NumPy to cast a 1-element MetricArray directly to a float."""
        return float(self.value)
        
    def __int__(self):
        """Allows Python and NumPy to cast a 1-element MetricArray directly to an int."""
        return int(self.value)

    # ==========================================
    # ARRAY / SEQUENCE BEHAVIORS
    # ==========================================
    
    def append(self, other):
        """Appends a raw number or another MetricArray's value to the end of the array."""
        if isinstance(other, MetricArray):
            self._check_unit_compatibility(other)
            
        val_to_append = self._get_val(other)
        self.value = np.append(self.value, val_to_append)
        
    def __getitem__(self, index):
        """Allows extracting items via my_array[index] or slices via my_array[1:4]."""
        extracted_val = self.value[index]
        
        # Always return a MetricArray to preserve labels and units. 
        # (Due to __post_init__, scalar slices will safely become 1D arrays of length 1)
        return self.__class__(
            value=extracted_val,
            dict_label=self.dict_label,
            plot_label=self.plot_label,
            unit=self.unit,
            
        )

    def __setitem__(self, index, new_value):
        """Allows overwriting items via my_array[0] = 5 or my_array[0] = MetricArray(...)."""
        self.value[index] = self._get_val(new_value)

    def __len__(self):
        """Allows you to call len(my_array)."""
        return len(self.value)
        
    def __iter__(self):
        """Allows you to loop over the array: for item in my_array: ..."""
        for val in self.value:
            yield self.__class__(
                value=val, 
                dict_label=self.dict_label,
                plot_label=self.plot_label, 
                unit=self.unit, 
                
            )
            
    # --- Native List Methods ---
    
    def __delitem__(self, index):
        """Allows deleting items via: del my_array[index]"""
        self.value = np.delete(self.value, index)

    def pop(self, index=-1):
        """Removes and returns the item at the given index (defaults to the last item)."""
        item = self[index] 
        self.__delitem__(index)
        return item

    def extend(self, iterable):
        """Extends the array by appending elements from another list or MetricArray."""
        vals_to_add = [self._get_val(x) for x in iterable]
        self.value = np.append(self.value, vals_to_add)

    def insert(self, index, item):
        """Inserts an item at a given position."""
        self.value = np.insert(self.value, index, self._get_val(item))

    def remove(self, item):
        """Removes the first item from the array whose value matches the input."""
        val_to_remove = self._get_val(item)
        indices = np.where(self.value == val_to_remove)[0]
        if len(indices) == 0:
            raise ValueError("MetricArray.remove(x): x not in array")
        self.__delitem__(indices[0])

    def clear(self):
        """Removes all items from the array."""
        self.value = np.array([])

    def index(self, item):
        """Returns the zero-based index of the first matching item."""
        val_to_find = self._get_val(item)
        indices = np.where(self.value == val_to_find)[0]
        if len(indices) == 0:
            raise ValueError(f"{val_to_find} is not in array")
        return int(indices[0])

    def count(self, item):
        """Returns the number of times an item appears in the array."""
        val_to_count = self._get_val(item)
        return int(np.sum(self.value == val_to_count))

    def sort(self, reverse=False):
        """Sorts the items of the array in place."""
        self.value = np.sort(self.value)
        if reverse:
            self.reverse()

    def reverse(self):
        """Reverses the elements of the array in place."""
        self.value = self.value[::-1]
        
    def copy(self):
        """Returns a shallow copy of the MetricArray."""
        return self.__class__(
            value=self.value.copy(),
            dict_label=self.dict_label,
            plot_label=self.plot_label,
            unit=self.unit,
            
        )