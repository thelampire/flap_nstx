#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 16:53:40 2025

@author: mlampert
"""
import math
import numpy as np
import pandas as pd

def rectangle_corners(R, Z):
    """Return the rectangle bounding the separatrix."""
    R0 = np.min(R)   # left (inner) edge
    Rout = np.max(R) # right (outer) edge
    Z0 = np.min(Z)   # bottom edge
    Ztop = np.max(Z) # top edge
    return R0, Rout, Z0, Ztop

def diagonal_quarter_ellipse_intersection_height(R0, Rout, Z0, Ztop):
    """
    Compute intersection point of diagonal and quarter ellipse.
    For rectangle [R0,Rout] x [Z0,Ztop], intersection is at t=1/sqrt(2).
    """
    a = Rout - R0
    b = Ztop - Z0
    t = 1.0 / math.sqrt(2.0)
    R_star = R0 + a * t
    Z_star = Z0 + b * t
    return R_star, Z_star, t

def find_R_at_Z(R, Z, Z_query, tol=1e-12):
    """Find all R values where the separatrix crosses the horizontal line Z=Z_query."""
    R = np.asarray(R)
    Z = np.asarray(Z)
    Rs = []
    for i in range(len(R)-1):
        z1, z2 = Z[i], Z[i+1]
        if (z1 - Z_query) * (z2 - Z_query) <= 0:  # crossing or touching
            if abs(z2 - z1) < tol:
                Rs.append(R[i])
                Rs.append(R[i+1])
            else:
                t = (Z_query - z1) / (z2 - z1)
                r = R[i] + t * (R[i+1] - R[i])
                Rs.append(r)
    # unique sorted values
    return sorted(list(set([round(float(r),12) for r in Rs])))

def compute_squareness_metrics(R, Z):
    """Compute geometric squareness at Z_star height."""
    R0, Rout, Z0, Ztop = rectangle_corners(R, Z)
    R_star, Z_star, t = diagonal_quarter_ellipse_intersection_height(R0, Rout, Z0, Ztop)
    R_at_Zstar = find_R_at_Z(R, Z, Z_star)

    metrics = {
        "R0": R0, "Rout": Rout, "Z0": Z0, "Ztop": Ztop,
        "R_star": R_star, "Z_star": Z_star,
        "R_at_Zstar_list": R_at_Zstar
    }

    if len(R_at_Zstar) >= 2:
        Rmin_line = min(R_at_Zstar)
        Rmax_line = max(R_at_Zstar)
        Rmean_line = 0.5*(Rmin_line + Rmax_line)
        width_total = Rmax_line - Rmin_line if abs(Rmax_line-Rmin_line)>0 else np.nan

        left_frac  = (Rmean_line - Rmin_line)/width_total if width_total else np.nan
        right_frac = (Rmax_line - Rmean_line)/width_total if width_total else np.nan

        inner_rel = (Rmin_line - R0) / (Rout - R0)
        outer_rel = (Rmax_line - R0) / (Rout - R0)

        metrics.update({
            "Rmin_at_Zstar": Rmin_line,
            "Rmax_at_Zstar": Rmax_line,
            "Rmean_at_Zstar": Rmean_line,
            "width_total_at_Zstar": width_total,
            "left_frac_at_Zstar": left_frac,
            "right_frac_at_Zstar": right_frac,
            "inner_rel_to_rectangle": inner_rel,
            "outer_rel_to_rectangle": outer_rel
        })
    else:
        metrics["note"] = "Less than two intersections found at Z_star."

    return metrics