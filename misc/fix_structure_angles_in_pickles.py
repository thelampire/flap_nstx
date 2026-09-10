#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recalculates the fit and the momentum (ALI) based angles and the angular
velocities in the already calculated structure pickle files.

The angles in the previously saved files are wrong for two reasons:
    1., The ALI angle was wrapped with np.arcsin(np.sin(angle)) which mirrors
        the angle instead of wrapping it, hence a part of the angle range is
        unreachable.
    2., The angle of the structures was set to np.nan when the elongation
        calculated from the axis aligned projected sizes was below the
        threshold. Those sizes are equal for any structure tilted by +-pi/4 no
        matter how elongated it is, hence the angles are missing around +-pi/4.

This script fixes both without re-reading the GPI data and without redoing the
structure identification or the tracking: the polygons and the fitted ellipse
boundaries are all stored in the pickle files, so the angles can be recomputed
from them directly.

@author: mlampert
"""

import argparse
import glob
import os
import pickle
import shutil

import numpy as np

import flap
import flap_nstx

thisdir = os.path.dirname(os.path.realpath(flap_nstx.__file__))
fn = os.path.join(thisdir, "flap_nstx.cfg")
flap.config.read(file_name=fn)
flap_nstx.register()

from flap_nstx.tools.shape_objects import FitShape
from flap_nstx.gpi import calculate_differential_structure_keys

ANGLE_KEYS = {'Angle fit': 'Angular velocity angle fit',
              'Angle ALI': 'Angular velocity ALI'}


def wrap_angle(angle):
    """Wrap a pi periodic angle onto the [-pi/2,pi/2) principal branch."""
    return np.mod(angle + np.pi/2, np.pi) - np.pi/2


def recalculate_structure_angles(structure, elongation_threshold=0.1):
    """Recalculate the angles of a single frame structure in place.

    Returns True when the structure has a valid angle after the recalculation.
    """
    # --- Momentum (ALI) based angle ---
    # Drop the cached value so the corrected implementation is evaluated.
    structure.__dict__.pop('principal_axes_angle', None)
    try:
        angle_ali = float(np.real(structure.principal_axes_angle))
    except Exception:
        angle_ali = np.nan

    # --- Fitted ellipse angle ---
    # The ellipse is refitted from the stored boundary. FitShape returns the
    # angle of the minor axis, the pi/2 shift rotates it onto the major axis so
    # it stays consistent with the ALI angle, the same way as in the original
    # calculation.
    angle_fit = np.nan
    axes_length = [np.nan, np.nan]
    if structure.x is not None and structure.y is not None:
        try:
            fit = FitShape(fitting='ellipse',
                           x=structure.x,
                           y=structure.y,
                           method='linalg')
            angle_fit = float(np.real(fit.fit_angle)) + np.pi/2
            axes_length = np.real(np.asarray(fit.fit_axes_length, dtype=complex)).astype(float)
        except Exception:
            angle_fit = np.nan

    # --- Elongation check on the ellipse axes, not on the projected sizes ---
    axes_sum = np.sum(axes_length)
    if np.any(np.isnan(axes_length)) or axes_sum == 0:
        elongation = np.nan
    else:
        elongation = np.abs(axes_length[0] - axes_length[1]) / axes_sum

    if np.isnan(elongation) or elongation < elongation_threshold:
        angle_fit = np.nan
        angle_ali = np.nan
        valid = False
    else:
        angle_fit = wrap_angle(angle_fit) if not np.isnan(angle_fit) else np.nan
        angle_ali = wrap_angle(angle_ali) if not np.isnan(angle_ali) else np.nan
        valid = True

    structure._angle = angle_fit
    structure.regular_parameters['Angle fit'] = angle_fit
    structure.regular_parameters['Angle ALI'] = angle_ali

    return valid


def unwrap_angle_series(angles):
    """Unwrap a pi periodic angle time series for the derivative calculation.

    np.diff on a wrapped angle produces spurious ~pi jumps whenever the angle
    crosses the branch edge. The series is unwrapped with a period of pi so the
    angular velocity stays physical. NaNs are kept in place and the unwrapping
    is restarted after each gap.
    """
    angles = np.asarray(angles, dtype=float)
    unwrapped = np.full_like(angles, np.nan)
    valid = ~np.isnan(angles)
    if not np.any(valid):
        return unwrapped
    # np.unwrap with period=np.pi handles the pi periodicity of the axes angles
    unwrapped[valid] = np.unwrap(angles[valid], period=np.pi)
    return unwrapped


def recalculate_dataset_angles(dataset, elongation_threshold=0.1,
                               unwrap_for_velocity=True):
    """Recalculate the angles and the angular velocities of a whole dataset."""
    n_struct = 0
    n_valid = 0

    for tracked in dataset.tracked_structures:
        if not tracked or not tracked.structures:
            continue

        angles = {key: [] for key in ANGLE_KEYS}

        for structure in tracked.structures:
            n_struct += 1
            if recalculate_structure_angles(structure,
                                            elongation_threshold=elongation_threshold):
                n_valid += 1
            structure.update_regular_parameters()
            for key in ANGLE_KEYS:
                angles[key].append(structure.regular_parameters[key])

        # Overwrite the time series of the tracked structure
        for key in ANGLE_KEYS:
            values = np.asarray(angles[key], dtype=float)
            if key in tracked.regular_parameters:
                metric = tracked.regular_parameters[key]
                if len(values) == len(metric.value):
                    metric.value = values

    # The angular velocities are recalculated from the unwrapped angles so the
    # branch jumps don't show up as huge spurious rotation rates.
    original = {}
    if unwrap_for_velocity:
        for tracked in dataset.tracked_structures:
            for key in ANGLE_KEYS:
                if key in tracked.regular_parameters:
                    metric = tracked.regular_parameters[key]
                    original[(id(tracked), key)] = metric.value.copy()
                    metric.value = unwrap_angle_series(metric.value)

    dataset = calculate_differential_structure_keys(dataset)

    if unwrap_for_velocity:
        # Restore the wrapped angles, only the derivatives needed the unwrapping
        for tracked in dataset.tracked_structures:
            for key in ANGLE_KEYS:
                saved = original.get((id(tracked), key))
                if saved is not None:
                    tracked.regular_parameters[key].value = saved

    return dataset, n_struct, n_valid


def fix_structure_angles_in_pickles(processed_data_dir=None,
                                    pattern='*structure_char*watershed.pickle',
                                    elongation_threshold=0.1,
                                    unwrap_for_velocity=True,
                                    backup=True,
                                    dry_run=False,
                                    filenames=None,
                                    ):
    """Recalculate the angles in every matching structure pickle file.

    Args:
        processed_data_dir (str, optional): Directory of the pickle files.
            Defaults to the processed_data directory of the working directory.
        pattern (str, optional): Glob pattern of the files to be fixed.
        elongation_threshold (float, optional): Same threshold as the one used
            in flap_nstx.gpi.identify_structures.validate_structure.
        unwrap_for_velocity (bool, optional): Unwrap the angles before the
            angular velocities are calculated.
        backup (bool, optional): Save the original file with a .bak extension
            before it is overwritten.
        dry_run (bool, optional): Do everything but don't write any file.
        filenames (list, optional): Explicit list of files, overrides pattern.
    """
    if processed_data_dir is None:
        wd = flap.config.get_all_section('Module NSTX_GPI')['Working directory']
        processed_data_dir = os.path.join(wd, 'processed_data')

    if filenames is None:
        filenames = sorted(glob.glob(os.path.join(processed_data_dir, pattern)))

    if not filenames:
        print(f'No file matches {pattern} in {processed_data_dir}')
        return []

    print(f'{len(filenames)} files are going to be processed in {processed_data_dir}')
    if dry_run:
        print('DRY RUN: no file is going to be overwritten.\n')

    failed = []
    for ind, filename in enumerate(filenames):
        basename = os.path.basename(filename)
        try:
            with open(filename, 'rb') as f:
                dataset = pickle.load(f)

            if getattr(dataset, 'mode', None) != 'tracked':
                print(f'({ind+1}/{len(filenames)}) {basename}: not tracked, skipped')
                continue

            dataset, n_struct, n_valid = recalculate_dataset_angles(
                dataset,
                elongation_threshold=elongation_threshold,
                unwrap_for_velocity=unwrap_for_velocity)

            if not dry_run:
                if backup and not os.path.exists(filename + '.bak'):
                    shutil.copy2(filename, filename + '.bak')
                # Write to a temporary file first so an interrupted run cannot
                # leave a half written pickle behind.
                tmp_filename = filename + '.tmp'
                with open(tmp_filename, 'wb') as f:
                    pickle.dump(dataset, f)
                os.replace(tmp_filename, filename)

            fraction = n_valid/n_struct*100 if n_struct else np.nan
            print(f'({ind+1}/{len(filenames)}) {basename}: '
                  f'{n_struct} structures, {fraction:.1f}% with a valid angle')

        except Exception as e:
            print(f'({ind+1}/{len(filenames)}) {basename}: FAILED, {e}')
            failed.append(filename)

    print(f'\nDone. {len(filenames)-len(failed)} files fixed, {len(failed)} failed.')
    if failed:
        for filename in failed:
            print(f'   failed: {os.path.basename(filename)}')
    return failed


def restore_backups(processed_data_dir=None, pattern='*structure_char*watershed.pickle.bak'):
    """Restore the .bak files written by fix_structure_angles_in_pickles."""
    if processed_data_dir is None:
        wd = flap.config.get_all_section('Module NSTX_GPI')['Working directory']
        processed_data_dir = os.path.join(wd, 'processed_data')

    filenames = sorted(glob.glob(os.path.join(processed_data_dir, pattern)))
    for filename in filenames:
        shutil.copy2(filename, filename[:-len('.bak')])
    print(f'{len(filenames)} files restored from the backups.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Recalculate the fit and ALI angles in the structure pickles.')
    parser.add_argument('--dir', default=None, help='Processed data directory')
    parser.add_argument('--pattern', default='*140384*structure_char*watershed.pickle',
                        help='Glob pattern of the files to be fixed')
    parser.add_argument('--elongation-threshold', type=float, default=0.1)
    parser.add_argument('--no-unwrap', action='store_true',
                        help="Don't unwrap the angles for the angular velocities")
    parser.add_argument('--no-backup', action='store_true',
                        help="Don't write the .bak files")
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--restore', action='store_true',
                        help='Restore the previously written .bak files')
    args = parser.parse_args()

    if args.restore:
        restore_backups(processed_data_dir=args.dir)
    else:
        fix_structure_angles_in_pickles(processed_data_dir=args.dir,
                                        pattern=args.pattern,
                                        elongation_threshold=args.elongation_threshold,
                                        unwrap_for_velocity=not args.no_unwrap,
                                        backup=not args.no_backup,
                                        dry_run=args.dry_run)
