#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 16:52:11 2026

@author: mlampert
"""
import numpy as np
from scipy.optimize import curve_fit
from functools import cached_property
from skimage.measure import EllipseModel
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon as PolygonShapely
from shapely import concave_hull

class Polygon:
    
    METADATA = {}

    def __init__(self,
                 x=None,  
                 y=None,  
                 smooth=0,
                 x_data=None,  
                 y_data=None,  
                 x_data_pix=None,  
                 y_data_pix=None,  
                 data=None,  
                 path_order=3,  
                 test=False,  
                 distance_unit='mm',
                 ):

        if (x is None or y is None) or len(x) != len(y):
            raise ValueError('The input x and y has to be defined and must have the same length.')

        self.x = np.asarray(x)
        self.y = np.asarray(y)

        self.x_data = None
        self.y_data = None
        self.x_data_pix = None
        self.y_data_pix = None

        self.data = None
        self.polygon_with_data = False

        if (x_data is not None and y_data is not None and data is not None):
            if (x_data.shape != y_data.shape or x_data.shape != data.shape):
                raise ValueError('The shapes of the input data do not match.')

            self.x_data = x_data
            self.y_data = y_data
            self.x_data_pix = x_data_pix
            self.y_data_pix = y_data_pix
            self.data = data
            self.polygon_with_data = True

        self._path = None
        self.path_order = path_order
        self.test = test
        self._distance_unit = distance_unit
        if smooth:
            self._smooth(refinements=smooth)
        self._create_valid_polygon()
        
    def _smooth(self, refinements=5):  
        coords = np.array([self.x, self.y]).T

        for _ in range(refinements):
            L = coords.repeat(2, axis=0)
            R = np.empty_like(L)
            R[0] = L[0]
            R[2::2] = L[1:-1:2]
            R[1:-1:2] = L[2::2]
            R[-1] = L[-1]
            coords = L * 0.75 + R * 0.25

        self.x = coords[:, 0].copy()
        self.y = coords[:, 1].copy()
        
    def _create_valid_polygon(self):
        poly = PolygonShapely(zip(self.x, self.y))

        if not poly.is_valid:
            poly = poly.buffer(0)

            if poly.geom_type == 'MultiPolygon':
                poly = max(poly.geoms, key=lambda p: p.area)
            elif poly.geom_type != 'Polygon':
                raise ValueError(f"Geometry resolution failed, returned {poly.geom_type}")

        self._shapely_polygon = poly
        coords = np.array(poly.exterior.coords)
        self.x = coords[:, 0]
        self.y = coords[:, 1]

    @property
    def shapely_polygon(self): 
        return self._shapely_polygon

    @cached_property
    def oriented_envelope(self):
        return self._shapely_polygon.oriented_envelope.normalize()

    # ==========================================
    # GEOMETRIC PROPERTIES & METADATA
    # ==========================================

    @cached_property
    def axes_length(self):
        bbox = np.asarray(self.oriented_envelope.exterior.coords)
        axis1 = np.linalg.norm(bbox[0] - bbox[3])
        axis2 = np.linalg.norm(bbox[0] - bbox[1])
        return [axis1, axis2] if axis1 <= axis2 else [axis2, axis1]

    METADATA['axes_length'] = [
        {'dict_label': 'Axes length minor', 
         'plot_label': '$a_{env}$', 
         'unit': 'DU', },
        {'dict_label': 'Axes length major', 
         'plot_label': '$b_{env}$', 
         'unit': 'DU', }
    ]

    @cached_property
    def size(self):
        alfa = self.envelope_angle
        a, b = self.axes_length
        xsize = (a * np.abs(np.cos(alfa)) + b * np.abs(np.sin(alfa)))
        ysize = (a * np.abs(np.sin(alfa)) + b * np.abs(np.cos(alfa)))
        return [xsize, ysize]
        
    METADATA['size'] = [
        {'dict_label': 'Size radial', 
         'plot_label': '$d_{rad}$', 
         'unit': 'DU', },
        {'dict_label': 'Size poloidal', 
         'plot_label': '$d_{pol}$', 
         'unit': 'DU', }
    ]

    @cached_property
    def envelope_angle(self):
        def _azimuth(point1, point2):
            angle = np.arctan2(point2[1] - point1[1], point2[0] - point1[0])
            return np.degrees(angle) if angle > 0 else np.degrees(angle) + 180

        bbox = np.asarray(self.oriented_envelope.exterior.coords)
        axis1 = np.linalg.norm(bbox[0] - bbox[3])
        axis2 = np.linalg.norm(bbox[0] - bbox[1])

        az = _azimuth(bbox[0], bbox[1]) if axis1 <= axis2 else _azimuth(bbox[0], bbox[3])
        return az / 180. * np.pi

    METADATA['envelope_angle'] = {
        'dict_label': 'Angle envelope', 
        'plot_label': r'$\theta_{env}$', 
        'unit': 'rad', 
    }

    @cached_property
    def concave_hull(self):
        return concave_hull(self.shapely_polygon)

    @cached_property
    def intensity(self):
        if self.polygon_with_data:
            return np.sum(self.data)
        raise ValueError('The polygon needs to have data to integrate the intensity')

    METADATA['intensity'] = {
        'dict_label': 'Intensity', 
        'plot_label': 'Intensity', 
        'unit': 'Digit', 
    }

    @property
    def vertices(self):
        return np.asarray([self.x, self.y]).transpose()

    @cached_property
    def path(self):
        from matplotlib.path import Path
        codes = [Path.MOVETO]
        for i_code in range(1, len(self.x)-1):
            if self.path_order == 3: codes.append(Path.CURVE4)
            elif self.path_order == 2: codes.append(Path.CURVE3)
            elif self.path_order == 1: codes.append(Path.LINETO)
            else: raise ValueError('Polygon.path_order cannot be higher than 3. Returning...')

        if self.path_order in [2, 3]: codes.append(Path.CURVE3)
        elif self.path_order == 1: codes.append(Path.LINETO)
        codes.append(Path.CLOSEPOLY)

        xy_looped = np.zeros([len(self.x)+1, 2])
        xy_looped[0:-1, :] = np.asarray([self.x, self.y]).transpose()
        xy_looped[-1, :] = [self.x[0], self.y[0]]
        return Path(xy_looped, codes)

    @cached_property
    def area(self):
        return 0.5 * np.abs(np.dot(self.x, np.roll(self.y, 1)) - np.dot(self.y, np.roll(self.x, 1)))

    METADATA['area'] = {
        'dict_label': 'Area', 
        'plot_label': 'Area', 
        'unit': 'DU^2', 
    }

    @cached_property
    def signed_area(self):
        return 0.5 * (np.dot(self.x, np.roll(self.y, 1)) - np.dot(self.y, np.roll(self.x, 1)))

    METADATA['signed_area'] = {
        'dict_label': 'Signed area', 
        'plot_label': '$Area_{signed}$', 
        'unit': 'DU^2', 
    }

    @cached_property
    def centroid(self):
        sa = self.signed_area
        if sa != 0:
            x_center = 1/(6*sa) * np.dot(self.x+np.roll(self.x, 1), self.x*np.roll(self.y, 1) - np.roll(self.x, 1)*self.y)
            y_center = 1/(6*sa) * np.dot(self.y+np.roll(self.y, 1), self.x*np.roll(self.y, 1) - np.roll(self.x, 1)*self.y)
            return [x_center, y_center]
        return [np.nan, np.nan]

    METADATA['centroid'] = [
        {'dict_label': 'Centroid radial', 
         'plot_label': 'Centr. rad.', 
         'unit': 'DU', },
        {'dict_label': 'Centroid poloidal', 
         'plot_label': 'Centr. pol.', 
         'unit': 'DU', }
    ]

    @cached_property
    def convex_hull_obj(self):
        coordinates = np.asarray([self.x, self.y]).transpose()
        try:
            hull = ConvexHull(coordinates)
            x_hull = coordinates[hull.vertices, 0]
            y_hull = coordinates[hull.vertices, 1]
        except:
            x_hull = self.x
            y_hull = self.y
        return Polygon(x=x_hull, y=y_hull, distance_unit=self._distance_unit)

    @cached_property
    def perimeter(self):
        return np.sum(np.sqrt(np.diff(self.x)**2 + np.diff(self.y)**2))

    METADATA['perimeter'] = {
        'dict_label': 'Perimeter', 
        'plot_label': 'Perimeter', 
        'unit': 'DU', 
    }

    @cached_property
    def center_of_gravity(self):
        if not self.polygon_with_data:
            raise ValueError('The polygon doesn\'t contain data.')
        x_cog = np.sum(self.x_data*self.data) / np.sum(self.data)
        y_cog = np.sum(self.y_data*self.data) / np.sum(self.data)
        return [x_cog, y_cog]

    METADATA['center_of_gravity'] = [
        {'dict_label': 'Center of gravity radial', 
         'plot_label': '$COG_{rad}$', 
         'unit': 'DU', },
        {'dict_label': 'Center of gravity poloidal', 
         'plot_label': '$COG_{pol}$', 
         'unit': 'DU', }
    ]

    @cached_property
    def second_central_moment(self):
        if not self.polygon_with_data:
            raise ValueError('The polygon doesn\'t contain data.')
        mu = np.zeros([2, 2])
        cog = self.center_of_gravity

        if not np.isnan(cog[0]):
            cx, cy = cog[0], cog[1] 
            mu[0, 0] = np.sum(self.data*(self.y_data - cy)**2)
            mu[0, 1] = -np.sum(self.data*(self.x_data - cx)*(self.y_data - cy))
            mu[1, 0] = mu[0, 1]
            mu[1, 1] = np.sum(self.data*(self.x_data - cx)**2)
        else:
            mu[:, :] = np.nan
        return mu

    @cached_property
    def principal_axes_angle(self):
        if not self.polygon_with_data:
            raise ValueError('The polygon doesn\'t have data within.')
        if not np.isnan(self.centroid[0]):
            try:
                mu = self.second_central_moment
                eigvalues, eigvectors = np.linalg.eig(mu)
                eig_ind = np.argmax(eigvalues)
                angle = np.arctan2(eigvectors[1, eig_ind], eigvectors[0, eig_ind])
                return np.arcsin(np.sin(angle))
            except:
                return np.nan
        return np.nan

    METADATA['principal_axes_angle'] = {
        'dict_label': 'Angle ALI', 
        'plot_label': r'$\theta_{ALI}$', 
        'unit': 'rad', 
    }

    @property
    def convexity(self):
        return self.convex_hull_obj.perimeter / self.perimeter

    METADATA['convexity'] = {
        'dict_label': 'Convexity', 
        'plot_label': 'Convexity', 
        'unit': '', 
    }

    @property
    def roundness(self):
        return 4 * np.pi * self.area / (self.convex_hull_obj.perimeter)**2

    METADATA['roundness'] = {
        'dict_label': 'Roundness', 
        'plot_label': 'Roundness', 
        'unit': '', 
    }

    @property
    def solidity(self):
        return self.area / self.convex_hull_obj.area

    METADATA['solidity'] = {
        'dict_label': 'Solidity', 
        'plot_label': 'Solidity', 
        'unit': '', 
    }

    @cached_property
    def _curvature(self):
        if self.x[0] == self.x[-1] and self.y[0] == self.y[-1]:
            x_l, y_l = self.x, self.y
        else:
            x_l = np.append(self.x, self.x[0])
            y_l = np.append(self.y, self.y[0])

        dsx, dsy = np.diff(x_l), np.diff(y_l)
        ds = np.sqrt(dsx**2 + dsy**2)
        ds = np.where(ds == 0, 1e-10, ds)
        Tx, Ty = dsx/ds, dsy/ds
        ds2 = 0.5 * (np.append(ds[-1], ds[:-1]) + ds)

        Hx = np.diff(np.append(Tx[-1], Tx)) / ds2
        Hy = np.diff(np.append(Ty[-1], Ty)) / ds2
        self._curvature_vector = np.asarray([Hx, Hy]).T
        return np.sqrt(Hx**2 + Hy**2)

    @property
    def total_curvature(self):
        return np.mean(np.abs(self._curvature))

    METADATA['total_curvature'] = {
        'dict_label': 'Total curvature', 
        'plot_label': r'$\kappa_{tot}$', 
        'unit': '', 
    }

    @property
    def bending_energy(self):
        return (self._curvature)**2

    @property
    def total_bending_energy(self):
        return np.mean(self.bending_energy)

    METADATA['total_bending_energy'] = {
        'dict_label': 'Total bending energy', 
        'plot_label': '$E_{bend}$', 
        'unit': '', 
    }


class FitShape:
    
    METADATA = {}
    
    def __init__(self, fitting='ellipse', x=None, y=None, x_data=None, y_data=None, data=None, 
                 method='linalg', elongation_base='size', distance_unit='m', verbose=False, test=False):
        
        self.fitting_type = fitting.lower()
        self._distance_unit = distance_unit
        self._elongation_base = elongation_base
        self._verbose = verbose
        self._test = test

        self._angle = np.nan
        self._axes_length = np.array([np.nan, np.nan])
        self._center = np.array([np.nan, np.nan])

        if self.fitting_type == 'ellipse':
            self.x = np.asarray(x, dtype=float) if x is not None else None
            self.y = np.asarray(y, dtype=float) if y is not None else None
            
            if self.x is None or self.y is None:
                raise ValueError("Ellipse fitting requires 'x' and 'y' boundary coordinates.")
            if len(self.x) != len(self.y):
                raise ValueError('The length of x and y must be the same.')

            if len(self.x) < 6:
                x_mid = (self.x + np.roll(self.x, -1)) / 2.0
                y_mid = (self.y + np.roll(self.y, -1)) / 2.0
                indices = np.arange(1, len(self.x) + 1)
                self.x = np.insert(self.x, indices, x_mid)
                self.y = np.insert(self.y, indices, y_mid)

            self._xmean, self._ymean = np.mean(self.x), np.mean(self.y)
            
            try:
                if method == 'linalg': self._fit_ellipse_linalg(self.x, self.y)
                elif method == 'skimage': self._fit_ellipse_skimage(self.x, self.y)
                elif method == 'leastsquare': self._fit_ellipse_leastsq(self.x, self.y)
                else: raise ValueError(f"Unknown ellipse method: {method}")
            except Exception as e:
                if self._verbose: print(f"Ellipse fitting failed: {e}")
                self.set_invalid()

        elif self.fitting_type == 'gaussian':
            self.x_data = np.asarray(x_data, dtype=float) if x_data is not None else None
            self.y_data = np.asarray(y_data, dtype=float) if y_data is not None else None
            self.data = np.asarray(data, dtype=float) if data is not None else None
            
            if self.x_data is None or self.y_data is None or self.data is None:
                raise ValueError("Gaussian fitting requires 'x_data', 'y_data', and 'data'.")
                
            self._fwhm_to_sigma = 2 * np.sqrt(2 * np.log(2))
            
            try:
                self._fit_gaussian(self.x_data, self.y_data, self.data)
            except Exception as e:
                if self._verbose: print(f"Gaussian fitting failed: {e}")
                self.set_invalid()
        else:
            raise ValueError(f"fitting argument must be 'ellipse' or 'gaussian', got '{fitting}'")

    def set_invalid(self):
        self._angle = np.nan
        self._axes_length = np.array([np.nan, np.nan])
        self._center = np.array([np.nan, np.nan])
        self._parameters = np.full(6, np.nan)
        self.popt = np.full(7, np.nan)

    @property
    def fit_angle(self): 
        return self._angle

    METADATA['fit_angle'] = {
        'dict_label': 'Angle fit', 
        'plot_label': r'$\phi_{fit}$', 
        'unit': 'rad', 
    }

    @property
    def fit_axes_length(self): 
        return [self._axes_length[0], self._axes_length[1]]

    METADATA['fit_axes_length'] = [
        {'dict_label': 'Axes length minor fit', 
         'plot_label': 'a', 
         'unit': 'DU', },
        {'dict_label': 'Axes length major fit', 
         'plot_label': 'b', 
         'unit': 'DU', }
    ]

    @property
    def fit_center(self): 
        return [self._center[0], self._center[1]]

    METADATA['fit_center'] = [
        {'dict_label': 'Position radial fit', 
         'plot_label': 'R', 
         'unit': 'DU', },
        {'dict_label': 'Position poloidal fit', 
         'plot_label': 'z', 
         'unit': 'DU', }
    ]

    @cached_property
    def fit_size(self):
        alfa = self.fit_angle
        a, b = self.fit_axes_length

        if np.isnan(a) or np.isnan(b) or a == 0 or b == 0:
            return [np.nan, np.nan]
            
        a0 = (np.cos(alfa)**2 / a**2) + (np.sin(alfa)**2 / b**2)
        a2 = (np.sin(alfa)**2 / a**2) + (np.cos(alfa)**2 / b**2)
        with np.errstate(invalid='ignore'):
            xsize, ysize = 2 / np.sqrt(a0), 2 / np.sqrt(a2)

        if np.isnan(xsize) or np.isnan(ysize):
            return [np.nan, np.nan]
        return [xsize, ysize]

    METADATA['fit_size'] = [
        {'dict_label': 'Size radial fit', 
         'plot_label': '$d_{rad}$', 
         'unit': 'DU', 
         },
        {'dict_label': 'Size poloidal fit', 
         'plot_label': '$d_{pol}$', 
         'unit': 'DU'
         }
    ]

    @cached_property
    def fit_elongation(self):
        if self._elongation_base == 'size':
            s1, s2 = self.fit_size
            # BUG FIX: Added np.abs() to prevent negative elongation
            return np.abs(s1 - s2) / (s1 + s2) if (s1 + s2) != 0 else np.nan
        else:
            a1, a2 = self.fit_axes_length
            # BUG FIX: Added np.abs() to prevent negative elongation
            return np.abs(a1 - a2) / (a1 + a2) if (a1 + a2) != 0 else np.nan

    METADATA['fit_elongation'] = {
        'dict_label': 'Elongation fit', 
        'plot_label': 'Elong.', 
        'unit': '', 
    }

    # ==========================================
    # INTERNAL ELLIPSE SOLVERS
    # ==========================================
    def _fit_ellipse_linalg(self, x, y):
        D1 = np.vstack([x**2, x*y, y**2]).T
        D2 = np.vstack([x, y, np.ones(len(x))]).T
        S1, S2, S3 = D1.T @ D1, D1.T @ D2, D2.T @ D2

        T = -np.linalg.solve(S3, S2.T)
        M = S1 + S2 @ T
        C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
        M = np.linalg.solve(C, M)

        eigval, eigvec = np.linalg.eig(M)
        con = 4 * eigvec[0] * eigvec[2] - eigvec[1]**2
        ak = eigvec[:, con > 0]

        self._parameters = np.concatenate((ak, T @ ak)).ravel()
        a, b, c = self._parameters[0], self._parameters[1]/2, self._parameters[2]
        d, f, g = self._parameters[3]/2, self._parameters[4]/2, self._parameters[5]
        den = b**2 - a*c
        
        if den > 0: raise ValueError('Coeffs do not represent an ellipse!')
        
        self._center = np.array([(c*d - b*f) / den, (a*f - b*d) / den])
        num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
        fac = np.sqrt((a - c)**2 + 4*b**2)
        ap, bp = np.sqrt(num / den / (fac - a - c)), np.sqrt(num / den / (-fac - a - c))
        self._axes_length = np.array([bp, ap]) if ap < bp else np.array([ap, bp])
        
        phi = np.arctan((2.*b) / (a - c)) / 2 if b != 0 else (0 if a < c else np.pi/2)
        if b != 0 and a > c: phi += np.pi/2
        if ap < bp: phi += np.pi/2
        if phi > np.pi/2:
            while phi > np.pi/2:
                phi -= np.pi
        else:
            while phi < -np.pi/2:
                phi += np.pi
        
        self._angle = phi   

    def _fit_ellipse_leastsq(self, x, y):
        aat = np.array([
            [np.sum(x**4), np.sum(2*x**3*y), np.sum(x**2*y**2), np.sum(2*x**3), np.sum(2*x**2*y)],
            [np.sum(2*x**3*y), np.sum(4*x**2*y**2), np.sum(2*x*y**3), np.sum(4*x**2*y), np.sum(4*x*y**2)],
            [np.sum(x**2*y**2), np.sum(2*x*y**3), np.sum(y**4), np.sum(2*x*y**2), np.sum(2*y**3)],
            [np.sum(2*x**3), np.sum(4*x**2*y), np.sum(2*x*y**2), np.sum(4*x**2), np.sum(4*x*y)],
            [np.sum(2*x**2*y), np.sum(4*x*y**2), np.sum(2*y**3), np.sum(4*x*y), np.sum(4*y**2)]
        ])
        coord_vec = np.array([np.sum(x**2), np.sum(2*x*y), np.sum(y**2), np.sum(2*x), np.sum(2*y)])

        self._parameters = np.linalg.solve(aat, coord_vec)
        a, b, c, d, f = self._parameters[0:5]
        den = b**2 - a*c
        self._center = np.array([(c*d - b*f) / den, (a*f - b*d) / den])
        self._angle = np.pi/2 + 0.5 * np.arctan2(2*b, a-c)
        
        nom = 2 * (a*f**2 + c*d**2 - b**2 - 2*b*d*f + a*c)
        term = np.sqrt((a-c)**2 + 4 * b**2)
        A = np.sqrt(nom / (den * (term - (a+c))))
        B = np.sqrt(nom / (den * (-term - (a+c))))
        self._axes_length = np.array([A, B])

    def _fit_ellipse_skimage(self, x, y):
        try:
            ellipse = EllipseModel()
            ellipse.estimate(np.column_stack((x.ravel(), y.ravel())))
            xc, yc, a, b, theta = ellipse.params
            self._center = np.array([xc, yc])
            if a < b:
                self._axes_length, self._angle = np.array([a, b]), theta
            else:
                self._axes_length, self._angle = np.array([b, a]), theta - np.pi/2
        except Exception:
            self.set_invalid()

    # ==========================================
    # INTERNAL GAUSSIAN SOLVER
    # ==========================================
    def _fit_gaussian(self, x, y, data):
        xdata = np.vstack((x.ravel(), y.ravel()))
        safe_sum = np.sum(data) if np.sum(data) != 0 else 1e-10

        initial_guess = [
            data.max(),
            np.sum(x * data) / safe_sum, np.sum(y * data) / safe_sum,
            (x.max() - x.min()) / 2 / self._fwhm_to_sigma,
            (y.max() - y.min()) / 2 / self._fwhm_to_sigma,
            0., np.mean(data)
        ]

        popt, _ = curve_fit(self.gaussian2D_fit_function, xdata, data, p0=initial_guess)
        popt[5] = np.arcsin(np.sin(popt[5]))
        self.popt = popt
        
        theta = self.popt[5]
        a, b = np.abs(np.array([self.popt[3], self.popt[4]]) * self._fwhm_to_sigma)

        if a < b:
            self._axes_length, self._angle = np.array([a, b]), np.arcsin(np.sin(theta))
        else:
            self._axes_length, self._angle = np.array([b, a]), np.arcsin(np.sin(theta - np.pi/2))
        self._center = np.array([self.popt[1], self.popt[2]])
    
    @staticmethod
    def gaussian2D_fit_function(coords, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
        x, y = coords
        xo, yo = float(xo), float(yo)

        a = (np.cos(theta)**2) / (2*sigma_x**2) + (np.sin(theta)**2) / (2*sigma_y**2)
        b = (np.sin(2*theta)) / (4*sigma_x**2) - (np.sin(2*theta)) / (4*sigma_y**2)
        c = (np.sin(theta)**2) / (2*sigma_x**2) + (np.cos(theta)**2) / (2*sigma_y**2)

        g = offset + amplitude * np.exp(-(a*(x-xo)**2 + 2*b*(x-xo)*(y-yo) + c*(y-yo)**2))
        return g.ravel()

        
class SamplePolygon(Polygon):
    def __init__(self):
        x_arr=np.asarray([1.4890597, 1.4890597, 1.4928097, 1.4928097, 1.4965597, 1.5003097,
                           1.5040597, 1.5078097, 1.5115597, 1.5153097, 1.5190597, 1.5265597,
                           1.5303097, 1.5340597, 1.5378097, 1.5378097, 1.5490597, 1.5490597,
                           1.5453097, 1.5490597, 1.5415597, 1.5415597, 1.5453097, 1.5453097,
                           1.5490597, 1.5565597, 1.5603097, 1.5640597, 1.5678097, 1.5678097,
                           1.5640597, 1.5603097, 1.5603097, 1.5528097, 1.5490597, 1.5415597,
                           1.5340597, 1.5303097, 1.5228097, 1.5153097, 1.5153097, 1.5115597,
                           1.5078097, 1.5003097, 1.5003097, 1.4965597, 1.4928097, 1.4928097,
                           1.4890597])
        y_arr=np.asarray([0.24304431, 0.26554431, 0.26929431, 0.28429431, 0.28804431,
                           0.28804431, 0.28429431, 0.28429431, 0.28054431, 0.28054431,
                           0.28429431, 0.28429431, 0.28804431, 0.28804431, 0.29179431,
                           0.30304431, 0.30304431, 0.29929431, 0.29554431, 0.29179431,
                           0.28429431, 0.28054431, 0.27679431, 0.26929431, 0.26554431,
                           0.26554431, 0.26929431, 0.26554431, 0.26554431, 0.25804431,
                           0.25804431, 0.25429431, 0.25054431, 0.25054431, 0.24679431,
                           0.24679431, 0.25429431, 0.25054431, 0.25054431, 0.24304431,
                           0.23929431, 0.23554431, 0.23554431, 0.22804431, 0.23179431,
                           0.23554431, 0.23554431, 0.23929431, 0.24304431])
        data_arr=np.asarray([1.11320755, 1.12121212, 1.13592233, 1.12121212, 1.12121212,
                               1.12037037, 1.11428571, 1.11627907, 1.11627907, 1.17592593,
                               1.17777778, 1.17708333, 1.17475728, 1.15384615, 1.15      ,
                               1.13761468, 1.13761468, 1.14423077, 1.14423077, 1.13541667,
                               1.10679612, 1.11627907, 1.13333333, 1.17777778, 1.2       ,
                               1.20212766, 1.20212766, 1.17708333, 1.20212766, 1.17475728,
                               1.17142857, 1.15730337, 1.15686275, 1.14953271, 1.14606742,
                               1.13541667, 1.11627907, 1.11627907, 1.17721519, 1.17777778,
                               1.2       , 1.23809524, 1.25      , 1.25301205, 1.24742268,
                               1.24742268, 1.24742268, 1.23958333, 1.18269231, 1.17346939,
                               1.15686275, 1.14583333, 1.10679612, 1.12048193, 1.13793103,
                               1.18072289, 1.22093023, 1.25      , 1.25301205, 1.26966292,
                               1.25301205, 1.28089888, 1.28571429, 1.26966292, 1.24742268,
                               1.23958333, 1.17346939, 1.14583333, 1.13157895, 1.13157895,
                               1.2       , 1.24050633, 1.25301205, 1.26966292, 1.2804878 ,
                               1.29761905, 1.28915663, 1.28915663, 1.25287356, 1.23958333,
                               1.15      , 1.14457831, 1.11111111, 1.11111111, 1.17333333,
                               1.2       , 1.24390244, 1.27272727, 1.3       , 1.32631579,
                               1.30612245, 1.29761905, 1.25287356, 1.2       , 1.15      ,
                               1.11111111, 1.13157895, 1.18072289, 1.23376623, 1.24390244,
                               1.28571429, 1.29885057, 1.29761905, 1.28915663, 1.25974026,
                               1.20512821, 1.15      , 1.15068493, 1.17333333, 1.20289855,
                               1.24390244, 1.28571429, 1.25287356, 1.25974026, 1.24      ,
                               1.2       , 1.16      , 1.11538462, 1.15068493, 1.17333333,
                               1.2       , 1.24      , 1.20833333, 1.20547945, 1.2       ,
                               1.17808219, 1.17333333, 1.14457831, 1.12676056, 1.15277778,
                               1.18181818, 1.2       , 1.2       , 1.19178082, 1.2       ,
                               1.17808219, 1.19178082, 1.16      , 1.11111111, 1.14666667,
                               1.15068493, 1.18181818, 1.1875    , 1.1875    , 1.18181818,
                               1.17808219, 1.17333333, 1.14666667, 1.11538462, 1.12676056,
                               1.13432836, 1.16176471, 1.16      , 1.16216216, 1.16216216,
                               1.16      , 1.15189873, 1.14666667, 1.12345679, 1.11111111,
                               1.13888889, 1.13888889, 1.16216216, 1.16216216, 1.16      ,
                               1.16216216, 1.15942029, 1.14666667, 1.14666667, 1.14666667,
                               1.13924051, 1.10810811, 1.10810811, 1.11764706, 1.11392405,
                               1.12676056, 1.15277778, 1.15714286, 1.18571429, 1.16      ,
                               1.14492754, 1.14864865, 1.14666667, 1.1125    , 1.12987013,
                               1.14666667, 1.13924051, 1.11111111, 1.11764706, 1.11764706,
                               1.125     , 1.15277778, 1.15277778, 1.15277778, 1.16      ,
                               1.14492754, 1.11842105, 1.10666667, 1.10958904, 1.12345679,
                               1.12345679, 1.11111111, 1.11764706, 1.11764706, 1.12328767,
                               1.13888889, 1.14492754, 1.14492754, 1.15277778, 1.10606061,
                               1.11111111, 1.11111111, 1.10810811, 1.125     , 1.14492754,
                               1.14492754, 1.14492754, 1.11842105, 1.12328767, 1.12162162,
                               1.12162162, 1.11842105, 1.10958904, 1.10666667, 1.11111111,
                               1.11594203, 1.11594203, 1.11111111, 1.10958904, 1.10666667,
                               1.11111111, 1.11111111, 1.10666667, 1.11111111, 1.11594203])

        x_data_arr=np.array([1.4890597, 1.4890597, 1.4890597, 1.4890597, 1.4890597, 1.4890597,
                              1.4890597, 1.4928097, 1.4928097, 1.4928097, 1.4928097, 1.4928097,
                              1.4928097, 1.4928097, 1.4928097, 1.4928097, 1.4928097, 1.4928097,
                              1.4928097, 1.4928097, 1.4928097, 1.4965597, 1.4965597, 1.4965597,
                              1.4965597, 1.4965597, 1.4965597, 1.4965597, 1.4965597, 1.4965597,
                              1.4965597, 1.4965597, 1.4965597, 1.4965597, 1.4965597, 1.4965597,
                              1.5003097, 1.5003097, 1.5003097, 1.5003097, 1.5003097, 1.5003097,
                              1.5003097, 1.5003097, 1.5003097, 1.5003097, 1.5003097, 1.5003097,
                              1.5003097, 1.5003097, 1.5003097, 1.5003097, 1.5003097, 1.5040597,
                              1.5040597, 1.5040597, 1.5040597, 1.5040597, 1.5040597, 1.5040597,
                              1.5040597, 1.5040597, 1.5040597, 1.5040597, 1.5040597, 1.5040597,
                              1.5040597, 1.5040597, 1.5078097, 1.5078097, 1.5078097, 1.5078097,
                              1.5078097, 1.5078097, 1.5078097, 1.5078097, 1.5078097, 1.5078097,
                              1.5078097, 1.5078097, 1.5078097, 1.5078097, 1.5115597, 1.5115597,
                              1.5115597, 1.5115597, 1.5115597, 1.5115597, 1.5115597, 1.5115597,
                              1.5115597, 1.5115597, 1.5115597, 1.5115597, 1.5115597, 1.5153097,
                              1.5153097, 1.5153097, 1.5153097, 1.5153097, 1.5153097, 1.5153097,
                              1.5153097, 1.5153097, 1.5153097, 1.5153097, 1.5153097, 1.5190597,
                              1.5190597, 1.5190597, 1.5190597, 1.5190597, 1.5190597, 1.5190597,
                              1.5190597, 1.5190597, 1.5190597, 1.5190597, 1.5228097, 1.5228097,
                              1.5228097, 1.5228097, 1.5228097, 1.5228097, 1.5228097, 1.5228097,
                              1.5228097, 1.5228097, 1.5265597, 1.5265597, 1.5265597, 1.5265597,
                              1.5265597, 1.5265597, 1.5265597, 1.5265597, 1.5265597, 1.5265597,
                              1.5303097, 1.5303097, 1.5303097, 1.5303097, 1.5303097, 1.5303097,
                              1.5303097, 1.5303097, 1.5303097, 1.5303097, 1.5303097, 1.5340597,
                              1.5340597, 1.5340597, 1.5340597, 1.5340597, 1.5340597, 1.5340597,
                              1.5340597, 1.5340597, 1.5340597, 1.5378097, 1.5378097, 1.5378097,
                              1.5378097, 1.5378097, 1.5378097, 1.5378097, 1.5378097, 1.5378097,
                              1.5378097, 1.5378097, 1.5378097, 1.5378097, 1.5378097, 1.5378097,
                              1.5415597, 1.5415597, 1.5415597, 1.5415597, 1.5415597, 1.5415597,
                              1.5415597, 1.5415597, 1.5415597, 1.5415597, 1.5415597, 1.5415597,
                              1.5415597, 1.5415597, 1.5415597, 1.5415597, 1.5453097, 1.5453097,
                              1.5453097, 1.5453097, 1.5453097, 1.5453097, 1.5453097, 1.5453097,
                              1.5453097, 1.5453097, 1.5453097, 1.5453097, 1.5453097, 1.5453097,
                              1.5490597, 1.5490597, 1.5490597, 1.5490597, 1.5490597, 1.5490597,
                              1.5490597, 1.5490597, 1.5490597, 1.5528097, 1.5528097, 1.5528097,
                              1.5528097, 1.5528097, 1.5565597, 1.5565597, 1.5565597, 1.5565597,
                              1.5565597, 1.5603097, 1.5603097, 1.5603097, 1.5603097, 1.5603097,
                              1.5603097, 1.5640597, 1.5640597, 1.5640597, 1.5678097, 1.5678097,
                              1.5678097])
        y_data_arr=np.asarray([0.24304431, 0.24679431, 0.25054431, 0.25429431, 0.25804431,
                                0.26179431, 0.26554431, 0.23554431, 0.23929431, 0.24304431,
                                0.24679431, 0.25054431, 0.25429431, 0.25804431, 0.26179431,
                                0.26554431, 0.26929431, 0.27304431, 0.27679431, 0.28054431,
                                0.28429431, 0.23554431, 0.23929431, 0.24304431, 0.24679431,
                                0.25054431, 0.25429431, 0.25804431, 0.26179431, 0.26554431,
                                0.26929431, 0.27304431, 0.27679431, 0.28054431, 0.28429431,
                                0.28804431, 0.22804431, 0.23179431, 0.23554431, 0.23929431,
                                0.24304431, 0.24679431, 0.25054431, 0.25429431, 0.25804431,
                                0.26179431, 0.26554431, 0.26929431, 0.27304431, 0.27679431,
                                0.28054431, 0.28429431, 0.28804431, 0.23179431, 0.23554431,
                                0.23929431, 0.24304431, 0.24679431, 0.25054431, 0.25429431,
                                0.25804431, 0.26179431, 0.26554431, 0.26929431, 0.27304431,
                                0.27679431, 0.28054431, 0.28429431, 0.23554431, 0.23929431,
                                0.24304431, 0.24679431, 0.25054431, 0.25429431, 0.25804431,
                                0.26179431, 0.26554431, 0.26929431, 0.27304431, 0.27679431,
                                0.28054431, 0.28429431, 0.23554431, 0.23929431, 0.24304431,
                                0.24679431, 0.25054431, 0.25429431, 0.25804431, 0.26179431,
                                0.26554431, 0.26929431, 0.27304431, 0.27679431, 0.28054431,
                                0.23929431, 0.24304431, 0.24679431, 0.25054431, 0.25429431,
                                0.25804431, 0.26179431, 0.26554431, 0.26929431, 0.27304431,
                                0.27679431, 0.28054431, 0.24679431, 0.25054431, 0.25429431,
                                0.25804431, 0.26179431, 0.26554431, 0.26929431, 0.27304431,
                                0.27679431, 0.28054431, 0.28429431, 0.25054431, 0.25429431,
                                0.25804431, 0.26179431, 0.26554431, 0.26929431, 0.27304431,
                                0.27679431, 0.28054431, 0.28429431, 0.25054431, 0.25429431,
                                0.25804431, 0.26179431, 0.26554431, 0.26929431, 0.27304431,
                                0.27679431, 0.28054431, 0.28429431, 0.25054431, 0.25429431,
                                0.25804431, 0.26179431, 0.26554431, 0.26929431, 0.27304431,
                                0.27679431, 0.28054431, 0.28429431, 0.28804431, 0.25429431,
                                0.25804431, 0.26179431, 0.26554431, 0.26929431, 0.27304431,
                                0.27679431, 0.28054431, 0.28429431, 0.28804431, 0.25054431,
                                0.25429431, 0.25804431, 0.26179431, 0.26554431, 0.26929431,
                                0.27304431, 0.27679431, 0.28054431, 0.28429431, 0.28804431,
                                0.29179431, 0.29554431, 0.29929431, 0.30304431, 0.24679431,
                                0.25054431, 0.25429431, 0.25804431, 0.26179431, 0.26554431,
                                0.26929431, 0.27304431, 0.27679431, 0.28054431, 0.28429431,
                                0.28804431, 0.29179431, 0.29554431, 0.29929431, 0.30304431,
                                0.24679431, 0.25054431, 0.25429431, 0.25804431, 0.26179431,
                                0.26554431, 0.26929431, 0.27304431, 0.27679431, 0.28804431,
                                0.29179431, 0.29554431, 0.29929431, 0.30304431, 0.24679431,
                                0.25054431, 0.25429431, 0.25804431, 0.26179431, 0.26554431,
                                0.29179431, 0.29929431, 0.30304431, 0.25054431, 0.25429431,
                                0.25804431, 0.26179431, 0.26554431, 0.25054431, 0.25429431,
                                0.25804431, 0.26179431, 0.26554431, 0.25054431, 0.25429431,
                                0.25804431, 0.26179431, 0.26554431, 0.26929431, 0.25804431,
                                0.26179431, 0.26554431, 0.25804431, 0.26179431, 0.26554431])
        
        super().__init__(
            x=x_arr, 
            y=y_arr, 
            x_data=x_data_arr, 
            y_data=y_data_arr, 
            data=data_arr,
            distance_unit='m' 
        )