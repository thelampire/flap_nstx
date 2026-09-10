#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 14:44:36 2021

@author: mlampert
"""

import numpy as np
from scipy.spatial import ConvexHull

from functools import cached_property

from shapely.geometry import Polygon as PolygonShapely
from shapely import concave_hull

from dataclasses import dataclass
from typing import Any

class Polygon:

    def __init__(self,
                 x=None,                                                        #x coord for defining polygon in DEVICE coordinates
                 y=None,                                                        #y coord for defining polygon in DEVICE coordinates

                 x_data=None,                                                   #x coordinates of pixels enclosed by the polygon in DEVICE coordinates
                 y_data=None,                                                   #y coordinates of pixels enclosed by the polygon in DEVICE coordinates

                 x_data_pix=None,                                               #x coordinates of pixels enclosed by the polygon in PIXEL coordinates
                 y_data_pix=None,                                               #y coordinates of pixels enclosed by the polygon in PIXEL coordinates

                 data=None,                                                     #Intensity data enclosed by the polygon

                 path_order=3,                                                  #Order of the polygon path fit onto the vertices defined by x,y

                 test=False,                                                    #Run test procedures
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
        self.polygon_with_data=False
        
        if (x_data is not None and
            y_data is not None and
            data is not None):
            if (x_data.shape != y_data.shape or
               x_data.shape != data.shape):
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
        self._create_valid_polygon()
        self._calculate_curvature_vector()
        
    def _create_valid_polygon(self):
        
        poly = PolygonShapely(zip(self.x, self.y))
        
        if not poly.is_valid:
            poly = poly.buffer(0)
            
            # If buffer(0) splits a pinched figure-8 polygon into multiple pieces, 
            # we grab the largest continuous piece to be our main polygon.
            if poly.geom_type == 'MultiPolygon':
                poly = max(poly.geoms, key=lambda p: p.area)
            elif poly.geom_type != 'Polygon':
                raise ValueError(f"Geometry resolution failed, returned {poly.geom_type}")

        self._shapely_polygon = poly
        
        # Shapely's exterior.coords automatically closes the loop (first point == last point).
        # We can extract them directly without manually rebuilding the array!
        coords = np.array(poly.exterior.coords)

        self.x = coords[:, 0]
        self.y = coords[:, 1]

    def _calculate_curvature_vector(self):
        """
        Returns the magnitude of the curvature vector for each vertex

        Returns
        -------
        ndarray
            DESCRIPTION.

        """
        # dsx = np.diff(self.x)
        # dsy = np.diff(self.y)
        # ds = np.sqrt(dsx**2+dsy**2)
        
        # # SAFETY NET: Replace exact zeros with a tiny number to prevent NaN crashes
        # # ds = np.where(ds == 0, 1e-10, ds)

        # Tx = dsx/ds
        # Ty = dsy/ds
        #TODO: this dies unfortunately, needs to be fixed.
        if self.x[0] == self.x[-1] and self.y[0] == self.y[-1]:
            x_looped=self.x
            y_looped=self.y
        
        else:
            x_looped=np.append(self.x,self.x[0])
            y_looped=np.append(self.y,self.y[0])
        
        dsx=np.diff(x_looped)
        dsy=np.diff(y_looped)
        ds=np.sqrt(dsx**2+dsy**2)
        Tx=dsx/ds
        Ty=dsy/ds
        ds2=0.5*(np.append(ds[-1],ds[:-1])+ds)

        ds2 = 0.5*(np.append(ds[-1],ds[:-1])+ds)

        Hx = np.diff(np.append(Tx[-1],Tx))/ds2
        Hy = np.diff(np.append(Ty[-1],Ty))/ds2
        
        self._curvature_vector  =  np.asarray([Hx,Hy]).T
    
    @property
    def shapely_polygon(self):  #initialized by remove_self_intersection
        return self._shapely_polygon
    
    @cached_property
    def oriented_envelope(self):    #return oriented envelope normalized in a way that the enveloping rectangle is starting with the lower left point.
        return self._shapely_polygon.oriented_envelope.normalize()
    
    @cached_property
    def axes_length(self):
        
        bbox = np.asarray(self.oriented_envelope.exterior.coords)
        axis1 = np.linalg.norm(bbox[0] - bbox[3])
        axis2 = np.linalg.norm(bbox[0] - bbox[1])
    
        if axis1 <= axis2: #Ellipse is also [minor, major]
            return np.array([axis1, axis2])
        else:
            return np.array([axis2, axis1])
        
    @cached_property
    def size(self):
        # Calculate the horizontal and vertical spans of the rotated rectangular envelope        
        alfa=self.envelope_angle
        a=self.axes_length[0]
        b=self.axes_length[1]

        xsize = (a * np.abs(np.cos(alfa)) + b * np.abs(np.sin(alfa)))
        ysize = (a * np.abs(np.sin(alfa)) + b * np.abs(np.cos(alfa)))

        return np.array([xsize,ysize])
    
    @cached_property
    def envelope_angle(self):
        def _azimuth(point1, point2):
            """azimuth between 2 points (interval 0 - 180)"""
            angle = np.arctan2(point2[1] - point1[1], point2[0] - point1[0])
            return np.degrees(angle) if angle > 0 else np.degrees(angle) + 180
        
        def azimuth_angle(mrr): #angle of the longer side of the rectangle w.r.t. horizontal
            """azimuth of minimum_rotated_rectangle"""
            bbox = np.asarray(mrr.exterior.coords)
            axis1 = np.linalg.norm(bbox[0] - bbox[3])
            axis2 = np.linalg.norm(bbox[0] - bbox[1])
        
            if axis1 <= axis2:
                az = _azimuth(bbox[0], bbox[1])
            else:
                az = _azimuth(bbox[0], bbox[3])
        
            return az/180.*np.pi
        
        return azimuth_angle(self.oriented_envelope)
    
    @cached_property
    def concave_hull(self):
        return concave_hull(self.shapely_polygon)
    

    def smooth(self, refinements=5): #chaikins_corner_cutting
        #This does not really mooothes the polygon.
        coords = np.array([self.x,self.y]).T

        for _ in range(refinements):
            L = coords.repeat(2, axis=0)
            R = np.empty_like(L)
            R[0] = L[0]
            R[2::2] = L[1:-1:2]
            R[1:-1:2] = L[2::2]
            R[-1] = L[-1]
            coords = L * 0.75 + R * 0.25

        self.x=coords[:,0].copy()
        self.y=coords[:,1].copy()


    @cached_property
    def intensity(self):
        if self.polygon_with_data:
            return np.sum(self.data)
        else:
            raise ValueError('The polygon needs to have data to integrate the intensity')
    @property
    def vertices(self):
        return np.asarray([self.x,self.y]).transpose()

    @cached_property
    def path(self):
        """
        Returns
        -------
        matplotlib Path of the polygon. Has built in methods for intersection, contain etc. See
        matplotlib documentation.

        """
        from matplotlib.path import Path
        codes=[Path.MOVETO]
        for i_code in range(1,len(self.x)-1):
            if self.path_order == 3:
                codes.append(Path.CURVE4)
            elif self.path_order == 2:
                codes.append(Path.CURVE3)
            elif self.path_order == 1:
                codes.append(Path.LINETO)
            else:
                raise ValueError('Polygon.path_order cannot be higher than 3. Returning...')

        if self.path_order == 3 or self.path_order == 2:
            codes.append(Path.CURVE3)
        elif self.path_order == 1:
            codes.append(Path.LINETO)

        codes.append(Path.CLOSEPOLY)

        xy_looped=np.zeros([len(self.x)+1,2])
        xy_looped[0:-1,:]=np.asarray([self.x, self.y]).transpose()
        xy_looped[-1,:]=[self.x[0], self.y[0]]

        return Path(xy_looped,codes)

    @cached_property
    def area(self):
        """
        Returns the area of the polygon based on the so called shoelace formula.
        Sources:
            https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
            https://en.wikipedia.org/wiki/Shoelace_formula
        """
        return 0.5*np.abs(np.dot(self.x,np.roll(self.y,1)) -
                          np.dot(self.y,np.roll(self.x,1)))

    @cached_property
    def signed_area(self):
        return 0.5*(np.dot(self.x,np.roll(self.y,1)) -
                    np.dot(self.y,np.roll(self.x,1)))

    @cached_property
    def centroid(self):
        """
        Source: https://en.wikipedia.org/wiki/Polygon
        Coding based on the area's convention.
        """

        if self.test:
            print('area', self.signed_area)
            print('x',self.x)
            print('y', self.y)

        if self.signed_area != 0:
            x_center=1/(6*self.signed_area) * np.dot(self.x+np.roll(self.x,1),
                                                     self.x*np.roll(self.y,1) -
                                                     np.roll(self.x,1)*self.y)
            y_center=1/(6*self.signed_area) * np.dot(self.y+np.roll(self.y,1),
                                                     self.x*np.roll(self.y,1) -
                                                     np.roll(self.x,1)*self.y)
        else:
            x_center=np.nan
            y_center=np.nan

        if self.test:
            print('centroid', [x_center,y_center])

        return np.asarray([x_center,y_center])

    @cached_property
    def convex_hull(self):
        '''
        Returns the convex hull of the polygon as [n_point,2] ndarray

        Returns
        -------
        ndarray
            CONVEX HULL COORDINATES OF THE INPUT POLYGON.

        '''
        coordinates=np.asarray([self.x,self.y]).transpose()

        try:
            hull = ConvexHull(coordinates)
            x_hull = coordinates[hull.vertices,0]
            y_hull = coordinates[hull.vertices,1]
        except:
            x_hull=self.x
            y_hull=self.y

        return Polygon(x=x_hull,
                       y=y_hull)
    

    @cached_property
    def perimeter(self):
        return np.sum(np.sqrt(np.diff(self.x)**2 + np.diff(self.y)**2))


    @cached_property
    def center_of_gravity(self):
        if not self.polygon_with_data:
            raise ValueError('The polygon doesn\'t contain data. Please provide x_data, y_data and data to Polygon()')

        x_cog=np.sum(self.x_data*self.data)/np.sum(self.data)
        y_cog=np.sum(self.y_data*self.data)/np.sum(self.data)
        return np.asarray([x_cog,y_cog]).transpose()

    @cached_property
    def second_central_moment(self):
        if not self.polygon_with_data:
            raise ValueError('The polygon doesn\'t contain data. Please provide x_data, y_data and data to Polygon()')
        mu=np.zeros([2,2])
        cog=self.center_of_gravity

        if not np.isnan(cog[0]):
            mu[0,0]=np.sum(self.data*(self.y_data-cog[1])**2)
            mu[0,1]=-np.sum(self.data*(self.x_data-cog[0])*(self.y_data-cog[1]))
            mu[1,0]=mu[0,1]
            mu[1,1]=np.sum(self.data*(self.x_data-cog[0])**2)

        else:
            mu[:,:]=np.nan

        if self.test:
            print('mu',mu)
            print('data',self.data)
            print('x_data',self.x_data)
            print('y_data',self.x_data)
            print('cog',self.centroid)
        return mu


    @cached_property
    def principal_axes_angle(self):
        if not self.polygon_with_data:
            raise ValueError('The polygon doesn\'t have data within, please add data and x_data,y_data coordinates. Returning...')

        if self.test:
            print('centroid', self.centroid)
            print('central moment',self.second_central_moment)

        if not np.isnan(self.centroid[0]):
            try:
                mu=self.second_central_moment
                eigvalues,eigvectors=np.linalg.eig(mu)
                eig_ind=np.argmax(eigvalues)
                
                
                angle=np.arctan2(eigvectors[1,eig_ind],
                                 eigvectors[0,eig_ind])
                # The eigenvector sign is arbitrary, so the axis direction is pi
                # periodic and needs to be wrapped modularly.
                return np.mod(angle + np.pi/2, np.pi) - np.pi/2
            
            #   return np.arctan2(eigvectors[1,eig_ind],
            #                     eigvectors[0,eig_ind])
            
            except:
                return np.nan
        else:
            return np.nan

    @cached_property
    def curvature(self):
        Hx,Hy = self._curvature_vector.T
        curvature=np.sqrt(Hx**2 + Hy**2)

        if self.test:
            print('curvature', curvature)

        return curvature

    @property
    def curvature_vector(self):
        return self._curvature_vector

    #Shape descriptors

    @property
    def convexity(self):
        return self.convex_hull.perimeter/self.perimeter

    @property
    def roundness(self):
        return 4*np.pi*self.area/(self.convex_hull.perimeter)**2

    @property
    def solidity(self):
        return self.area/self.convex_hull.area

    @property
    def total_curvature(self):
        return np.mean(np.abs(self.curvature))

    @property
    def bending_energy(self):
        return (self.curvature)**2

    @property
    def total_bending_energy(self):
        return np.mean(self.bending_energy)