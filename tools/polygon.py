#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 14:44:36 2021

@author: mlampert
"""

import numpy as np
import scipy


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

        if ((x is not None and y is not None) and  (len(x) == len(y))):
            self.x=np.asarray(x)
            self.y=np.asarray(y)
        else:
            raise ValueError('The input x and y has to be defined and must have the same length.')

        if (x_data is not None and
            y_data is not None and
            data is not None):
            if (x_data.shape != y_data.shape or
               x_data.shape != data.shape):
                raise ValueError('The shapes of the input data do not match.')

            self.x_data=x_data
            self.y_data=y_data

            self.x_data_pix=x_data_pix
            self.y_data_pix=y_data_pix

            self.data=data
            self.polygon_with_data=True
        else:
            self.x_data=None
            self.y_data=None

            self.data=None
            self.polygon_with_data=False
            self.x_data_pix=None
            self.y_data_pix=None

        self._path=None
        self.path_order=path_order
        self.test=test
        self.remove_self_intersection()

    def remove_self_intersection(self):
        from shapely.geometry import Polygon as PolygonShapely
        from shapely import concave_hull

        polygon=PolygonShapely(zip(self.x,self.y))
        polygon.is_valid
        if False:   #This
            polygon=concave_hull(polygon.buffer(0)) #Prevents self-intersection
        else:
            polygon=polygon.buffer(0)

        self._shapely_polygon=polygon
        try:
            points = []
            try:
                for poly in polygon:
                    points.extend(poly.exterior.coords[:-1])
            except:
                points.extend(polygon.exterior.coords[:-1])

            points=np.asarray(points)

            self.x = points[:,0]
            self.y = points[:,1]

            xy_looped=np.zeros([len(self.x)+1,2])
            xy_looped[0:-1,:]=np.asarray([self.x, self.y]).T
            xy_looped[-1,:]=[self.x[0], self.y[0]]

            self.x=xy_looped[:,0]
            self.y=xy_looped[:,1]

        except Exception as e:
            raise e

    def smooth(self, refinements=5): #chaikins_corner_cutting

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

    @property
    def intensity(self):
        if self.polygon_with_data:
            return np.sum(self.data)
        else:
            raise ValueError('The polygon needs to have data to integrate the intensity')
    @property
    def vertices(self):
        return np.asarray([self.x,self.y]).transpose()

    @property
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

    @property
    def shapely_polygon(self):
        try:
            return self._shapely_polygon
        except:
            self.remove_self_intersection()
            return self._shapely_polygon

    @property
    def area(self):
        """
        Returns the area of the polygon based on the so called shoelace formula.
        Sources:
            https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
            https://en.wikipedia.org/wiki/Shoelace_formula
        """
        return 0.5*np.abs(np.dot(self.x,np.roll(self.y,1)) -
                          np.dot(self.y,np.roll(self.x,1)))

    @property
    def signed_area(self):
        return 0.5*(np.dot(self.x,np.roll(self.y,1)) -
                    np.dot(self.y,np.roll(self.x,1)))

    @property
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

    @property
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
            hull = scipy.spatial.ConvexHull(coordinates)
            x_hull = coordinates[hull.vertices,0]
            y_hull = coordinates[hull.vertices,1]
        except:
            x_hull=self.x
            y_hull=self.y

        return Polygon(x=x_hull,
                       y=y_hull)
    @property
    def concave_hull(self):
        raise NotImplementedError('Do your thing...')

    @property
    def perimeter(self):
        '''
        Returns the perimeter of the polygon calculated from the
        Eucledian distance between the vertices. Assumes that the polygon's
        coordinates are in order.'

        Returns
        -------
        float
            perimeter of the polygon

        '''
        perimeter=0
        for i_dist in range(-1,len(self.x)-1):
            perimeter+=np.sqrt((self.x[i_dist]-self.x[i_dist+1])**2+
                               (self.y[i_dist]-self.y[i_dist+1])**2)

        return perimeter

    @property
    def roundness(self):
        return 4*np.pi*self.area/(self.convex_hull.perimeter)**2

    @property
    def convexity(self):
        return self.convex_hull.perimeter/self.perimeter

    @property
    def solidity(self):
        return self.area/self.convex_hull.area

    @property
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


    @property
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
                angle=np.arctan(eigvectors[1,eig_ind]/
                                eigvectors[0,eig_ind])
                return np.arcsin(np.sin(angle))
            
            #   return np.arctan2(eigvectors[1,eig_ind],
            #                     eigvectors[0,eig_ind])
            
            except:
                return np.nan
        else:
            return np.nan

    @property
    def center_of_gravity(self):
        if not self.polygon_with_data:
            raise ValueError('The polygon doesn\'t contain data. Please provide x_data, y_data and data to Polygon()')

        x_cog=np.sum(self.x_data*self.data)/np.sum(self.data)
        y_cog=np.sum(self.y_data*self.data)/np.sum(self.data)
        return np.asarray([x_cog,y_cog]).transpose()

    @property
    def curvature(self):
        """
        Returns the magnitude of the curvature vector for each vertex

        Returns
        -------
        ndarray
            DESCRIPTION.

        """
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

        if self.test:
            print('x_looped', x_looped)
            print('y_looped', y_looped)
            print('dsx', dsx)
            print('dsy', dsy)
            print('ds', ds)
            print('ds2', ds2)

        Hx=np.diff(np.append(Tx[-1],Tx))/ds2
        Hy=np.diff(np.append(Ty[-1],Ty))/ds2
        self._curvature_vector=np.asarray([Hx,Hy]).transpose()
        curvature=np.sqrt(Hx**2+Hy**2)

        if self.test:
            print('curvature', curvature)

        return curvature

    @property
    def curvature_vector(self):
        try:
            return self._curvature_vector
        except:
            self.curvature
            return self._curvature_vector

    @property
    def total_curvature(self):
        return np.mean(np.abs(self.curvature))

    @property
    def bending_energy(self):
        return (self.curvature)**2

    @property
    def total_bending_energy(self):
        return np.mean(self.bending_energy)
