#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:55:15 2026

@author: mlampert
"""

import copy

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
import matplotlib.animation as animation
import matplotlib.lines as mlines
from matplotlib.widgets import Button, Slider
from matplotlib import ticker

class PlotAnimation:
    
    def __init__(self, ax_list, axes, axes_unit_conversion, d, xdata, ydata, tdata, xdata_range, ydata_range,
                 cmap_obj, contour_levels, coord_t, coord_x,
                 coord_y, cmap, options, overplot_options, xrange, yrange,
                 zrange, image_like, plot_options, language, plot_id, gs):
   
        self.ax_list = ax_list
        self.axes = axes
        self.axes_unit_conversion=axes_unit_conversion
        self.contour_levels = contour_levels        
        self.cmap = cmap
        self.cmap_obj = cmap_obj
        self.coord_t = coord_t  
        self.coord_x = coord_x
        self.coord_y = coord_y        
        self.current_frame = 0.
        self.d = d
        self.fig = plt.figure(plot_id.figure) 
        self.gs = gs
        self.image_like = image_like
        self.language = language
        self.options = options
        self.overplot_options = overplot_options
        self.pause = False
        self.plot_id = plot_id
        self.plot_options = plot_options        
        self.speed = 40.
        
        self.tdata = tdata
        self.xdata = xdata
        self.ydata = ydata
        if xdata_range is not None:
            self.xdata_range = [self.axes_unit_conversion[0] * xdata_range[0],
                                self.axes_unit_conversion[0] * xdata_range[1]]
        else:
            self.xdata_range=None
        if ydata_range is not None:
            self.ydata_range = [self.axes_unit_conversion[1] * ydata_range[0],
                                self.axes_unit_conversion[1] * ydata_range[1]]
        else:
            self.ydata_range=None
        
        self.xrange = xrange
        self.yrange = yrange
        self.zrange = zrange

        
        if (self.contour_levels is None):
            self.contour_levels = 255
            
    def animate(self):
        
        pause_ax = plt.figure(self.plot_id.figure).add_axes((0.78, 0.94, 0.1, 0.04))
        self.pause_button = Button(pause_ax, 'Pause', hovercolor='0.975')
        self.pause_button.on_clicked(self._pause_animation)
        pause_ax._button=self.pause_button
        
        reset_ax = plt.figure(self.plot_id.figure).add_axes((0.78, 0.89, 0.1, 0.04))
        reset_button = Button(reset_ax, 'Reset', hovercolor='0.975')
        reset_button.on_clicked(self._reset_animation)
        reset_ax._button=reset_button
        
        slow_ax = plt.figure(self.plot_id.figure).add_axes((0.88, 0.94, 0.1, 0.04))
        self.slow_button = Button(slow_ax, str(int(1000./(self.speed/0.8)))+'fps', hovercolor='0.975')
        self.slow_button.on_clicked(self._slow_animation)
        slow_ax._button=self.slow_button
        
        speed_ax = plt.figure(self.plot_id.figure).add_axes((0.88, 0.89, 0.1, 0.04))
        self.speed_button = Button(speed_ax, str(int(1000./(self.speed*0.8)))+'fps', hovercolor='0.975')
        self.speed_button.on_clicked(self._speed_animation)
        speed_ax._button=self.speed_button
        
        
        slider_ax = plt.figure(self.plot_id.figure).add_axes((0.1, 0.94, 0.5, 0.04))
        self.time_slider = Slider(slider_ax, label=self.axes[2],
                                  valmin=self.tdata[0]*self.axes_unit_conversion[2], 
                                  valmax=self.tdata[-1]*self.axes_unit_conversion[2],
                                  valinit=self.tdata[0]*self.axes_unit_conversion[2])
        self.time_slider.on_changed(self._set_animation)
        
        #The following line needed to be removed for matplotlib 3.4.1
        #plt.subplot(self.plot_id.base_subplot)
        
        self.ax_act = plt.subplot(self.gs[0,0])
        if (len(self.coord_x.dimension_list) == 3 or
            len(self.coord_y.dimension_list) == 3):
            self.ax_act.set_autoscale_on(False)
            
            
#The following lines set the axes to be equal if the units of the axes-to-be-plotted are the same
        if self.options['Equal axes']:
            axes_coordinate_decrypt=[0] * len(self.axes)
            for i_axes in range(len(self.axes)):
                for j_coordinate in range(len(self.d.coordinates)):
                    if (self.d.coordinates[j_coordinate].unit.name == self.axes[i_axes]):
                        axes_coordinate_decrypt[i_axes]=j_coordinate
            for i_check in range(len(self.axes))        :
                for j_check in range(i_check+1,len(self.axes)):
                    if (self.d.coordinates[axes_coordinate_decrypt[i_check]].unit.unit ==
                        self.d.coordinates[axes_coordinate_decrypt[j_check]].unit.unit):
                        self.ax_act.set_aspect(1.0)
                    
        time_index = [slice(0,dim) for dim in self.d.data.shape]
        time_index[self.coord_t.dimension_list[0]] = 0
        time_index = tuple(time_index)
        
        act_ax_pos=self.ax_act.get_position()
        slider_ax.set_position([act_ax_pos.x0,0.94,0.5,0.04])

        if (self.zrange is None):
            self.zrange=[np.nanmin(self.d.data),
                         np.nanmax(self.d.data)]
        self.vmin = self.zrange[0]
        self.vmax = self.zrange[1]


        if (self.vmax <= self.vmin):
            raise ValueError("Invalid z range.")
            
        if (self.options['Log z']):
            if (self.vmin <= 0):
                raise ValueError("z range[0] cannot be negative or zero for logarithmic scale.")
            self.norm = colors.LogNorm(vmin=self.vmin, vmax=self.vmax)
            self.locator = ticker.LogLocator(subs='all')
            ticker.LogLocator(subs='all')
        else:
            self.norm = None
            self.locator = None
                
            
        _plot_opt = self.plot_options[0]

        if (self.image_like):
            try: 
                if (self.coord_x.dimension_list[0] < self.coord_y.dimension_list[0]):
                    im = np.clip(np.transpose(self.d.data[time_index]),self.vmin,self.vmax)
                else:
                    im = np.clip(self.d.data[time_index],self.vmin,self.vmax)
                img = plt.imshow(im,extent=self.xdata_range + self.ydata_range,norm=self.norm,
                                cmap=self.cmap_obj,vmin=self.vmin,aspect=self.options['Aspect ratio'],interpolation=self.options['Interpolation'],
                                vmax=self.vmax,origin='lower',**_plot_opt)            
                del im
            except Exception as e:
                raise e
        else:
            if (len(self.xdata.shape) == 3 and len(self.ydata.shape) == 3):
                xgrid, ygrid = grid_to_box(self.xdata[time_index]*self.axes_unit_conversion[0],
                                           self.ydata[time_index]*self.axes_unit_conversion[1])
            else:
                xgrid, ygrid = grid_to_box(self.xdata*self.axes_unit_conversion[0],
                                           self.ydata*self.axes_unit_conversion[1])
                
            im = np.clip(np.transpose(self.d.data[time_index]),self.vmin,self.vmax)
            try:
                img = plt.pcolormesh(xgrid,ygrid,im,norm=self.norm,cmap=self.cmap,vmin=self.vmin,
                                  vmax=self.vmax,**_plot_opt)
            except Exception as e:
                raise e
            del im
            
            self.xdata_range=None
            self.ydata_range=None
            if (self.xrange is None):
                self.xrange=[np.min(self.xdata),np.max(self.xdata)]
            
            plt.xlim(self.xrange[0]*self.axes_unit_conversion[0],
                     self.xrange[1]*self.axes_unit_conversion[0])
                
            if (self.yrange is None):
                self.yrange=[np.min(self.ydata),np.max(self.ydata)]
                
            plt.ylim(self.yrange[0]*self.axes_unit_conversion[1],
                     self.yrange[1]*self.axes_unit_conversion[1]) 

        if (self.options['Colorbar']):
            cbar = plt.colorbar(img,ax=self.ax_act)
            cbar.set_label(self.d.data_unit.name)
#EFIT overplot feature implementation:
            #It needs to be more generalied in the future as the coordinates are not necessarily in this order: [time_index,spat_index]
            #This needs to be cross-checked with the time array's dimensions wherever there is a call for a certain index.            

        
        if self.axes_unit_conversion[0] == 1.:
            plt.xlabel(self.ax_list[0].title(language=self.language))
        else:
            plt.xlabel(self.ax_list[0].title(language=self.language, 
                                             new_unit=self.options['Plot units'][self.axes[0]]))
            
        if self.axes_unit_conversion[1] == 1.:
            plt.ylabel(self.ax_list[1].title(language=self.language))
        else:
            plt.ylabel(self.ax_list[1].title(language=self.language, 
                                             new_unit=self.options['Plot units'][self.axes[1]]))
            
        if (self.options['Log x']):
            plt.xscale('log')
        if (self.options['Log y']):
            plt.yscale('log')
        
        if self.options['Plot units'] is not None:
            if self.axes[2] in self.options['Plot units']:
                time_unit=self.options['Plot units'][self.axes[2]]
                time_coeff=self.axes_unit_conversion[2]
            else:
                time_unit=self.coord_t.unit.unit
            time_coeff=1.
        else:
            time_unit=self.coord_t.unit.unit
            time_coeff=1.
        title = str(self.d.exp_id)+' @ '+self.coord_t.unit.name+'='+"{:10.7f}".format(self.tdata[0]*time_coeff)+\
                ' ['+time_unit+']'
                              
        plt.title(title)

        plt.show(block=False)        
        self.anim = animation.FuncAnimation(self.fig, self.animate_plot, 
                                            len(self.tdata),
                                            interval=self.speed,blit=False)
        
    def animate_plot(self, it):
        
        time_index = [slice(0,dim) for dim in self.d.data.shape]
        time_index[self.coord_t.dimension_list[0]] = it
        time_index = tuple(time_index)
        
        self.time_slider.eventson = False
        self.time_slider.set_val(self.tdata[it]*self.axes_unit_conversion[2])
        self.time_slider.eventson = True       
        
        self.current_frame = it

        plot_opt = copy.deepcopy(self.plot_options[0])
        self.ax_act.clear()
        
        if (self.image_like):
            try: 
                if (self.coord_x.dimension_list[0] < self.coord_y.dimension_list[0]):
                    im = np.clip(np.transpose(self.d.data[time_index]),self.vmin,self.vmax)
                else:
                    im = np.clip(self.d.data[time_index],self.vmin,self.vmax)
                plt.imshow(im,extent=self.xdata_range + self.ydata_range,norm=self.norm,
                           cmap=self.cmap_obj,vmin=self.vmin,
                           aspect=self.options['Aspect ratio'],
                           interpolation=self.options['Interpolation'],
                           vmax=self.vmax,origin='lower',**plot_opt)
                del im
            except Exception as e:
                raise e
        else:
            self.ax_act.set_autoscale_on(False)
            self.ax_act.set_xlim(self.xrange[0]*self.axes_unit_conversion[0],
                                 self.xrange[1]*self.axes_unit_conversion[0])  
            self.ax_act.set_ylim(self.yrange[0]*self.axes_unit_conversion[1],
                                 self.yrange[1]*self.axes_unit_conversion[1])
            if (len(self.xdata.shape) == 3 and len(self.ydata.shape) == 3):
                xgrid, ygrid = flap.tools.grid_to_box(self.xdata[time_index]*self.axes_unit_conversion[0],
                                                      self.ydata[time_index]*self.axes_unit_conversion[1]) #Same issue, time is not necessarily the first flap.coordinate.
            else:
                xgrid, ygrid = flap.tools.grid_to_box(self.xdata*self.axes_unit_conversion[0],
                                                      self.ydata*self.axes_unit_conversion[1])
            im = np.clip(np.transpose(self.d.data[time_index]),self.vmin,self.vmax)
            try:
                plt.pcolormesh(xgrid,ygrid,im,norm=self.norm,cmap=self.cmap,vmin=self.vmin,
                               vmax=self.vmax,**plot_opt)
            except Exception as e:
                raise e
            del im
        
        if (self.overplot_options is not None):
            for path_obj_keys in self.overplot_options['path']:
                if self.overplot_options['path'][path_obj_keys]['Plot']:
                    im = plt.plot(self.overplot_options['path'][path_obj_keys]['data']['Data resampled'][0,it,:]*self.axes_unit_conversion[0],
                                  self.overplot_options['path'][path_obj_keys]['data']['Data resampled'][1,it,:]*self.axes_unit_conversion[1],
                                  color=self.overplot_options['path'][path_obj_keys]['Color'])

            for contour_obj_keys in self.overplot_options['contour']:
                if self.overplot_options['contour'][contour_obj_keys]['Plot']:
                    im = plt.contour(self.overplot_options['contour'][contour_obj_keys]['data']['X coord resampled'][it,:,:].transpose()*self.axes_unit_conversion[0],
                                     self.overplot_options['contour'][contour_obj_keys]['data']['Y coord resampled'][it,:,:].transpose()*self.axes_unit_conversion[1],
                                     self.overplot_options['contour'][contour_obj_keys]['data']['Data resampled'][it,:,:],
                                     levels=self.overplot_options['contour'][contour_obj_keys]['nlevel'],
                                     cmap=self.overplot_options['contour'][contour_obj_keys]['Colormap'])
                    
            for contour_obj_keys in self.overplot_options['arrow']:
                if self.overplot_options['arrow'][contour_obj_keys]['Plot']:
                    
                    time_index = [slice(0,dim) for dim in self.overplot_options['arrow'][contour_obj_keys]['data']['Data X'].shape]
                    time_index[self.overplot_options['arrow'][contour_obj_keys]['data']['Time dimension']] = it
                    time_index = tuple(time_index)
                    
                    x_coords=self.overplot_options['arrow'][contour_obj_keys]['data']['X coord'][time_index].flatten()*self.axes_unit_conversion[0]
                    y_coords=self.overplot_options['arrow'][contour_obj_keys]['data']['Y coord'][time_index].flatten()*self.axes_unit_conversion[1]
                    data_x=self.overplot_options['arrow'][contour_obj_keys]['data']['Data X'][time_index].flatten()*self.axes_unit_conversion[0]
                    data_y=self.overplot_options['arrow'][contour_obj_keys]['data']['Data Y'][time_index].flatten()*self.axes_unit_conversion[1]
                    
                    for i_coord in range(len(x_coords)):
                        im = plt.arrow(x_coords[i_coord],
                                       y_coords[i_coord], 
                                       data_x[i_coord], 
                                       data_y[i_coord],
                                       width=self.overplot_options['arrow'][contour_obj_keys]['width'],
                                       color=self.overplot_options['arrow'][contour_obj_keys]['color'],
                                       length_includes_head=True,
                                       )              
                    
            for line_obj_keys in self.overplot_options['line']:
                xmin, xmax = self.ax_act.get_xbound()
                ymin, ymax = self.ax_act.get_ybound()
                if self.overplot_options['line'][line_obj_keys]['Plot']:
                    
                    if 'Horizontal' in self.overplot_options['line'][line_obj_keys]:
                        h_coords=self.overplot_options['line'][line_obj_keys]['Horizontal']
                        for segments in h_coords:
                            if segments[0] > ymin and segments[0] < ymax:
                                l = mlines.Line2D([xmin,xmax], [segments[0],segments[0]], color=segments[1])
                                self.ax_act.add_line(l)
                                
                    if 'Vertical' in self.overplot_options['line'][line_obj_keys]:
                        v_coords=self.overplot_options['line'][line_obj_keys]['Vertical']
                        for segments in v_coords:
                            if segments[0] > xmin and segments[0] < xmax:
                                l = mlines.Line2D([segments[0],segments[0]], [ymin,ymax], color=segments[1])
                                self.ax_act.add_line(l)

        if self.axes_unit_conversion[0] == 1.:
            plt.xlabel(self.ax_list[0].title(language=self.language))
        else:
            plt.xlabel(self.ax_list[0].title(language=self.language, new_unit=self.options['Plot units'][self.axes[0]]))
            
        if self.axes_unit_conversion[1] == 1.:
            plt.ylabel(self.ax_list[1].title(language=self.language))
        else:
            plt.ylabel(self.ax_list[1].title(language=self.language, new_unit=self.options['Plot units'][self.axes[1]]))
        
        if self.options['Plot units'] is not None:
            if self.axes[2] in self.options['Plot units']:
                time_unit=self.options['Plot units'][self.axes[2]]
                time_coeff=self.axes_unit_conversion[2]
            else:
                time_unit=self.coord_t.unit.unit
                time_coeff=1.
        else:
            time_unit=self.coord_t.unit.unit
            time_coeff=1.
            
        title = str(self.d.exp_id)+' @ '+self.coord_t.unit.name+'='+"{:10.7f}".format(self.tdata[it]*time_coeff)+\
                ' ['+time_unit+']'
        
        self.ax_act.set_title(title)
    
    def _reset_animation(self, event):
        self.anim.event_source.stop()
        self.speed = 40.
        self.anim = animation.FuncAnimation(plt.figure(self.plot_id.figure), self.animate_plot, 
                                            len(self.tdata),interval=self.speed,blit=False)
        self.anim.event_source.start()
        self.pause = False
        
    def _pause_animation(self, event):
        if self.pause:
            self.anim.event_source.start()
            self.pause = False
            self.pause_button.label.set_text("Pause")
        else:
            self.anim.event_source.stop()
            self.pause = True
            self.pause_button.label.set_text("Start")
        
    def _set_animation(self, time):
        self.anim.event_source.stop()
        frame=(np.abs(self.tdata*self.axes_unit_conversion[2]-time)).argmin()
        self.anim = animation.FuncAnimation(plt.figure(self.plot_id.figure), self.animate_plot, 
                                            frames=np.arange(frame,len(self.tdata)-1),
                                            interval=self.speed,blit=False)
        self.anim.event_source.start()
        self.pause = False
        
    def _slow_animation(self, event):
        self.anim.event_source.stop()
        self.speed=self.speed/0.8
        self.anim = animation.FuncAnimation(plt.figure(self.plot_id.figure), self.animate_plot, 
                                            frames=np.arange(self.current_frame,len(self.tdata)-1),
                                            interval=self.speed,blit=False)
        self.speed_button.label.set_text(str(int(1000./(self.speed*0.8)))+'fps')
        self.slow_button.label.set_text(str(int(1000./(self.speed/0.8)))+'fps')
        self.anim.event_source.start()
        self.pause = False
        
    def _speed_animation(self, event):  
        self.anim.event_source.stop()
        self.speed=self.speed*0.8
        self.anim = animation.FuncAnimation(plt.figure(self.plot_id.figure), self.animate_plot, 
                                            frames=np.arange(self.current_frame,len(self.tdata)-1),
                                            interval=self.speed,blit=False)
        self.speed_button.label.set_text(str(int(1000./(self.speed*0.8)))+'fps')
        self.slow_button.label.set_text(str(int(1000./(self.speed/0.8)))+'fps')
        self.anim.event_source.start()
        self.pause = False

def show_cine_video(data_object, 
                    filename=None,
                    axes=None,
                    slicing=None,
                    summing=None,
                    slicing_options=None,
                    options=None,
                    plot_type=None,
                    plot_options={},
                    plot_id=None):
    
    default_options = {'All points': False, 'Error':True, 'Y separation': None,
                       'Log x': False, 'Log y': False, 'Log z': False, 'maxpoints':4000, 'Complex mode':'Amp-phase',
                       'X range':None, 'Y range': None, 'Z range': None,'Aspect ratio':'auto',
                       'Clear':False,'Force axes':False,'Language':'EN','Maxpoints': 4000,
                       'Levels': 10, 'Colormap':None, 'Waittime':1,'Colorbar':True,'Nan color':None,
                       'Interpolation':'bilinear','Video file':None, 'Video framerate': 20,'Video format':'avi',
                       'Overplot options':None, 'Prevent saturation':False, 'Plot units':None,
                       'Equal axes':False, 'Axes visibility':[True,True],
                       }
    
    _options = flap.config.merge_options(default_options, options, data_source=data_object.data_source, section='Plot')
    
    if (plot_options is None):
        _plot_options = {}
    else:
        _plot_options = plot_options
    if (type(_plot_options) is not list):
        _plot_options = [_plot_options]


    if (type(_options['Clear']) is not bool):
        raise TypeError("Invalid type for option Clear. Should be boolean.")
    if (type(_options['Force axes']) is not bool):
        raise TypeError("Invalid type for option 'Force axes'. Should be boolean.")    
        
    if (_options['Z range'] is not None) and (_options['Prevent saturation']):
        if (_options['Z range'] is not None):
            if ((type(_options['Z range']) is not list) or (len(_options['Z range']) != 2)):
                raise ValueError("Invalid Z range setting.")
        if ((slicing is not None) or (summing is not None)):
            d = copy.deepcopy(data_object.slice_data(slicing=slicing, summing=summing, options=slicing_options))
        else:
            d = copy.deepcopy(data_object)  
        d.data=np.mod(d.data,_options['Z range'][1])
    else:
        if ((slicing is not None) or (summing is not None)):
            d = data_object.slice_data(slicing=slicing, summing=summing, options=slicing_options)
        else:
            d = data_object        
        
    # Determining a PlotID:
    # argument, actual or a new one
    if (type(plot_id) is PlotID):
        _plot_id = plot_id
    else:
        _plot_id = get_plot_id()
        if (_plot_id is None): 
            # If there is no actual plot we create a new one 
            _plot_id = PlotID()
            _plot_id.figure = plt.gcf().number
            _plot_id.base_subplot = plt.gca()
        if (_plot_id.plt_axis_list is not None):    
            if ((_plot_id.plt_axis_list[-1] != plt.gca()) or (_plot_id.figure != plt.gcf().number)):
                # If the actual subplot is not the one in the plot ID then either the subplot was
                # changed to a new one or the plot_ID changed with set_plot. 
                if (not __get_gca_invalid()):
                    # This means the plot ID was not changed, the actual plot or axis was changed.
                    # Therefore we need to use the actual values         
                    _plot_id = PlotID()
                    _plot_id.figure = plt.gcf().number
                    _plot_id.base_subplot = plt.gca()  
                    plt.cla()
    if (_options['Clear'] ):
        _plot_id = PlotID()
        if (_plot_id.figure is not None):
            plt.figure(_plot_id.figure)
        else:
            _plot_id.figure = plt.gcf().number
        if (_plot_id.base_subplot is not None):
            plt.subplot(_plot_id.base_subplot)
        else:
            _plot_id.base_subplot = plt.gca()          
        plt.cla()
            
    # Setting plot type
    known_plot_types = ['xy','scatter','multi xy', 'image', 'anim-image','contour','anim-contour','animation']
    if (plot_type is None):
        if (len(d.shape) == 1):
            _plot_type = 'xy'
        elif (len(d.shape) == 2):
            _plot_type = 'multi xy'
        elif (len(d.shape) == 3):
            _plot_type = 'anim-image'
        else:
            raise ValueError("No default plot type for this kind of data, set plot_type.")
    else:
        try:
            _plot_type = flap.tools.find_str_match(plot_type,known_plot_types)
        except TypeError:
            raise TypeError("Invalid type for plot_type. String is expected.")
        except ValueError:
            raise ValueError("Unknown plot type or too short abbreviation")    

    # Processing some options
    if ((_plot_type == 'xy') or (_plot_type == 'multi xy')):
        all_points = _options['All points']
        if (type(all_points) is not bool):
            raise TypeError("Option 'All points' should be boolean.")
    else:
        all_points = True
    plot_error = _options['Error']
    if (type(plot_error) is not bool):
        try:
            if (int(plot_error) <= 0):
                raise ValueError("Invalid number of error bars in plot (option: Error).")
            errorbars = int(plot_error)
            plot_error = True
        except:
            raise ValueError("Invalid 'Error' option.")
    else:
        errorbars = -1
    # The maximum number of points expected in the horizontal dimension of a plot
    try:
        maxpoints = int(_options['Maxpoints'])
    except ValueError:
        raise ValueError("Invalid maxpoints setting.")

    try:
        compt = flap.tools.find_str_match(_options['Complex mode'], ['Amp-phase','Real-imag'])
    except:
        raise ValueError("Invalid 'Complex mode' option:" +_options['Complex mode'])
    if (compt == 'Amp-phase'):
        comptype = 0
    elif (compt == 'Real-imag'):    
        comptype = 1
    if (_plot_id.number_of_plots > 0):
        if (_plot_id.options[-1]['Complex mode'] != _options['Complex mode']):
            raise ValueError("Different complex plot mode in overplot.")

    language = _options['Language']
    
    if (_options['Video file'] is not None):
        if ((os.sys.platform == 'darwin' or 'linux' in os.sys.platform) and 
            (options['Video format'] not in ['avi','mkv', 'mp4'])):
            raise ValueError("The chosen video format is not cupported on macOS.")
        if os.sys.platform == 'win32' and _options['Video format'] != 'avi':
            raise ValueError("The chosen video format is not cupported on Windows.")
        video_codec_decrypt={'avi':'XVID',
                             'mkv':'X264',
                             'mp4':'mp4v'}
        video_codec_code=video_codec_decrypt[_options['Video format']]
        print('Forcing waittime to be 0s for video saving.')
        _options['Waittime']=0.
            
    #These lines do the coordinate unit conversion
    axes_unit_conversion=[1.,1.,1.]
    
    
    # X range and Z range is processed here, but Y range not as it might have multiple entries for some plots
    xrange = _options['X range']
    if (xrange is not None):
        if ((type(xrange) is not list) or (len(xrange) != 2)):
            raise ValueError("Invalid X range setting.")
            
    zrange = _options['Z range']
    if (zrange is not None):
        if ((type(zrange) is not list) or (len(zrange) != 2)):
            raise ValueError("Invalid Z range setting.")
     
    cmap = _options['Colormap']
    if ((cmap is not None) and (type(cmap) is not str)):
        raise ValueError("Colormap should be a string.")
    
    contour_levels = _options['Levels']    
    # Here _plot_id is a valid (maybe empty) PlotID
    
    
    if (d.data is None):
        raise ValueError("Cannot plot DataObject without data.")
    if (len(d.shape) != 3):
        raise TypeError("Animated image plot is applicable to 3D data only. Use slicing.")
    if (d.data.dtype.kind == 'c'):
        raise TypeError("Animated image plot is applicable only to real data.")
    # Checking for numeric type
    try:
        d.data[0,0]+1
    except TypeError:
        raise TypeError("Animated image plot is applicable only to numeric data.")
    
    yrange = _options['Y range']
    if (yrange is not None):
        if ((type(yrange) is not list) or (len(yrange) != 2)):
            raise ValueError("Invalid Y range setting.")
    # Processing axes
    # Although the plot will be cleared the existing plot axes will be considered
    default_axes = [d.coordinates[0].unit.name, d.coordinates[1].unit.name,d.coordinates[2].unit.name,'__Data__']
    try:
        pdd_list, ax_list = _plot_id.check_axes(d, 
                                                axes, 
                                                clear=_options['Clear'], 
                                                default_axes=default_axes, 
                                                force=_options['Force axes'])
    except ValueError as e:
        raise e
    
    if (not ((pdd_list[3].data_type == PddType.Data) and (pdd_list[3].data_object == d))):
        raise ValueError("For the animation plot only data can be plotted on the z axis.")
    if ((pdd_list[0].data_type != PddType.Coordinate) or (pdd_list[1].data_type != PddType.Coordinate)) :
        raise ValueError("X and y coordinates of the animation plot type should be coordinates.")
    if (pdd_list[2].data_type != PddType.Coordinate) :
        raise ValueError("Time coordinate of the animation plot should be flap.coordinate.")

    coord_x = pdd_list[0].value
    coord_y = pdd_list[1].value
    coord_t = pdd_list[2].value
    
    if (len(coord_t.dimension_list) != 1):
        raise ValueError("Time coordinate for anim-image/anim-contour plot should be changing only along one dimension.")
        
    index = [0] * 3
    index[coord_t.dimension_list[0]] = ...
    tdata = coord_t.data(data_shape=d.shape,index=index)[0].flatten()
    if (not coord_y.isnumeric()):
        raise ValueError('Coordinate '+coord_y.unit.name+' is not numeric.')

    if ((coord_x.mode.equidistant) and (len(coord_x.dimension_list) == 1) and
        (coord_y.mode.equidistant) and (len(coord_y.dimension_list) == 1)):
        # This data is image-like with data points on a rectangular array
        image_like = True
    elif ((len(coord_x.dimension_list) == 1) and (len(coord_y.dimension_list) == 1)):
        if (not coord_x.isnumeric()):
            raise ValueError('Coordinate '+coord_x.unit.name+' is not numeric.')
        if (not coord_y.isnumeric()):
            raise ValueError('Coordinate '+coord_y.unit.name+' is not numeric.')
        index = [0] * len(d.shape)
        index[coord_x.dimension_list[0]] = ...
        xdata,xdata_low,xdata_high = coord_x.data(data_shape=d.shape,index=index)
        xdata = xdata.flatten()
        dx = xdata[1:] - xdata[:-1]
        index = [0] * len(d.shape)
        index[coord_y.dimension_list[0]] = ...
        ydata,ydata_low,ydata_high = coord_y.data(data_shape=d.shape,index=index)
        ydata = ydata.flatten()
        dy = ydata[1:] - ydata[:-1]
        if ((np.nonzero(np.abs(dx - dx[0]) / math.fabs(dx[0]) > 0.01)[0].size == 0) and
            (np.nonzero(np.abs(dy - dy[0]) / math.fabs(dy[0]) > 0.01)[0].size == 0)):
            # Actually the non-equidistant coordinates are equidistant
            image_like = True
        else:
            image_like = False
    else:
        image_like = False
    if (image_like):        
        xdata_range = coord_x.data_range(data_shape=d.shape)[0]   
        ydata_range = coord_y.data_range(data_shape=d.shape)[0]   
        ydata = np.squeeze(coord_y.data(data_shape=d.shape,index=index)[0])
        xdata = np.squeeze(coord_x.data(data_shape=d.shape,index=index)[0])
    else:
        index = [...]*3
        if (len(coord_x.dimension_list) < 3 and 
            len(coord_y.dimension_list) < 3):
            index[coord_t.dimension_list[0]] = 0
        xdata_range = None
        ydata_range = None
        ydata = np.squeeze(coord_y.data(data_shape=d.shape,index=index)[0])
        xdata = np.squeeze(coord_x.data(data_shape=d.shape,index=index)[0])
    try:
        cmap_obj = plt.cm.get_cmap(cmap)
        if (_options['Nan color'] is not None):
            cmap_obj.set_bad(_options['Nan color'])
    except ValueError:
        raise ValueError("Invalid color map.")
    gs = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=_plot_id.base_subplot.get_subplotspec(),hspace=0.4,wspace=0.3)

    _plot_id.plt_axis_list = []
    _plot_id.plt_axis_list.append(plt.subplot(gs[0,0]))

    oargs=(ax_list, axes, axes_unit_conversion, d, xdata, ydata, tdata, xdata_range, ydata_range,
           cmap_obj, contour_levels, 
           coord_t, coord_x, coord_y, cmap, _options, overplot_options,
           xrange, yrange, zrange, image_like, 
           _plot_options, language, _plot_id, gs)
    
    anim = PlotAnimation(*oargs)
    anim.animate()
        
    plt.show(block=False)
    
def grid_to_box(xdata,ydata):
    """
    Given 2D x and y coordinate matrices create box coordinates around the points as
    needed by matplotlib.pcolomesh.
    xdata: X coordinates. 
    ydata: Y coordinates. 
    In both arrays x direction is along first dimension, y direction along second dimension.
    Returns xbox, ybox.
    """
    xdata = np.transpose(xdata.astype(float))
    xbox_shape = list(xdata.shape)
    xbox_shape[0] += 1
    xbox_shape[1] += 1
    xbox = np.empty(tuple(xbox_shape),dtype=xdata.dtype)
    xbox[1:,1:-1] = (xdata[:,:-1] + xdata[:,1:]) / 2 
    xbox[1:-1,1:-1] = (xbox[2:,1:-1] + xbox[1:-1,1:-1]) / 2
    xbox[1:-1,0] = ((xdata[1:,0] + xdata[:-1,0]) / 2 - xbox[1:-1,1]) * 2 + xbox[1:-1,1]
    xbox[1:-1,-1] = ((xdata[1:,-1] + xdata[:-1,-1]) / 2 - xbox[1:-1,-2]) * 2 + xbox[1:-1,-2]
    xbox[0,1:-1] = ((xdata[0,:-1] + xdata[0,1:]) / 2 - xbox[1,1:-1]) * 2 + xbox[1,1:-1]
    xbox[-1,1:-1] = ((xdata[-1,:-1] + xdata[-1,1:]) / 2 - xbox[-2,1:-1]) * 2 + xbox[-2,1:-1]
    xbox[0,0] = xbox[1,1] + (xbox[0,1] - xbox[1,1]) + (xbox[1,0] - xbox[1,1])
    xbox[-1,-1] = xbox[-2,-2] + (xbox[-1,-2] - xbox[-2,-2]) + (xbox[-2,-1] - xbox[-2,-2])
    xbox[0,-1] = xbox[1,-2] + (xbox[0,-2] - xbox[1,-2]) + (xbox[1,-1] - xbox[1,-2]) 
    xbox[-1,0] = xbox[-2,1] + (xbox[-1,1] - xbox[-2,1]) + (xbox[-2,0] - xbox[-2,1])
      
    ydata = np.transpose(ydata.astype(float))
    ybox_shape = list(ydata.shape)
    ybox_shape[0] += 1
    ybox_shape[1] += 1
    ybox = np.empty(tuple(ybox_shape),dtype=ydata.dtype)
    ybox[1:,1:-1] = (ydata[:,:-1] + ydata[:,1:]) / 2 
    ybox[1:-1,1:-1] = (ybox[2:,1:-1] + ybox[1:-1,1:-1]) / 2
    ybox[1:-1,0] = ((ydata[1:,0] + ydata[:-1,0]) / 2 - ybox[1:-1,1]) * 2 + ybox[1:-1,1]
    ybox[1:-1,-1] = ((ydata[1:,-1] + ydata[:-1,-1]) / 2 - ybox[1:-1,-2]) * 2 + ybox[1:-1,-2]
    ybox[0,1:-1] = ((ydata[0,:-1] + ydata[0,1:]) / 2 - ybox[1,1:-1]) * 2 + ybox[1,1:-1]
    ybox[-1,1:-1] = ((ydata[-1,:-1] + ydata[-1,1:]) / 2 - ybox[-2,1:-1]) * 2 + ybox[-2,1:-1]
    ybox[0,0] = ybox[1,1] + (ybox[0,1] - ybox[1,1]) + (ybox[1,0] - ybox[1,1])
    ybox[-1,-1] = ybox[-2,-2] + (ybox[-1,-2] - ybox[-2,-2]) + (ybox[-2,-1] - ybox[-2,-2])
    ybox[0,-1] = ybox[1,-2] + (ybox[0,-2] - ybox[1,-2]) + (ybox[1,-1] - ybox[1,-2]) 
    ybox[-1,0] = ybox[-2,1] + (ybox[-1,1] - ybox[-2,1]) + (ybox[-2,0] - ybox[-2,1])
    
    return xbox,ybox