import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
from random import random
import matplotlib
from netCDF4 import Dataset
import sys
from slopes_and_binning import *

import pandas as pd

def add_buffer(A,n_extra):
    """Adds n_extra cells/columns in x and y direction to array A. Works with 2d and 3d arrays, super advanced stuff right here. """
    if A.ndim == 2:
        A_extra = np.vstack([A[-n_extra:,:],A,A[:n_extra,:]])
        A_extra = np.hstack([A_extra[:,-n_extra:],A_extra,A_extra[:,:n_extra]])
    if A.ndim == 3:
        A_extra = np.concatenate((A[:,-n_extra:,:],A,A[:,:n_extra,:]),axis=1)
        A_extra = np.concatenate((A_extra[:,:,-n_extra:],A_extra,A_extra[:,:,:n_extra]),axis=2)
        
    
    return A_extra
 
def cluster_2D(A,buffer_size):
    """
    Written by Lennéa Hayo, 2019
    Updated By Till Vondenhoff, 20-04-07
    
    Creates Buffer around A to compensate periodic boundary effects
    
    Parameters:
        A:                    bitmap of cloud area -> 1=cloud, 0=no cloud
        buffer_size:          size of the surrounding buffer as a dercentage of the size of A
                              (should be bigger than the largest occuring cloud)
        
    Returns:
        labeled_clouds_clean: labeled clouds with the integer values sorted according to their frequency, starting from 0 upwards
        A_buf:                bitmap of cloud area with added buffer on all sides
        n_buffer:             number of pixels reserved for the buffer
        cloud_center:         center of each cloud (bitmap)
        cloud_pixels:          array with corresponding number of pixels for each cloud
    """
    cloud_center_x = []
    cloud_center_y = []
    cloud_pixels = []
    
    #Uses a default periodic boundary domain
    n_max = A.shape[0]
    
    n_buffer = int(buffer_size*n_max)

    #Explanding c and w fields with a buffer on each edge to take periodic boundaries into account. 
    A_buf=add_buffer(A,n_buffer)
    
    #This is already very impressive, ndi.label detects all areas with marker =1 that are connected and gives each resulting cluster an individual integer value 
    labeled_clouds,n_clouds  = ndi.label(A_buf)
    labels = np.arange(1, n_clouds + 1)

    #fancy quick sorting. 
    unique_labels, unique_label_counts = np.unique(labeled_clouds,return_counts=True)
    lin_idx       = np.argsort(labeled_clouds.ravel(), kind='mergesort')
    lin_idx_split = np.split(lin_idx, np.cumsum(np.bincount(labeled_clouds.ravel())[:-1]))
    
    for c in range(1,n_clouds+1):
        idx_x,idx_y = np.unravel_index(lin_idx_split[c],labeled_clouds.shape)

        #idx_x,idx_y = np.where(labeled_clouds==c)
        idx_x_m = np.mean(idx_x)
        idx_y_m = np.mean(idx_y)

        if idx_x_m< n_buffer or idx_x_m>n_buffer+n_max-1 or idx_y_m< n_buffer or idx_y_m>n_buffer+n_max-1:
            #cluster is outside, chuck it
            #print(c,'cluster out of bounds',idx_x,idx_y)
            #segmentation_cp[segmentation==c] = 0
            bla = 1

        else:
            cloud_center_x.append(int(idx_x_m)-n_buffer)
            cloud_center_y.append(int(idx_y_m)-n_buffer)
            if (len(idx_x) == len(idx_y)):
                cloud_pixels.append(len(idx_x))
            else:
                print ('da läuft was schief')
            idx_x_max = np.max(idx_x)
            idx_x_min = np.min(idx_x)
            idx_y_min = np.min(idx_y)
            idx_y_max = np.max(idx_y)
            if idx_x_min< n_buffer or idx_x_max>n_buffer+n_max or idx_y_min< n_buffer or idx_y_max>n_buffer+n_max:
                #print(c,'this is our guniea pig')
                if idx_x_min<n_buffer:
                    idx_x_sel = idx_x[idx_x<n_buffer]+n_max
                    idx_y_sel = idx_y[idx_x<n_buffer]
                    labeled_clouds[idx_x_sel,idx_y_sel] = c
                if idx_x_max>=n_buffer+n_max:
                    idx_x_sel = idx_x[idx_x>=n_buffer+n_max]-n_max
                    idx_y_sel = idx_y[idx_x>=n_buffer+n_max]
                    labeled_clouds[idx_x_sel,idx_y_sel] = c
                if idx_y_min<n_buffer:
                    idx_x_sel = idx_x[idx_y<n_buffer]
                    idx_y_sel = idx_y[idx_y<n_buffer]+n_max
                    labeled_clouds[idx_x_sel,idx_y_sel] = c
                if idx_y_max>=n_buffer+n_max:
                    idx_x_sel = idx_x[idx_y>=n_buffer+n_max]
                    idx_y_sel = idx_y[idx_y>=n_buffer+n_max]-n_max
                    labeled_clouds[idx_x_sel,idx_y_sel] = c

    #Now cut to the original domain
    labeled_clouds_orig = labeled_clouds[n_buffer:-n_buffer,n_buffer:-n_buffer]
    
    #And to clean up the missing labels 
    def sort_and_tidy_labels_2D(segmentation):
        """
        For a given 2D integer array sort_and_tidy_labels will renumber the array 
        so no gaps are between the the integer values and replace them beginning with 0 upward. 
        Also, the integer values will be sorted according to their frequency.
        https://www.unidata.ucar.edu/software/netcdf/inefficient
        1D example: 
        [4,4,1,4,1,4,4,3,3,3,3,4,4]
        -> 
        [0,0,2,0,2,0,0,1,1,1,1,0,0]
        """
       
        unique_labels, unique_label_counts = np.unique(segmentation,return_counts=True)
        n_labels = len(unique_labels)
        unique_labels_sorted = [x for _,x in sorted(zip(unique_label_counts,unique_labels))][::-1]
        new_labels = np.arange(n_labels)
       
        lin_idx       = np.argsort(segmentation.ravel(), kind='mergesort')
        lin_idx_split = np.split(lin_idx, np.cumsum(np.bincount(segmentation.ravel())[:-1]))
        #think that I can now remove lin_idx, as it is an array with the size of the full domain. 
        del(lin_idx)
        
        for l in range(n_labels):
            c = unique_labels[l]
            
            idx_x,idx_y = np.unravel_index(lin_idx_split[c],segmentation.shape)
            segmentation[idx_x,idx_y] = new_labels[l]
       
        return segmentation 

    labeled_clouds_clean = sort_and_tidy_labels_2D(labeled_clouds_orig)

    return labeled_clouds_clean, A_buf, n_buffer, cloud_center_x, cloud_center_y, cloud_pixels


def plot_daily_slope(time, n_clouds, time_labels, slope_log, slope_cum, n_cloud_min):
    fig        = plt.figure(figsize=(10,6))
    axes_left  = fig.add_subplot(1, 1, 1)
    axes_right = axes_left.twinx()
    
    axes_left.bar(time, n_clouds, color='lightgrey')
    axes_left.axhline(y=n_cloud_min, color='grey', linestyle='--')
    axes_left.text(0, n_cloud_min-50, "n_cloud_min")
    axes_left.set_xlabel('Time')
    axes_left.set_ylabel('Number of clouds')

    axes_right.plot(time_labels, slope_log, linewidth=3, color='tab:pink', label='log slope')
    axes_right.plot(time_labels, slope_cum, linewidth=3, color='tab:green', label='cum slope')
    axes_right.set_ylabel('Slope')
    axes_right.set_ylim(-3,-1)
    axes_right.legend()
    
    fig.suptitle('cloud size distribution slope', fontsize=16)
    fig.autofmt_xdate()
    
    return fig

def plot_cloud_slope(data,time,timestep,bin_min,bin_max,n_bins):
    """
    Written by Lennéa Hayo, 19-11-28
    
    Creates a plot with slopes of cloud size distribution. Can be used either for one specific timestep or for a series of timesteps.
    
    Parameters:
        data: distribution of logarithmic data (here cloud sizes)
        time: at this time the slope is being calculated
        timestep: if it is a single timestep: False, if it is a time series: True
        bin_min: value of the first bin
        bin_max: value of the last bin
        n_bins: number of bins
        
    """
    if not (timestep):
        l2D = data[time,:,:]
        
        l2D_bi = np.zeros_like(l2D).astype(int)

        l2D_bi[l2D>1e-6]=1
        labeled_clouds = cluster_2D(l2D_bi,buffer_size=20)
        
        n_clouds = ndi.label(l2D_bi)
        
        #Grosse jeder wolken
        label, cl_pixels = np.unique(labeled_clouds.ravel(),return_counts=True)

        cl_size = np.sqrt(cl_pixels)*25.
        bins_log_mm, ind, CSD = log_binner_minmax(cl_size[1:], bin_min, bin_max, n_bins)
        x_bins_log_mm = bins_log_mm[:-1]/2.+bins_log_mm[1:]/2.

        pos_min = 0
        pos_max = -1

        if np.isnan(np.sum(CSD)):
            nan_pos = [0]
            for i in range(CSD.size):
                if np.isnan(CSD[i]):
                    nan_pos.append(i)
            nan_pos.append(CSD.size)
            nan_pos = np.asarray(nan_pos)

            pos_min = nan_pos[np.argmax(nan_pos[1:]-nan_pos[:-1])]+1
            pos_max = nan_pos[np.argmax(nan_pos[1:]-nan_pos[:-1])+1]-1

            print('nan_pos:',nan_pos)

            m2, b2 = np.polyfit(np.log(x_bins_log_mm[pos_min:pos_max]),np.log(CSD[pos_min:pos_max]), 1)
            f2 = 10 ** (m2 * np.log10(x_bins_log_mm)) * np.exp(b2)
        else:
            m2, b2 = np.polyfit(np.log(x_bins_log_mm),np.log(CSD), 1)
            f2 = 10 ** (m2 * np.log10(x_bins_log_mm)) * np.exp(b2)

        plt.plot(x_bins_log_mm,CSD,'-o')
        plt.plot(x_bins_log_mm,f2)
        plt.xscale('log')
        plt.yscale('log')
        plt.show
    else:
        time_unraveled = time.ravel()
        cloud_number = []
        slope = []
        time_nozeros = []
        time_hack = np.arange(time.size)/2.+6.
        for k in range(time.size):
            l2D = data[k,:,:]
            l2D_bi = np.zeros_like(l2D).astype(int)

            l2D_bi[l2D>1e-6]=1
            labeled_clouds = cluster_2D(l2D_bi,buffer_size=20)
            
            labeled_clouds, n_clouds = ndi.label(l2D_bi)
            cloud_number.append(n_clouds)
            
            #Grosse jeder wolken
            label, cl_pixels = np.unique(labeled_clouds.ravel(),return_counts=True)

            if (len(cl_pixels)>1):
                cl_size = np.sqrt(cl_pixels)*25.
                bins_log_mm, ind, CSD = log_binner_minmax(cl_size[1:],1,300,100)
                x_bins_log_mm = bins_log_mm[:-1]/2.+bins_log_mm[1:]/2.

                pos_min = 0
                pos_max = -1

                if np.isnan(np.sum(CSD)):
                    nan_pos = [0]
                    for i in range(CSD.size):
                        if np.isnan(CSD[i]):
                            nan_pos.append(i)
                    nan_pos.append(CSD.size)
                    nan_pos = np.asarray(nan_pos)

                    pos_min = nan_pos[np.argmax(nan_pos[1:]-nan_pos[:-1])]+1
                    pos_max = nan_pos[np.argmax(nan_pos[1:]-nan_pos[:-1])+1]-1

                    print('nan_pos:',nan_pos)

                    m2, b2 = np.polyfit(np.log(x_bins_log_mm[pos_min:pos_max]),np.log(CSD[pos_min:pos_max]), 1)
                else:
                    m2, b2 = np.polyfit(np.log(x_bins_log_mm),np.log(CSD), 1)
                    
                slope.append(m2)
                time_nozeros.append(time_hack[k])
            else:
                k = k+1
        
        print ('Numbers of clouds:', cloud_number)
        plt.plot(time_nozeros, slope)
        
        
def plot_plumes_slope(area,time,bin_min,bin_max,bin_n,prop_plumes,series=True,timestep=None):
    """
    Written by Lennéa Hayo, 19-11-28
    
    Creates a plot with slopes of plume size distribution. Can be used either for one specific timestep or for a series of timesteps.
    
    Parameters:
        area: distribution of logarithmic data (here plume sizes)
        time: at this time the slope is being calculated
        bin_min: value of the first bin
        bin_max: value of the last bin
        bin_n: number of bins
        prop_plumes: plumes from data_file
        series: if it is a single timestep: False, if it is a time series: True
        timestep: specific moment in time to plot dist and slope, is only used for series=False
        
    """
    plumes_time_area = prop_plumes[[time,area]]
    del(prop_plumes)
    plumes_time_area = plumes_time_area.loc[plumes_time_area[area]<25600]
    plume_times = np.unique(plumes_time_area[time])
    
    if not series:
        if timestep is None:
            raise ValueError("timestep must be set when series is False")
        #calculate slope of plumes at spezific timestep
        plume_time_area_timestep = plumes_time_area.loc[plumes_time_area[time]==plume_times[timestep-1]]
        bins_log_mm, ind, CSD = log_binner_minmax(plume_time_area_timestep[area],bin_min,bin_max,bin_n)
        x_bins_log_mm = bins_log_mm[:-1]/2.+bins_log_mm[1:]/2.

        pos_min = 0
        pos_max = -1

        if np.isnan(np.sum(CSD)):
            nan_pos = [0]
            for i in range(CSD.size):
                if np.isnan(CSD[i]):
                    nan_pos.append(i)
            nan_pos.append(CSD.size)
            nan_pos = np.asarray(nan_pos)

            pos_min = nan_pos[np.argmax(nan_pos[1:]-nan_pos[:-1])]+1
            pos_max = nan_pos[np.argmax(nan_pos[1:]-nan_pos[:-1])+1]-1

            print('nan_pos:',nan_pos)

            m2, b2 = np.polyfit(np.log(x_bins_log_mm[pos_min:pos_max]),np.log(CSD[pos_min:pos_max]), 1)
            f2 = 10 ** (m2 * np.log10(x_bins_log_mm)) * np.exp(b2)
        else:
            m2, b2 = np.polyfit(np.log(x_bins_log_mm),np.log(CSD), 1)
            f2 = 10 ** (m2 * np.log10(x_bins_log_mm)) * np.exp(b2)
            
        plt.plot(x_bins_log_mm,CSD,'-o')
        plt.plot(x_bins_log_mm,f2)
        plt.xscale('log')
        plt.yscale('log')
    else:
        #calculates slopes for a series of timesteps
        timehack = np.arange(plume_times.size)/2.+6.5
        time_nozeros_plumes = []
        slope_plumes = []
        for k in range(timehack.size):
            plume_time_area_timestep = plumes_time_area.loc[plumes_time_area[time]==plume_times[k-1]]    

            bins_log_mm, ind, CSD = log_binner_minmax(plume_time_area_timestep[area],bin_min,bin_max,bin_n)
            x_bins_log_mm = bins_log_mm[:-1]/2.+bins_log_mm[1:]/2.

            pos_min = 0
            pos_max = -1

            if np.isnan(np.sum(CSD)):
                nan_pos = [0]
                for i in range(CSD.size):
                    if np.isnan(CSD[i]):
                        nan_pos.append(i)
                nan_pos.append(CSD.size)
                nan_pos = np.asarray(nan_pos)

                pos_min = nan_pos[np.argmax(nan_pos[1:]-nan_pos[:-1])]+1
                pos_max = nan_pos[np.argmax(nan_pos[1:]-nan_pos[:-1])+1]-1

                print('nan_pos:',nan_pos)

                m2, b2 = np.polyfit(np.log(x_bins_log_mm[pos_min:pos_max]),np.log(CSD[pos_min:pos_max]), 1)
                f2 = 10 ** (m2 * np.log10(x_bins_log_mm)) * np.exp(b2)
            else:
                m2, b2 = np.polyfit(np.log(x_bins_log_mm),np.log(CSD), 1)
                
            slope_plumes.append(m2)
            time_nozeros_plumes.append(timehack[k])
                
        plt.plot(time_nozeros_plumes[:-1],slope_plumes[:-1])
    

def plot_cloud_alpha(data, time, n_bins, size_min, size_max, ref_min, n_cloud_min):
    """
    Written by Till Vondenhoff, 20-03-25
    
    Calculates slopes using 3 different methods (described by Newman)
    
    Parameters:
        data:           unfiltered cloud data including multiple timesteps
        time:           array with all timesteps
        n_bins:         number of bins
        ref_min:        threhold for smallest value to be counted as cloud 
        size_min:       value of the first bin
        size_max:       value of the last bin
        n_cloud_min:    minimum number of clouds per timestep required to calculate slope
        
    Returns:
        valid_time:     updated time -> timesteps where n_cloud_min is not matched are not included
        valid_n_clouds: updated number of clouds -> n_clouds where n_cloud_min is not matched are not included
        m1:             slope linear regression of power-law distribution with linear binning
        m2:             slope linear regression of power-law distribution with logarithmic binning
        m3:             slope linear regression of cumulative distribution (alpha-1)
        
    """
    slope_lin = []
    slope_log = []
    slope_cum = []
    
    valid_n_clouds = []
    valid_time     = []

    for timestep in time:
        timestep_data = data[timestep,:,:]
    
        # marks everything above ref_min as a cloud
        cloud_2D_mask = np.zeros_like(timestep_data)
        cloud_2D_mask[timestep_data > ref_min] = 1

        # calculates how many clouds exist in cloud_2D_mask, returns total number of clouds
        labeled_clouds, n_clouds = ndi.label(cloud_2D_mask)
        labels = np.arange(1, n_clouds + 1)
        
        if (n_clouds <= n_cloud_min):
            slope_lin.append(np.NaN)
            slope_log.append(np.NaN)
            slope_cum.append(np.NaN)

            valid_n_clouds.append(n_clouds)
            valid_time.append(timestep)
            print ('timestap',timestep,'has too few clouds:', n_clouds)
            continue
        
        valid_n_clouds.append(n_clouds)
        valid_time.append(timestep)
        # Calculating how many cells belong to each labeled cloud using ndi.labeled_comprehension
        # returns cloud_area and therefore it's 2D size
        cloud_number_cells = ndi.labeled_comprehension(cloud_2D_mask,labeled_clouds,labels, np.size, float, 0)
        
        cloud_area = np.sqrt(cloud_number_cells)*25
        
    # linear power-law distribution of the data (a,b)
        f, slope, intercept = lin_binning(cloud_area, n_bins, size_min, size_max, show_plt=0)
        slope_lin.append(slope)

    # logarithmic binning of the data (c)
        f, slope, intercept = log_binning(cloud_area, n_bins, size_min, size_max, show_plt=0)
        slope_log.append(slope)

    # cumulative distribution by sorting the data (d)
        f, slope, intercept = cum_dist(cloud_area, size_min, size_max, show_plt=0)
        slope_cum.append(slope)
        
    return valid_time, valid_n_clouds, slope_lin, slope_log, slope_cum
    
    
def plot_plume_alpha(plumes_time_area, n_bins, size_min, size_max, n_plume_min, show_plt):
    """
    Written by Till Vondenhoff, 20-03-27
    
    Calculates slopes using 3 different methods (described by Newman)
    
    Parameters:
        plumes_time_area: plume data including multiple timesteps
        bin_n:            number of bins
        bin_min:          value of the first bin
        bin_max:          value of the last bin
        min_pixel:        threshold for minimum number of pixels per cloud
        n_plume_min:      minimum number of clouds per timestep required to calculate slope
        
    Returns:
        valid_time:       updated time -> timesteps where n_plume_min is not matched are not included
        valid_n_plumes:   updated number of plumes -> n_plumes where n_plume_min is not matched are not included
        m1:               slope linear regression of power-law distribution with linear binning
        m2:               slope linear regression of power-law distribution with logarithmic binning
        m3:               slope linear regression of cumulative distribution (alpha-1)
        
    """
    slope_lin = []
    slope_log = []
    slope_cum = []
    valid_time = []
    valid_n_plumes = []
    
    plume_time = np.unique(plumes_time_area['time'])
    timehack = np.arange(plume_time.size)/2.+6.5
    
    for timestep in range(timehack.size):
        plumes_time_area_timestep = plumes_time_area.loc[plumes_time_area['time']==plume_time[timestep-1]]
        
        n_plumes = np.size(plumes_time_area_timestep['sq Area'])

        if (n_plumes == 0 and len(valid_time) == 0):
            print ('No plumes in this timestep.')
            continue
        elif (n_plumes <= n_plume_min):
            slope_lin.append(np.NaN)
            slope_log.append(np.NaN)
            slope_com.append(np.NaN)
            
            valid_n_plumes.append(n_plumes)
            valid_time.append(timestep)
            continue
        valid_n_plumes.append(n_plumes)
        valid_time.append(timestep)
        
        plume_area = plumes_time_area_timestep['sq Area'] #/(25**2)
        plume_area_min = np.min(plume_area)
        plume_area_max = np.max(plume_area)+1
        
        if show_plt:
            CSD, bins = np.histogram(plume_area, bins=n_bins)

            bin_width = (bins[-1]-bins[0])/len(bins)
            bins = bins[1:]/2 + bins[:-1]/2

            plt.axvline(x=size_min, color='red', linewidth=1.5, alpha=0.8, linestyle='--')
            plt.axvline(x=size_max, color='red', linewidth=1.5, alpha=0.8, linestyle='--')

            plt.plot(bins, CSD/bin_width)
            plt.axvspan(0, size_min, color='gray', alpha=0.4, lw=0)
            plt.axvspan(size_max, np.max(plume_area), color='gray', alpha=0.4, lw=0)
            plt.xlabel('cloud size [m]')
            plt.ylabel('probability density function')
            plt.title('linear histogram')
            plt.xlim(0, np.max(plume_area))
            plt.show()

        # linear power-law distribution of the data
        f, slope, intercept = lin_binning(plume_area, n_bins, size_min, size_max, show_plt)
        slope_lin.append(slope)

        # logarithmic binning of the data
        f, slope, intercept = log_binning(plume_area, n_bins, size_min, size_max, show_plt)
        slope_log.append(slope)

        # cumulative distribution by sorting the data
        f, slope, intercept = cum_dist(plume_area, size_min, size_max, show_plt)
        slope_cum.append(slope)

    return valid_time, valid_n_plumes, slope_lin, slope_log, slope_cum