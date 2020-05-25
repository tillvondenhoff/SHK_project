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

        j = 0
        global_j = 0
        global_i = 0
        for i in range(CSD.size):
            j += 1
            if math.isnan(CSD[i]):
                  j = 0
            elif j > global_j:
                global_j = j
                global_i = i     
        if (len(x_bins_log_mm[global_i-global_j+1:global_i+1])==0):
            print('there are no clouds for this timestep')
        else:
            m2,b2 = np.polyfit(np.log(x_bins_log_mm[global_i-global_j+1:global_i+1]),np.log(CSD[global_i-global_j+1:global_i+1]), 1)
            f2 = 10**(m2*np.log10(x_bins_log_mm))*np.exp(b2)

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

                j = 0
                global_j = 0
                global_i = 0
                for i in range(CSD.size):
                    j += 1
                    if math.isnan(CSD[i]):
                          j = 0
                    elif j > global_j:
                        global_j = j
                        global_i = i     
                if (len(x_bins_log_mm[global_i-global_j+1:global_i+1])==0):
                    continue
                else:
                    m2,b2 = np.polyfit(np.log(x_bins_log_mm[global_i-global_j+1:global_i+1]),np.log(CSD[global_i-global_j+1:global_i+1]), 1)
                    f2 = 10**(m2*np.log10(x_bins_log_mm))*np.exp(b2)
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

        j = 0
        global_j = 0
        global_i = 0
        for i in range(CSD.size):
            j += 1
            if math.isnan(CSD[i]):
                j = 0
            elif j > global_j:
                global_j = j
                global_i = i

            m2,b2 = np.polyfit(np.log(x_bins_log_mm[global_i-global_j+1:global_i+1]),np.log(CSD[global_i-global_j+1:global_i+1]), 1)
            f2 = 10**(m2*np.log10(x_bins_log_mm))*np.exp(b2)
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

            j = 0
            global_j = 0
            global_i = 0
            for i in range(CSD.size):
                j += 1
                if math.isnan(CSD[i]):
                        j = 0
                elif j > global_j:
                    global_j = j
                    global_i = i     
            if (len(x_bins_log_mm[global_i-global_j+1:global_i+1])==0):
                continue
            else:
                m2,b2 = np.polyfit(np.log(x_bins_log_mm[global_i-global_j+1:global_i+1]),np.log(CSD[global_i-global_j+1:global_i+1]), 1)
                f2 = 10**(m2*np.log10(x_bins_log_mm))*np.exp(b2)
                slope_plumes.append(m2)
                time_nozeros_plumes.append(timehack[k]) 
        plt.plot(time_nozeros_plumes[:-1],slope_plumes[:-1])
    

def plot_cloud_alpha(data, time, bin_n, bin_min, bin_max, ref_min, min_pixel,n_cloud_min):
    """
    Written by Till Vondenhoff, 20-03-25
    
    Calculates slopes using 3 different methods (described by Newman)
    
    Parameters:
        data:           unfiltered cloud data including multiple timesteps
        time:           array with all timesteps
        bin_n:          number of bins
        ref_min:        threhold for smallest value to be counted as cloud 
        bin_min:        value of the first bin
        bin_max:        value of the last bin
        min_pixel:      threshold for minimum cloud size in pixel
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
    slope_com = []
    valid_time = []
    valid_n_clouds = []

    for timestep in time:
        timestep_data = data[timestep,:,:]
    
        # marks everything above ref_min as a cloud
        cloud_2D_mask = np.zeros_like(timestep_data)
        cloud_2D_mask[timestep_data > ref_min] = 1


        # calculates how many clouds exist in cloud_2D_mask, returns total number of clouds
        labeled_clouds, n_clouds = ndi.label(cloud_2D_mask)
        labels = np.arange(1, n_clouds + 1)
        
        if (n_clouds == 0 and len(valid_time) == 0):
            continue
        elif (n_clouds <= n_cloud_min):
            slope_lin.append(np.NaN)
            slope_log.append(np.NaN)
            slope_com.append(np.NaN)
            
            valid_n_clouds.append(n_clouds)
            valid_time.append(timestep)
            print ('n_clouds[',timestep,']:', n_clouds)
            continue
            
        valid_n_clouds.append(n_clouds)
        valid_time.append(timestep)
        #print('number of clouds', n_clouds)

        # Calculating how many cells belong to each labeled cloud using ndi.labeled_comprehension
        # returns cloud_area and therefore its 2D size
        cloud_number_cells = ndi.labeled_comprehension(cloud_2D_mask,labeled_clouds,labels, np.size, float, 0)
        
        #cloud_number_cells = cloud_number_cells[cloud_number_cells>min_pixel]
        cloud_area = np.sqrt(cloud_number_cells)*25
        cloud_area_min = np.sqrt(min_pixel)*25.
        cloud_area_max = np.max(cloud_area)+1
        #print('min cloud area:', np.min(cloud_area),'\nmax cloud area:', np.max(cloud_area))

        
    # linear power-law distribution of the data (a,b)
        y, bins_lin = np.histogram(cloud_area, bins=bin_n, density=True, range=(cloud_area_min, cloud_area_max))
        x_bins_lin = bins_lin[:-1] / 2. + bins_lin[1:] / 2.
        x_nozeros = []
        y_nozeros = []
        for i in range(y.size):
            if y[i] != 0.:
                y_nozeros.append(y[i])
                x_nozeros.append(x_bins_lin[i])
            elif y[i] == 0.:
                x_min_shade = x_bins_lin[i]
                break
        m1, b1 = np.polyfit(np.log(x_nozeros), np.log(y_nozeros), 1)
        slope_lin.append(m1)

    # logarithmic binning of the data (c)
        bins_log_mm, ind, CSD = log_binner_minmax(cloud_area, bin_min, bin_max, bin_n)
        #bin_min = max(cloud_area_min, bin_min)
        #bin_max = min(cloud_area_max, bin_max)
        #CSD, bins_log_mm = np.histogram(cloud_area, bins=np.logspace(np.log10(bin_min),np.log10(bin_max), bin_n+1),range=(cloud_area_min, cloud_area_max))
        x_bins_log_mm = bins_log_mm[:-1] / 2. + bins_log_mm[1:] / 2.
        
        """x_mm_nozeros = []
        valid_bins = []
        valid_CSD = []
        j = 0
        global_j = 0
        global_i = 0
        for i in range(CSD.size):
            j += 1
            if np.isnan(CSD[i]):
                j = 0
                x_mm_nozeros.append(x_bins_log_mm[i])
            elif j > global_j:
                valid_CSD.append(CSD[i])
                valid_bins.append(x_bins_log_mm[i])
                global_j = j
                global_i = i
            else:
                valid_CSD.append(CSD[i])
                valid_bins.append(x_bins_log_mm[i])
        if (len(x_bins_log_mm[global_i-global_j+1:global_i+1])==0):
            break
        else:
            m2, b2 = np.polyfit(np.log(valid_bins),np.log(valid_CSD), 1)"""
        x_mm_nozeros = []
        CSD_nozeros = []
        for i in range(CSD.size):
            if np.isnan(CSD[i]):
                CSD_nozeros.append(CSD[i])
                x_mm_nozeros.append(x_bins_log_mm[i])
        j = 0
        global_j = 0
        global_i = 0
        for i in range(CSD.size):
            j += 1
            if np.isnan(CSD[i]):
                j = 0
            elif j > global_j:
                global_j = j
                global_i = i
                
        if (len(x_bins_log_mm[global_i-global_j+1:global_i+1])==0):
            break
        else:
            m2,b2 = np.polyfit(np.log(x_bins_log_mm[global_i-global_j+1:global_i+1]),np.log(CSD[global_i-global_j+1:global_i+1]), 1)
            slope_log.append(m2)

    # cumulative distribution by sorting the data (d)
        alpha = alpha_newman5(cloud_area, np.sqrt(min_pixel)*25.)
        #m3 = -alpha + 1
        m3 = -alpha
        slope_com.append(m3)
    
    return valid_time, valid_n_clouds, slope_lin, slope_log, slope_com
    

def plot_plume_alpha(plumes_time_area, bin_n, bin_min, bin_max, min_pixel, n_plume_min):
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
    slope_com = []
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
        plume_area_min = max(np.sqrt(min_pixel)*25.,np.min(plume_area))
        plume_area_max = max(plume_area)+1
        
    # linear power-law distribution of the data (a,b)
        y, bins_lin = np.histogram(plume_area, bins=bin_n, density=True, range=(plume_area_min, plume_area_max))
        x_bins_lin = bins_lin[:-1] / 2. + bins_lin[1:] / 2.
        x_nozeros = []
        y_nozeros = []
        for i in range(y.size):
            if y[i] != 0.:
                y_nozeros.append(y[i])
                x_nozeros.append(x_bins_lin[i])
            elif y[i] == 0.:
                x_min_shade = x_bins_lin[i]
                break
        m1, b1 = np.polyfit(np.log(x_nozeros), np.log(y_nozeros), 1)
        slope_lin.append(m1)        
        
    # logarithmic binning of the data (c)
        bins_log_mm, ind, CSD = log_binner_minmax(plume_area, bin_min, bin_max, bin_n)
        x_bins_log_mm = bins_log_mm[:-1] / 2. + bins_log_mm[1:] / 2.
        x_mm_nozeros = []
        CSD_nozeros = []
        for i in range(CSD.size):
            if math.isnan(CSD[i]):
                CSD_nozeros.append(CSD[i])
                x_mm_nozeros.append(x_bins_log_mm[i])
        j = 0
        global_j = 0
        global_i = 0
        for i in range(CSD.size):
            j += 1
            if math.isnan(CSD[i]):
                j = 0
            elif j > global_j:
                global_j = j
                global_i = i
                
        if (len(x_bins_log_mm[global_i-global_j+1:global_i+1])==0):
            continue
        else:
            m2,b2 = np.polyfit(np.log(x_bins_log_mm[global_i-global_j+1:global_i+1]),np.log(CSD[global_i-global_j+1:global_i+1]), 1)
            f2 = 10**(m2*np.log10(x_bins_log_mm))*np.exp(b2)
            slope_log.append(m2)

    # cumulative distribution by sorting the data (d)
        alpha = alpha_newman5(plume_area, plume_area_min)
        #m3 = -alpha + 1
        m3 = -alpha
        slope_com.append(m3)
        
    return valid_time, valid_n_plumes, slope_lin, slope_log, slope_com
