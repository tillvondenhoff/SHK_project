import numpy as np
from importlib import reload
import slopes_and_binning
from slopes_and_binning import *
reload(slopes_and_binning)
import cloud_and_plumes_slopes
from cloud_and_plumes_slopes import *
reload(cloud_and_plumes_slopes)
import matplotlib.pyplot as plt


def variability(center_x, center_y, size, time_vector, start_time, increment, domain_size, subdomains, n_bins, min_size, max_size, log_binning, inbetween_subs, show_subs):
    '''
    Written by Till Vondenhoff, 20-05-16
    
    This function checks for organization in a cloud/plume field.
    Therefore the full domain gets divided into a number of subdomains from which the variance and the mean number of clouds/plumes for one size are calculated.

    Parameters:
        center_x: contains the x values of each cloud/plume center
        center_y: contains the y values of each cloud/plume center
        size: for every x and y values there is a corresponding area stored in 'size' [m^2]
        time_vector: an array of times over which to iterate with no particular order needed
        domain_size: holds the side length of the full domain. Domains are expected to be the same length both the x and y-axis
        subdomains: holds values to specify the subdivision of the full domain. Values being applied to both x and y-axis. Useful values seem to be [2,4,8,16,32]
        n_bins: number of bins for the histogram, using either logarithmic or linear binning
        bin_min: value of the first bin
        bin_max: value of the last bin

        log_binning: True or False. Linear binning is used if set to False
        inbetween_subs: if set to True overlapping subdomains will be used in addition to the normal gridded ones to get a more accurate variance
        show_subs: creates a log(σ/N)-L and a log(σ/N)-log(L/l) plot for every timestep in time_vector

    Returns:
        f: f holds slope and intercept for all cloud/plume sizes of all timesteps
        
    '''
    f = np.zeros((n_bins,2,len(time_vector)))
    avg_slope = []
    index = 0
    
    for timestep in time_vector:
        new_arr = np.zeros((domain_size,domain_size))
        new_arr[center_x[timestep],center_y[timestep]] = size[timestep]
        
        var    = []
        N_mean = []
        L      = []
        
        for n_slices in subdomains:
            x_split = int(domain_size/n_slices)
            y_split = x_split

            dist             = []                   #dist holds the numbers of clouds/plumes per bin for all subdomains of one subdomain size
            var_subdomain    = []
            N_mean_subdomain = []
            
            L.append(len(new_arr)/n_slices)
            
            for y in range(n_slices):
                for x in range(n_slices):
                    tmp_subdomain = new_arr[y*y_split:(y+1)*y_split,x*x_split:(x+1)*x_split]
                    sub_size = tmp_subdomain[tmp_subdomain>min_size]
                    if log_binning:
                        n_per_bin, l = np.histogram(sub_size, bins=np.logspace(np.log10(min_size),np.log10(max_size), n_bins+1),range=(min_size, max_size))
                    else:
                        n_per_bin, l = np.histogram(sub_size, bins=n_bins,range=(min_size, max_size))
                    dist.append(n_per_bin)
                    
            if inbetween_subs:
                for y in range(n_slices):
                    for x in range(n_slices-1):
                        tmp_subdomain = new_arr[y*y_split:(y+1)*y_split,int((x+.5)*x_split):int((x+1.5)*x_split)]
                        tmp_size = tmp_subdomain[tmp_subdomain>min_size]
                        if log_binning:
                            n_per_bin, l = np.histogram(tmp_size, bins=np.logspace(np.log10(min_size),np.log10(max_size), n_bins+1),range=(min_size, max_size))
                        else:
                            n_per_bin, l = np.histogram(tmp_size, bins=n_bins,range=(min_size, max_size))
                        dist.append(n_per_bin)

                for y in range(n_slices-1):
                    for x in range(n_slices):
                        tmp_subdomain = new_arr[int((y+.5)*y_split):int((y+1.5)*y_split),x*x_split:(x+1)*x_split]
                        sub_size = tmp_subdomain[tmp_subdomain>min_size]
                        if log_binning:
                            n_per_bin, l = np.histogram(sub_size, bins=np.logspace(np.log10(min_size),np.log10(max_size), n_bins+1),range=(min_size, max_size))
                        else:
                            n_per_bin, l = np.histogram(sub_size, bins=n_bins,range=(min_size, max_size))
                        dist.append(n_per_bin)

            dist = np.asarray(dist)
            for i in range(n_bins):
                N_mean_subdomain.append(np.mean(dist[:,i]))
                var_subdomain.append(np.sqrt(np.sum((dist[:,i]-N_mean_subdomain[i])**2)/len(dist[:,0])))

            var.append(var_subdomain)
            N_mean.append(N_mean_subdomain)

        N_mean = np.asarray(N_mean)
        var = np.asarray(var)

        y_axis = np.log(var/N_mean)
        
        slope     = []
        intercept = []
        L_labels = [25/2,25/4,25/8,25/16,25/32] #['0.781','1.563','3.125','6.25','12.5']
        if show_subs:
            fig,ax = plt.subplots(1,2,figsize=(10,5))
            for i in range(n_bins):
                x_axis = np.log(L/l[i])
                m, b = np.polyfit(x_axis, y_axis[:,i], 1)
                slope.append(m)
                intercept.append(b)
                
                color = plt.cm.cool((i+1)/(n_bins+1))
                im1 = ax[0].plot(np.log(L),y_axis[:,i], '-o', label='l = %i m - %i m' %(int(l[i]),int(l[i+1])-1), color=color)
                im2 = ax[1].plot(x_axis,y_axis[:,i], '-o', label='l = %i m - %i m' %(int(l[i]),int(l[i+1])-1), color=color)
                
            im2 = ax[1].plot([-4,2],np.polyval([-1,-1],[-4,2]),'--')
            #ax[0].set_title('organization for timestep %i' %timestep)
            ax[0].legend(loc='best')
            ax[0].set_xlabel('L [ km ]')
            ax[0].set_ylabel('log( σ / N )')
            #ax[0].set_xscale('log')
            ax[1].legend(loc='best')
            ax[1].set_xlabel('log( L / l )')
            ax[1].set_ylabel('log( σ / N )')
            
            plt.sca(ax[0])
            plt.xticks(np.log(L), ['12.5','6.25','3.125','1.563','0.781'])
            #fig.suptitle('plume organization \ndate: 30th august 2016, time: 12:30', fontsize=16)
            #plt.savefig('organization_plumes_Ts12_20160830.pdf',bbox_inches='tight')
            plt.show()
        else:
            for i in range(n_bins):
                x_axis = np.log(L/l[i])
                m, b = np.polyfit(x_axis, y_axis[:,i], 1)
                slope.append(m)
                intercept.append(b)

        f[:,0,index] = slope
        f[:,1,index] = intercept
        avg_slope.append(np.nanmean(slope))
        index += 1
    
    if (len(time_vector) > 1):
        time_labels = get_time_labels(start_time+time_vector[0]/2, increment, len(time_vector))
        plt.figure(figsize=(10,8))
        for i in range(len(f[:,0,0])):
            color = plt.cm.cool((i+1)/(len(f[:,0,0])+1))
            plt.plot(time_labels,f[i,0,:],'-', linewidth=2, label='l = %i m - %i m'%(int(l[i]),int(l[i+1])-1), color=color)
            
        #plt.plot(time_labels,avg_slope,'-', linewidth=1, label='avg. (all sizes)',color='red')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel(r'slope ($\alpha$)')
        plt.ylim(-1.5, -0.6)
        plt.axhline(y=-1, color='grey', linestyle='--')
    plt.gcf().autofmt_xdate()
    
    return f