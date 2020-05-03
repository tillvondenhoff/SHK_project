import numpy as np
import math
import scipy.ndimage as ndi
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy import stats
#import lmfit
#from lmfit import Model


def log_binner_minmax(var, bin_min, bin_max, bin_n, N_min=0):
    """
    written by Lennéa Hayo, 19-07-20
    
    Bins a vector of values into logarithmic bins
    Starting from bin_min and ending at bin_max
    
    Parameters:
        var: input vector
        bin_min: value of the first bin
        bin_max: value of the last bin
        bin_n: number of bins 
        
    Returns:
        bins: vector of bin edges, is bin_n+1 long
        ind: gives each value of the input vector the index of its respective bin
        CSD: Non normalized distribution of var over the bins. 
    """
    
    #max_val = max(var)
    #min_val = min(var)
    #bin_min = max(min_val, bin_min)
    #bin_max = min(max_val, bin_max)
    
    max_log = np.log10(bin_max / bin_min)

    bins = bin_min * np.logspace(0, max_log, num=bin_n + 1, base=10.0)

    ind = np.digitize(var, bins)
    CSD = np.zeros(bin_n)
    for b in range(bin_n):
        if len(ind[ind == b + 1]) > N_min:
            CSD[b] = float(np.count_nonzero(ind == b + 1)) / (bins[b + 1] - bins[b])
        else:
            CSD[b] = 'nan'
    return bins, ind, CSD


def alpha_newman5(dist, cloud_size_min):
    """
    Calculates alpha from cumulative distribution acording to newman paper.
    
    """
    if cloud_size_min == None:
        xmin = np.min(dist)
    else:
        xmin = cloud_size_min
    x = dist[dist > xmin]
    n = x.size
    alpha = 1. + n / (np.sum(np.log(x / xmin)))
    return alpha


def func_newmann3(dist, bin_n, bin_min, bin_max, x_min, x_max, min_pixel, show_plt):
    """
    Written by Lennéa Hayo, 19-08-01
    Edited by Till Vondenhoff, 20-03-28
    
    Creates Newmanns figure 3 plots of a logarithmic distribution. Elements which are not used
    for fitting are shaded in grey.
    
    Parameters:
        dist:      distribution of logarithmic data
        bin_n:     number of bins
        bin_min:   value of the first bin
        bin_max:   value of the last bin
        x_min:     smallest value of x (used in linear power-law-dist for x-axis)
        x_max:     highest value of x (used in linear power-law-dist for x-axis)
        min_pixel: smallest value for which the power law holds (used in alpha_newman5 to calculate the
                   alpha of cumulative dist.)
    Added Parameters:
        show_plt:  Creates 4 tile plot with different slopes if show_plt=True
        
    Returns:
        fig:       plots that resemble Newmans figure 3
        m1:        slope linear regression of power-law distribution with log scales
        m2:        slope linear regression of power-law distribution with log binning
        m3:        slope linear regression of cumulative distribution (alpha-1)

    """
    import numpy as np

    N_samples = len(dist)
    

    # linear power-law distribution of the data (a,b)
    y, bins_lin = np.histogram(dist, bins=bin_n, density=True, range=(x_min, x_max))
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
    try:
        x_min_shade
    except:
        x_min_shade = x_max_shade
    m1, b1 = np.polyfit(np.log(x_nozeros), np.log(y_nozeros), 1)
    f1 = np.exp(m1 * np.log(x_bins_lin)) * np.exp(b1)
    x_max_shade = x_bins_lin[-1]

    # logarithmic binning of the data (c)
    bins_log_mm, ind, CSD = log_binner_minmax(dist, bin_min, bin_max, bin_n)
    #CSD, bins_log_mm = np.histogram(dist, bins=np.logspace(np.log10(x_min),np.log10(x_max), bin_n+1),range=(x_min, x_max))
    x_bins_log_mm = bins_log_mm[:-1] / 2. + bins_log_mm[1:] / 2.
    
    x_mm_nozeros = []
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

    m2, b2 = np.polyfit(np.log(valid_bins),np.log(valid_CSD), 1)
    f2 = 10 ** (m2 * np.log10(x_bins_log_mm)) * np.exp(b2)
    # where to start and end shadow of values that are not being used for the fit
    try:
        x_min_shade1 = x_mm_nozeros[0]
    except:
        x_min_shade1 = 0
    x_max_shade1 = x_bins_log_mm[global_i - global_j + 1]
    
    try:
        x_min_shade2 = x_bins_log_mm[global_i + 1]
    except:
        x_min_shade2 = x_bins_log_mm[-1]
    x_max_shade2 = x_max
    
    # cumulative distribution by sorting the data (d)
    dist_sort = np.sort(dist)
    dist_sort = dist_sort[::-1]
    p = np.array(range(N_samples)) / float(N_samples)
    # calculate slope (alpha)
    #print (np.min(dist))
    alpha = alpha_newman5(dist, np.sqrt(min_pixel)*25.)
    x_min_shade3 = np.min(dist_sort)
    x_max_shade3 = np.sqrt(min_pixel)*25.

    # calculate y-intercept of linear equation with alpha-1
    def powerlaw_func(x, alpha, C):
        return C * x ** -alpha

    def cumulative_dist_func(x, alpha, C):
        return C / (alpha - 1) * x ** -(alpha - 1)

    # reverse p and dist_sort so they go from smallest to biggest and first index for which powerlaw holds
    dist_sort_rev = dist_sort[::-1]
    p_rev = p[::-1]
    first_idx = np.where(dist_sort_rev > np.sqrt(min_pixel)*25.)[0][0]
    #print('first index:',first_idx)
    sum_cum_dist = np.sum(p_rev[first_idx:].dot(dist_sort_rev[first_idx:] - dist_sort_rev[first_idx - 1:-1]))
    # integrate powerlaw of alpha-1 with sum over all values
    integral_powerlaw = np.sum(powerlaw_func(dist_sort_rev[first_idx:], alpha - 1, 1) * (dist_sort_rev[first_idx:] - dist_sort_rev[first_idx - 1:-1]))

    b4 = (alpha - 1) * sum_cum_dist / integral_powerlaw
    f3 = cumulative_dist_func(dist_sort, alpha, b4)
    
    if (show_plt==True):
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))

        axes[0, 0].plot(x_bins_lin, y)
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('probability density function')
        axes[0, 0].set_title(r'Power-law distribution with an exponent of $\alpha$')

        axes[0, 1].plot(x_bins_lin, y, 'o')
        axes[0, 1].loglog(x_bins_lin, y)
        axes[0, 1].plot(x_bins_lin, f1)
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('samples')
        axes[0, 1].text(np.max(x_bins_lin), np.max(y), r'$\alpha = %f$' % m1, horizontalalignment='right',
                        verticalalignment='top')
        axes[0, 1].set_title('Power-law distribution with log scales')
        axes[0, 1].axvspan(x_min_shade, x_max_shade, color='lightgray', alpha=0.5, lw=0)

        axes[1, 0].plot(x_bins_log_mm, CSD, 'o')
        axes[1, 0].loglog(x_bins_log_mm, CSD)
        axes[1, 0].plot(x_bins_log_mm, f2)
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('samples')
        axes[1, 0].text(x_min_shade2, CSD[global_i - global_j + 1], r'$\alpha = %f$' % m2,
                        horizontalalignment='right', verticalalignment='top')
        axes[1, 0].set_title('Power-law distribution with log binning')
        axes[1, 0].axvspan(x_min_shade1, x_max_shade1, color='lightgray', alpha=0.5, lw=0)
        axes[1, 0].axvspan(x_min_shade2, x_max_shade2, color='lightgray', alpha=0.5, lw=0)

        axes[1, 1].plot(dist_sort, p, 'o')
        axes[1, 1].loglog(dist_sort, p)
        axes[1, 1].plot(dist_sort, f3)
        axes[1, 1].set_xlabel('normed distribution')
        axes[1, 1].set_ylabel('samples with values > x')
        axes[1, 1].text(np.max(dist_sort), np.max(p), r'$\alpha = %f$' % (-alpha), horizontalalignment='right',
                        verticalalignment='top')
        axes[1, 1].set_title(r'Cumulative distribution with an exponent of $\alpha-1$')
        axes[1, 1].axvspan(x_min_shade3, np.asarray(x_max_shade3), color='lightgray', alpha=0.5, lw=0)

        m3 = -alpha
        return fig, m1, m2, m3
    else:
        m3 = -alpha
        fig = 0
        return fig, m1, m2, m3


def cloud_size_dist(dist, time, bin_n, bin_min, bin_max, ref_min, file, min_pixel, show_plt):
    """
    Written by Lennéa Hayo, 2019
    Edited by Till Vondenhoff, 20-03-28
    
    Creates Newmanns figure 3 of cloud size distribution
    
    Parameters:
        dist:      netcdf file of satelite shot with clouds
        bin_n:     number of bins
        ref_min:   smallest value 
        bin_min:   value of the first bin
        bin_max:   value of the last bin
        file:      name given to dataset of netcdf file
        min_pixel: smallest cloud size value for which the power law holds (used in alpha_newman5 to calculate the
                   alpha of cumulative dist.)
    Added Parameters:
        show_plt:  Creates 4 tile plot with different slopes if show_plt=True
        
    Returns:
        fig:       plots that resemble Newmans figure 3
        m1:        slope linear regression of power-law distribution with log scales
        m2:        slope linear regression of power-law distribution with log binning
        m3:        slope linear regression of cumulative distribution (alpha-1)
    
    """

    r1 = file[dist][time]

    # marks everything above ref_min as a cloud
    cloud_2D_mask = np.zeros_like(r1)
    cloud_2D_mask[r1 > ref_min] = 1

    # calculates how many clouds exist in cloud_2D_mask, returns total number of clouds
    labeled_clouds, n_clouds = ndi.label(cloud_2D_mask)
    labels = np.arange(1, n_clouds + 1)
    print('number of clouds:',n_clouds)
    
    # Calculating how many cells belong to each labeled cloud using ndi.labeled_comprehension
    # returns cloud_area and therefore its 2D size
    cloud_pixel = ndi.labeled_comprehension(cloud_2D_mask,labeled_clouds,labels, np.size, float, 0)
    cloud_area = np.sqrt(cloud_pixel)*25.
    cloud_area_min = np.sqrt(min_pixel)*25.
    cloud_area_max = np.max(cloud_area)
    #print('number of Clouds:',n_clouds,'\nmin cloud area:',np.min(cloud_area),'\nmax cloud area:',np.max(cloud_area))

    fig, m1, m2, m3 = func_newmann3(cloud_area, bin_n, bin_min, bin_max, cloud_area_min, cloud_area_max, min_pixel, show_plt)

    return fig, m1, m2, m3


def alpha_ref_min(dist, bin_n, ref_min, bin_min, bin_max, file, min_pixel):
    """
    Written by Lennéa Hayo, 19-09-20
    
    Creates a plot comparing the alphas from func_newman3. It uses an array of different ref_mins,
    to see how alpha changes
    
    Parameters: 
        dist: netcdf file of satelite shot with clouds
        bin_n: number of bins
        ref_min: array containing the smallest values which determine what is to be considered a cloud
        bin_min: value of the first bin
        bin_max: value of the last bin
        file: name given to dataset of netcdf file
        
    Returns:
        plot: plot that contains the different alphas
    """
    alpha_lin = []
    alpha_log = []
    alpha_cum_dist = []
    for i in range(len(ref_min)):
        fig, m1, m2, m3 = cloud_size_dist(dist, bin_n, ref_min[i], bin_min, bin_max, file, min_pixel)
        alpha_lin.append(m1)
        alpha_log.append(m2)
        alpha_cum_dist.append(m3 - 1)
        plt.close(fig)

    plt.plot(ref_min, alpha_lin, '-o', label=r'$\alpha$ lin')
    plt.plot(ref_min, alpha_log, '-o', label=r'$\alpha$ log')
    plt.plot(ref_min, alpha_cum_dist, '-o', label=r'$\alpha$ cumulative dist')

    plt.xlabel('ref_min')
    plt.ylabel(r'$\alpha$')
    plt.legend(loc='best')
    plt.show()

    # print(alpha_lin,alpha_log,alpha_cum_dist)
