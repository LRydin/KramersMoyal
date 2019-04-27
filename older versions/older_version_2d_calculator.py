#! /usr/bin/env python
# -*- coding: utf-8 -*-

def KMC_kernel_estimator_NEW(y, p, bandwidth, dt, n_sigma_statistics):
    """This function estimates 2D Kramers-Moyal coefficients from 2 timeseries using an kernel estimator method. This is the recent version (April 30, 2018).

    *parameters*:
    -y: a [N,2] matrix with the two ts of length N
    -p: a vector with the information, which kmc should be computed
    -bandwidth: the bandwidth for the kernel
    -n_sigma_statistics: the limits of the BinsSpace are set to +/- n_sigma_statistics*sigma, if n_sigma_statistics = 42 is given, the maximum and minimum of the timeseries are taken as limits instead.
    -method=2: the method to compute the kmc. For large N, method 2 is faster.

    *return*:
    -kmc2D
    -BinsSpace

    *Further Information*:
    This function uses an Epanechnikov kernel for the estimation of the kmc.
    For reliable results, 1e7 data points per timeseries are recommended."""

    # Main variable allocation and Kernel shape generator
    NofMoments = np.size(p,1)
    Dim = 2
    N = np.size(y,0)                                                     # Find size of the timeseries
    n  = 200                                                             # no. of points of Kernel space
    Mn = int(n * bandwidth / 10)                                         # no. of points of Kernel itself
    x1  = np.linspace(-1 * bandwidth, 1 * bandwidth, Mn)                 # 1d linear space for Kernel generator
    x1_2D, y1_2D = np.meshgrid(x1, x1, sparse=True)                      # 2d linear space for Kernel generator
    BinsSpace = np.zeros([n + Mn, Dim])                                  # Records x1 and x2 spaces     
    kmc = np.zeros([n + Mn, n + Mn, np.size(p,1)])                       # Array for the 2d coefficients 
    b = np.zeros([N - 1, Dim]).astype(int)                               # Array for the binning for Kernels


    # Epanechnikov kernel:   (8/3*pi)*3/4(1 - (x² + y²), x=-1 to x=1
    Kernel_2D = (1 - ((x1_2D ** 2 + y1_2D ** 2) / (bandwidth ** 2))) / (bandwidth * np.pi)   #Generate Epanechnikov kernel
    Kernel_2D[Kernel_2D<0.] = 0.                                                             #Remove <0 values


    ###### Here starts the main code section. #####

    yDist = y[1:, :] - y[:-1, :]                                         # Find distances

    for k in range(0, 2):                                               # Use or find lower- and upper-bounds

        if (n_sigma_statistics < 42):
            Mean = np.mean(y[:,k])
            Sigma = np.std(y[:,k])
            Min = Mean - n_sigma_statistics * Sigma
            Max = Mean + n_sigma_statistics * Sigma

        else:
            Min =  np.min(y[:,k])
            Max =  np.max(y[:,k])

        BinsSpace[:, k] = np.linspace(Min, Max, n + Mn)                               # Define bin spacing

        b[:, k] = ((((y[:-1, k]-Min) / (abs(Max - Min))) * (n))).astype(int)          # Cast each point into bin


    #############################         Method            ######################################
    # The calculation method has been vectorialised by FRANCISCO MEIRINHOS and is x10 faster now #

    pos0 = map(lambda i: np.where(b[:,0]==i), range(n))
    pos1 = list(map(lambda j: np.where(b[:,1]==j), range(n)))

    #run over all bins in x0
    for l, p0 in enumerate(list(pos0)):          

        if np.size(p0) == 0:
            continue

        #run over all bins in x1
        for m, p1 in enumerate(pos1):
            pos  = np.intersect1d(p0, p1)

            if np.size(pos) == 0:
                continue

            kmc[l:l+Mn, m:m+Mn] += Kernel_2D[:,:,np.newaxis] * np.sum(np.power(yDist[pos,0,np.newaxis],p[0,:]) * np.power(yDist[pos,1,np.newaxis],p[1,:]), axis=0)


    ##### Normalisation #####
    kmc[:, :, 0][kmc[:, :, 0]==0.] = 1.

    kmc[:, :, 1:] = np.divide(kmc[:, :, 1:],(kmc[:, :, 0, np.newaxis] * dt))

    return kmc, BinsSpace

