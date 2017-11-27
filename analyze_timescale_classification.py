'''
Find the jitter interval lengths that best classifies the spike train pairs
in terms of 'presence' or 'absence' of a monosynaptic coupling.

Jonathan Platkiewicz, 2017-11-26
'''

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from scipy import stats,interpolate
from ccg import correlograms
import matplotlib.pyplot as plt


def analyze(parameters,train_ref0,train_targ0):
    '''
    Inputs:
        parameters: Generative simulation parameters
        train_ref0: Reference train set 
        train_targ0: Target train set 
    Outputs:
        Figure 1: ROC curve for monosynapse detection
        Figure 2: Area under the curve for various jitter interval lenghts
    '''

    #--------------------------------------------------------------------------
    # Load the spike data
    #--------------------------------------------------------------------------

    Ntrial = parameters[0]         # Number of pairs
    duration = parameters[1]       # Trial duration in (ms)
    interval_true = parameters[2]  # Nonstationarity timescale in (ms)
    Fs = parameters[3]             # Sampling frequency

    train0 = np.append(train_ref0,train_targ0) 
    cell0 = np.int64(np.append(np.zeros(len(train_ref0)),
                               np.ones(len(train_targ0))))

    #--------------------------------------------------------------------------
    # Define analysis parameters
    #--------------------------------------------------------------------------

    inject_count = 10                         # Number of injected synchronies
    Ninjectedtrial = int(Ntrial/2.)           # Number of pairs to be injected
    synch_width = 1.                          # Width of synchrony window
    latency = 0.                              # Synaptic delay (for visual purpose)
    Njitter = 110                             # Number of jitter surrogates
    Ndelta = 20                               # Number of tested jitter timescales
    Ntest = 100                               # Number of detection threshold

    #--------------------------------------------------------------------------
    # Inject synchronous spikes
    #--------------------------------------------------------------------------

    # Inject spikes at random times avoiding present spikes 
    Nwidth = int(duration/synch_width)
    allwidths = np.arange(int(Ntrial*duration/synch_width)) 
    include_index = np.int64(np.floor(train0/synch_width))
    include_idx = list(set(include_index)) 
    mask = np.zeros(allwidths.shape,dtype=bool)
    mask[include_idx] = True
    wheretoinject = synch_width*allwidths[~mask]
    alreadythere = synch_width*allwidths[mask]
    widths = np.append(wheretoinject,alreadythere)
    tags = np.append(np.zeros(len(wheretoinject)),np.ones(len(alreadythere)))
    ind_sort = np.argsort(widths)
    widths = widths[ind_sort]
    tags = tags[ind_sort]
    widths = widths[:Ninjectedtrial*Nwidth]
    tags = tags[:Ninjectedtrial*Nwidth]
    widths = np.reshape(widths,(Ninjectedtrial,Nwidth))
    tags = np.reshape(tags,(Ninjectedtrial,Nwidth))
    ind_perm = np.transpose(np.random.permutation(np.mgrid[:Nwidth,:Ninjectedtrial][0])) 
    widths = widths[np.arange(np.shape(widths)[0])[:,np.newaxis],ind_perm]
    tags = tags[np.arange(np.shape(tags)[0])[:,np.newaxis],ind_perm]
    ind_sort = np.argsort(tags,axis=1)
    widths = widths[np.arange(np.shape(widths)[0])[:,np.newaxis],ind_sort]
    tags = tags[np.arange(np.shape(tags)[0])[:,np.newaxis],ind_sort]
    train_inject = np.ravel(widths[:,:inject_count])                    # Injected spike trains 
    train_ref = np.sort(np.append(train_ref0,train_inject))  
    train_targ = np.sort(np.append(train_targ0,train_inject+latency)) 
    train = np.append(train_ref,train_targ)
    cell = np.int64(np.append(np.zeros(len(train_ref)),np.ones(len(train_targ))))

    #--------------------------------------------------------------------------
    # Check the impact of injection
    #--------------------------------------------------------------------------

    lagmax = 100.                   # Correlogram window in (ms)
    bine = 1.                       # Correlogram time bin in (ms)

    # Select one pair of reference-target trains (without and with injection)
    ind_sort = np.argsort(train0) 
    T0 = train0[ind_sort]
    G0 = cell0[ind_sort]
    ind_sort = np.argsort(train)
    T = train[ind_sort]
    G = cell[ind_sort]
    i = 0
    j = 0
    while T0[i] < duration or T[j] < duration:
        if T0[i] < duration:
            i += 1
        if T[j] < duration:
            j += 1
    T0 = T0[:i]
    G0 = G0[:i]
    T = T[:j]
    G = G[:j]

    # Compute the correlogram matrix for the chosen pairs 
    ind_sort = np.argsort(T0)
    st = T0[ind_sort]*.001
    sc = G0[ind_sort]
    C0 = correlograms(st,sc,sample_rate=Fs,bin_size=bine/1000.,
                      window_size=lagmax/1000.)
    lag = (np.arange(len(C0[0,1]))-len(C0[0,1])/2.)*bine
    ind_sort = np.argsort(T)
    st = T[ind_sort]*.001
    sc = G[ind_sort]
    C = correlograms(st,sc,sample_rate=Fs,bin_size=bine/1000.,
                     window_size=lagmax/1000.)

    # Represent the cross-correlograms
    FigCCG = plt.figure()
    plt.xlim(-lagmax/2.,lagmax/2.)
    plt.title('Cross-Correlogram',fontsize=18)
    plt.xlabel('Time lag  (ms)',fontsize=18)
    plt.ylabel('Firing rate (Hz)',fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.plot(lag,C[0,1]/(len(train_ref)*bine*.001),'.-k')
    plt.plot(lag,C0[0,1]/(len(train_ref)*bine*.001),'--c')

    #--------------------------------------------------------------------------
    # Compute Receiver Operating Characteristic curve
    #--------------------------------------------------------------------------

    # Remove the synaptic delay for synchrony computation
    train_targ = np.sort(np.append(train_targ0,train_inject)) 
    train = np.append(train_ref,train_targ)
    cell = np.int64(np.append(np.zeros(len(train_ref)), 
                              np.ones(len(train_targ))))

    # Count the number of total observed synchronies per pair 
    Tref = synch_width*np.floor(train_ref/synch_width)
    Ttarg = synch_width*np.floor(train_targ/synch_width)
    Tsynch = np.array(list(set(Tref) & set(Ttarg)))
    synch_count = np.bincount(np.int64(np.floor(Tsynch/duration)),
                              minlength=Ntrial)

    # Compute the true positive and false positive rates for a range of detection thresholds
    delta_range = np.concatenate((np.linspace(5,100,int(Ndelta/2.)), 
                                  np.linspace(100,500,int(Ndelta/2.))))
    pvalue_inj = np.zeros((Ndelta,Ninjectedtrial))
    pvalue_noinj = np.zeros((Ndelta,Ntrial-Ninjectedtrial))
    threshold_range = np.linspace(0,1.,Ntest)
    proba_truepositive = np.zeros((Ndelta,Ntest))
    proba_falsepositive = np.zeros((Ndelta,Ntest))
    for k in range(Ndelta):
        print('Tested timescale no: ',k+1)
        interval = delta_range[k]
        # Jitter the target trains
        Tjitter = (np.tile(train_targ,Njitter) 
                  + np.sort(np.tile(np.arange(Njitter),len(train_targ)))*Ntrial*duration)
        Tjitter = (interval*np.floor(Tjitter/interval) 
                  + np.random.uniform(0,interval,len(Tjitter)))
        Tjitter = synch_width*np.floor(Tjitter/synch_width)
        # Compute the p-values under the jitter null
        Tref_jitter = (np.tile(Tref,Njitter) 
                      + np.sort(np.tile(np.arange(Njitter),len(Tref)))*Ntrial*duration)
        Tsynch_jitter = np.array(list(set(Tref_jitter) & set(Tjitter)))
        jitter_synchrony = np.bincount(np.int64(np.floor(Tsynch_jitter/duration)),
                                       minlength=Ntrial*Njitter)
        observed_synchrony = np.tile(synch_count,Njitter)
        comparison = np.reshape(np.sign(np.sign(jitter_synchrony-observed_synchrony)+1),
                                (Njitter,Ntrial))
        pvalue = (1+np.sum(comparison,axis=0))/(Njitter+1.)
        pvalue_inj[k,:] = pvalue[:Ninjectedtrial]  # Correspond to injected trains 
        pvalue_noinj[k,:] = pvalue[Ninjectedtrial:] # Correspond to non-injected trains
        # Compute the detection probabilities for each threshold value
        th = np.reshape(np.tile(threshold_range,Ninjectedtrial),
                        (Ninjectedtrial,Ntest))
        pval = np.tile(np.reshape(pvalue_inj[k,:],(Ninjectedtrial,1)),Ntest)
        proba_truepositive[k,:] = np.sum(np.sign(np.sign(th-pval)+1),axis=0)
        pval = np.tile(np.reshape(pvalue_noinj[k,:],(Ninjectedtrial,1)),Ntest)
        proba_falsepositive[k,:] = np.sum(np.sign(np.sign(th-pval)+1),axis=0)
    proba_truepositive = proba_truepositive/Ninjectedtrial
    proba_falsepositive = proba_falsepositive/(Ntrial-Ninjectedtrial)    

    #--------------------------------------------------------------------------
    # Represent the classification result
    #--------------------------------------------------------------------------

    cm = plt.get_cmap('gist_rainbow')
    FigROC = plt.figure()
    ax = FigROC.add_subplot(111)
    ax.set_color_cycle([cm(1.*i/Ndelta) for i in range(Ndelta)])
    plt.title('ROC curve',fontsize=18)
    plt.xlabel('False positive rate',fontsize=18)
    plt.ylabel('True positive rate',fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(-.1,1.1)
    plt.ylim(-.1,1.1)
    clr = np.linspace(0,.5,Ndelta)
    x = np.arange(0,1,.01)
    y = np.zeros((Ndelta,Ntest))
    area = np.zeros(Ndelta)
    for i in range(Ndelta):
        ax.plot(proba_falsepositive[i,:],proba_truepositive[i,:],'.-')
        # Compute the area under the ROC curve
        Fx = interpolate.interp1d(proba_falsepositive[i,:],proba_truepositive[i,:],
                                  bounds_error=False,fill_value=0.,kind='nearest')
        y_interp = Fx(x) 
        area[i] = np.sum(y_interp)*(x[1]-x[0])
    plt.plot([0,1],[0,1],'--b')
    plt.plot(.5*np.ones(2),[0,1],'--b')
    plt.plot([0,1],.5*np.ones(2),'--b')

    FigDroc = plt.figure()
    plt.title('Injection Classifier Quantification',fontsize=18)
    plt.xlabel('Jitter interval length (ms)',fontsize=18)
    plt.ylabel('Area under the ROC curve',fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.plot(interval_true*np.ones(2),[np.amin(area),np.amax(area)],'--r')
    plt.plot(delta_range,area,'o-k')
    plt.show()