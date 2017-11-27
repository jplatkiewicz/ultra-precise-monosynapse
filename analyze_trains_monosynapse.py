'''
Estimate injected synchrony count.
Relate it to the monosynaptic peak conductance.

Jonathan Platkiewicz, 2017-11-26
'''

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import numpy as np
from scipy import stats,interpolate
from ccg import correlograms
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# Define function
#------------------------------------------------------------------------------

def analyze(parameters,train_ref0,train_targ0,train_ref,train_targ):
    '''
    Inputs:
        parameters: Generative simulation parameters
        train_ref0: Reference train set (synapse off)
        train_targ0: Target train set (synapse off) 
        train_ref: Reference train set (synapse on)
        train_targ: Target train set (synapse on)         
    Outputs:
        Figure: Relationship between the estimate and the true value
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
    train = np.append(train_ref,train_targ) 
    cell = np.int64(np.append(np.zeros(len(train_ref)),
                               np.ones(len(train_targ))))

    # Measure the distribution of synchrony count before injection
    synch_width = 1.*5
    #--WITH SYNAPSE--
    Tref = synch_width*np.floor(train_ref/synch_width)
    lmax = lag[np.argmax(Craw[0,1])]
    x = (train_targ-lmax)*(np.sign(train_targ-lmax)+1)/2.
    x = x[np.nonzero(x)]
    Ttarg = synch_width*np.floor(train_targ/synch_width)
    Tsynch = np.array(list(set(Tref) & set(Ttarg)))
    synch_count = np.bincount(np.int64(np.floor(Tsynch/(Ntrial*phase))),
                              minlength=Nphase)
    #--WITHOUT SYNAPSE--
    Tref0 = synch_width*np.floor(train_ref0/synch_width)
    lmax0 = lag[np.argmax(Craw0[0,1])]
    x = (train_targ0-lmax0)*(np.sign(train_targ0-lmax0)+1)/2.
    x = x[np.nonzero(x)]
    Ttarg0 = synch_width*np.floor(x/synch_width)
    Tsynch0 = np.array(list(set(Tref0) & set(Ttarg0)))
    synch_count0 = np.bincount(np.int64(np.floor(Tsynch0/(Ntrial*phase))),
                               minlength=Nphase)

    # Excess synchrony count unbiased estimation
    delta = period
    Ndelta = int(Ntrial*duration/delta)
    count_ref = np.bincount(np.int64(np.floor(train_ref/delta)),minlength=Ndelta)
    count_targ = np.bincount(np.int64(np.floor(train_targ/delta)),minlength=Ndelta)
    count_synch = np.bincount(np.int64(np.floor(Tsynch/delta)),minlength=Ndelta)
    Ndelta_phase = int(Ntrial*phase/delta)
    RS_prod = sum(np.reshape(count_ref*count_synch,(Nphase,Ndelta_phase)),axis=1)
    alpha = RS_prod/(delta*synch_count)  
    RT_prod = sum(np.reshape(count_ref*count_targ,(Nphase,Ndelta_phase)),axis=1)
    alphaN = alpha[~np.isnan(alpha)]
    synch_countN = synch_count[~np.isnan(alpha)]
    RT_prodN = RT_prod[~np.isnan(alpha)]
    estimate = (synch_countN-RT_prodN/delta)/(1-alphaN)

    # Check the result
    x = g0/gm*weight_value
    y = estimate
    gradient,intercept,r_value,p_value,std_err = stats.linregress(x,y)
    print("(Linear regression) Correlation coefficient: ",r_value,
          "p-value: ",p_value)

    FigE = plt.figure()
    plt.plot(x,y,'ok')
    plt.plot(x,gradient*x+intercept,'-r')
    plt.show()