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

def analyze(parameters,weight_value,train_ref,train_targ):
    '''
    Inputs:
        parameters: Generative simulation parameters
        weight_value: Simulated monosynaptic weight values
        train_ref: Reference train set 
        train_targ: Target train set          
    Outputs:
        Figure: Relationship between the estimate and the true value
    '''

    #--------------------------------------------------------------------------
    # Extract the spike data
    #--------------------------------------------------------------------------

    Ntrial = int(parameters[0])  # Number of pairs
    duration = parameters[1]     # Trial duration in (ms)
    period = parameters[2]       # Nonstationarity timescale in (ms)
    Fs = parameters[3]           # Sampling frequency
    Nphase = int(parameters[4])
    phase = duration/Nphase      # Trial duration per weight value in (ms)

    train = np.append(train_ref,train_targ) 
    cell = np.int64(np.append(np.zeros(len(train_ref)),
                               np.ones(len(train_targ))))

    #--------------------------------------------------------------------------
    # Predict the weight based on the model
    #--------------------------------------------------------------------------

    # Measure the distribution of synchrony count before injection
    synch_width = 5.
    Tref = synch_width*np.floor(train_ref/synch_width)
    Ttarg = synch_width*np.floor(train_targ/synch_width)
    Tsynch = np.array(list(set(Tref) & set(Ttarg)))
    synch_count = np.bincount(np.int64(np.floor(Tsynch/(Ntrial*phase))),
                              minlength=Nphase)

    # Estimate the excess synchrony count using the formula
    Nperiod = int(Ntrial*duration/period)
    count_ref = np.bincount(np.int64(np.floor(train_ref/period)),
                            minlength=Nperiod)
    count_targ = np.bincount(np.int64(np.floor(train_targ/period)),
                             minlength=Nperiod)
    count_synch = np.bincount(np.int64(np.floor(Tsynch/period)),
                              minlength=Nperiod)
    Nperiod_phase = int(Ntrial*phase/period)
    RS_prod = np.sum(np.reshape(count_ref*count_synch,(Nphase,Nperiod_phase)),
                     axis=1)
    alpha = RS_prod/(period*synch_count)  
    RT_prod = np.sum(np.reshape(count_ref*count_targ,(Nphase,Nperiod_phase)),
                     axis=1)
    alphaN = alpha[~np.isnan(alpha)]
    synch_countN = synch_count[~np.isnan(alpha)]
    RT_prodN = RT_prod[~np.isnan(alpha)]
    estimate = (synch_countN-RT_prodN/period)/(1-alphaN)

    #--------------------------------------------------------------------------
    # Assess the model's prediction
    #--------------------------------------------------------------------------

    # Quantify the prediction's performance
    x = weight_value
    y = estimate
    gradient,intercept,r_value,p_value,std_err = stats.linregress(x,y)
    print("(Linear regression) Correlation coefficient: ",r_value,
          "p-value: ",p_value)

    # Represent the result
    FigE = plt.figure()
    plt.title('Model Prediction Assessment',fontsize=18)
    plt.xlabel('True synaptic weight',fontsize=18)
    plt.ylabel('Excess synchrony estimate',fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)   
    plt.plot(x,y,'ok')
    plt.plot(x,gradient*x+intercept,'-r')
    plt.show()