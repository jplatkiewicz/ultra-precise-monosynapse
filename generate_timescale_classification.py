'''
Generate a collection of spike train pairs
using two unconnected spiking neurons. 

Jonathan Platkiewicz, 2017-11-26
'''

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from brian2 import *
from scipy import interpolate
from ccg import correlograms


def generate(Ntrial,duration,period):
    '''
    Inputs:
        Ntrial: Number of pairs
        duration: Trial duration in (ms)
        period: Duration of the piecewise constant intervals in (ms)
    Outputs:
        train_ref0: Collection of reference spike trains
        train_targ0: Collection of target spike trains
        params: Simulation parameters values
    '''

    #--------------------------------------------------------------------------
    # Define model parameters 
    #--------------------------------------------------------------------------

    # Simulation parameters
    time_step = 0.1                  
    defaultclock.dt = time_step*ms  # Time step of equations integration 
    Fs = 1/(time_step*.001)          

    # Neuron parameters
    cm = 250*pF               # Membrane capacitance
    gm = 25*nS                # Membrane conductance
    tau = cm/gm               # Membrane time constant
    El = -70*mV               # Resting potential
    Vt = El+20*mV             # Spike threshold
    Vr = El+10*mV             # Reset voltage
    refractory_period = 0*ms  # Refractory period
    print("Spike threshold: ",Vt/mV,"(mV)",
          "Refractory period: ",refractory_period/ms,"(ms)")

    # Background input parameters
    tauI = 10*ms       # Auto-correlation time constant
    sigmaI = 1.*mvolt  # Noise standard-deviation 
    muI = Vt-.5*mV
    xmin = muI-.5*mV   # Minimal amplitude of the nonstationary input 
    xmax = muI+.5*mV   # Maximal amplitude
    print("background input time constant: ",tauI/ms,"(ms)",
          "Input average amplitude: ",muI/mV,"(mV)",
          "Input amplitude range:",.1*floor((xmax-xmin)/mV/.1),"(mV)",
          "Input standard-deviation",sigmaI/mV,"(mV)")

    #--------------------------------------------------------------------------
    # Define model  
    #--------------------------------------------------------------------------

    # Define neurons equations
    # -- Reference neuron
    eqs_ref = Equations('''                
    dV/dt = (-V+mu+sigmaI*I)/tau : volt 
    dI/dt = -I/tauI+(2/tauI)**.5*xi : 1
    mu : volt
    ''')
    # -- Target neuron
    eqs_targ = Equations('''
    dV/dt = (-V+mu+sigmaI*I)/tau : volt 
    dI/dt = -I/tauI+(2/tauI)**.5*xi : 1
    mu : volt (linked)
    ''')

    # Constrain the model
    reference = NeuronGroup(Ntrial,model=eqs_ref,threshold='V>Vt',reset='V=Vr',
                            refractory=refractory_period,method='euler')
    target = NeuronGroup(Ntrial,model=eqs_targ,threshold='V>Vt',reset='V=Vr',
                         refractory=refractory_period,method='euler')
    reference.run_regularly('''mu = xmin+(xmax-xmin)*rand()''',dt=period*ms)
    target.mu = linked_var(reference,'mu')

    # Initialize variables
    reference.V = (Vt-.1*mV-Vr)*rand(Ntrial)+Vr
    reference.I = 2*rand(Ntrial)-1
    reference.mu = xmin+(xmax-xmin)*rand(Ntrial)
    target.V = (Vt-.1*mV-Vr)*rand(Ntrial)+Vr
    target.I = 2*rand(Ntrial)-1

    # Record variables
    Sref = SpikeMonitor(reference) 
    Starg = SpikeMonitor(target)

    # Integrate equations
    run(duration*ms)

    #--------------------------------------------------------------------------
    # Check the resulting spike trains 
    #--------------------------------------------------------------------------

    # Organize the collection of spike train pairs into two long spike trains
    train_ref0 = unique(Sref.i*duration+Sref.t/ms)
    train_targ0 = unique(Starg.i*duration+Starg.t/ms)
    train0 = append(train_ref0,train_targ0)
    cell0 = int64(append(zeros(len(train_ref0)),ones(len(train_targ0))))

    # Basic statistical measure of firing 
    print("Reference train: # spikes/trial",len(train_ref0)/Ntrial*1.,
          "firing rate",len(train_ref0)/(Ntrial*duration*.001),"(Hz)",
          "CV",std(diff(train_ref0))/mean(diff(train_ref0)))
    print("Target train: # spikes/trial",len(train_targ0)/Ntrial*1.,
          "firing rate",len(train_targ0)/(Ntrial*duration*.001),"(Hz)",
          "CV",std(diff(train_targ0))/mean(diff(train_targ0)))

    # Compute the correlogram matrix between the two long trains
    lagmax = 100.  # Correlogram window in (ms)
    bine = 1.      # Correlogram time bin in (ms)
    ind_sort = np.argsort(train0)
    st = train0[ind_sort]*.001
    sc = cell0[ind_sort]
    Craw = correlograms(st,sc,sample_rate=Fs,bin_size=bine/1000.,
                        window_size=lagmax/1000.)
    lag = (np.arange(len(Craw[0,1]))-len(Craw[0,1])/2)*bine

    # Represent the auto- and the cross-correlograms
    FigACG = figure()
    title('Auto-correlograms',fontsize=18)
    xlim(-lagmax/2.,lagmax/2.)
    xlabel('Time lag  (ms)',fontsize=18)
    ylabel('Firing rate (Hz)',fontsize=18)
    xticks(fontsize=18)
    yticks(fontsize=18)
    plot(lag,Craw[0,0]/(len(train_ref0)*bine*.001),'.-k')
    plot(lag,Craw[1,1]/(len(train_targ0)*bine*.001),'.-b')
    FigCCG = figure()
    xlim(-lagmax/2.,lagmax/2.)
    title('Cross-correlogram',fontsize=18)
    xlabel('Time lag  (ms)',fontsize=18)
    ylabel('Firing rate (Hz)',fontsize=18)
    xticks(fontsize=18)
    yticks(fontsize=18)
    plot(lag,Craw[0,1]/(len(train_ref0)*bine*.001),'.-k')
    #show()

    # Save the relevant model parameters and the resulting spike trains
    parameters = np.array([Ntrial,duration,period,Fs])
    
    return parameters,train_ref0,train_targ0