'''
Generate spike train pairs
using two monosynaptically connected spiking neurons,
and varying the monosynaptic peak conductance.

Jonathan Platkiewicz, 2017-11-26
'''

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from brian2 import *
from scipy import stats
from ccg import correlograms

#------------------------------------------------------------------------------
# Define function
#------------------------------------------------------------------------------

def generate(Ntrial,duration):
    '''
    Inputs:
        Ntrial: Number of trials
        duration: Trial duration in (ms)
    Outputs:
        parameters: Simulation parameters values
        train_ref0: Reference spike trains with synapse off
        train_targ0: Target spike trains with synapse off
        train_ref: Reference spike trains with synapse on
        train_targ: Target spike trains with synapse on 
    '''

    #--------------------------------------------------------------------------
    # Define model parameters 
    #--------------------------------------------------------------------------

    # Simulation parameters
    time_step = 0.1            #-in (ms)
    defaultclock.dt = time_step*ms  
    Fs = 1/(time_step*.001)    #-in (Hz)

    # Neuron parameters
    cm = 250*pF               # membrane capacitance
    gm = 25*nS                # membrane conductance
    tau = cm/gm               # membrane time constant
    El = -70*mV               # resting potential
    Vt = El+20*mV             # spike threshold
    Vr = El+10*mV             # reset value
    refractory_period = 0*ms  # refractory period

    # Background input parameters
    tauI = 10*ms       # Auto-correlation time constant
    sigmaI = 1.*mvolt  # Noise standard-deviation 
    muI = Vt-.5*mV
    xmin = muI-.5*mV   # Minimal amplitude of the nonstationary input 
    xmax = muI+.5*mV   # Maximal amplitude
    period = 50.       # Duration of the piecewise constant intervals in (ms)
    print("background input time constant: ",tauI/ms,"(ms)",
          "Input average amplitude: ",muI/mV,"(mV)",
          "Input amplitude range:",.1*floor((xmax-xmin)/mV/.1),"(mV)",
          "Input standard-deviation",sigmaI/mV,"(mV)",
          "Interval duration: ",period,"(ms)")

    # Monosynapse parameters
    tauS = 3*ms              # Synaptic time constant
    Esyn = 0*mV              # Synaptic reversal potential 
    PSC = 25*pA              # Postsynaptic current ammplitude
    g0 = PSC/(Esyn-muI)
    latency = 1.5*ms         # Spike transmission delay
    Nphase = 10
    phase = duration/Nphase  # Duration of session with fixed synaptic weight 
    wmin = .5                # Minimal synaptic weight
    wmax = 4.                # Maximal synaptic weight
    print("Monosynaptic peak conductance: ",g0/nsiemens,"(siemens)")

    #--------------------------------------------------------------------------
    # Define model  
    #--------------------------------------------------------------------------

    # Define neurons equations
    # -- Reference neuron (synapse turned on)
    eqs_ref = Equations('''
    dV/dt = (-V+mu+sigmaI*I)/tau : volt 
    I : 1 (linked)
    mu : volt
    ''')
    # -- Reference neuron (synapse turned off)  
    eqs_ref0 = Equations('''
    dV/dt = (-V+mu+sigmaI*I)/tau : volt 
    I : 1 (linked)
    mu : volt (linked)
    ''')
    # -- Input noise to reference neuron (same for synapse on/off)
    eqs_refnoise = Equations('''
    dx/dt = -x/tauI+(2/tauI)**.5*xi : 1
    ''') 
    # -- Target neuron (synapse turned on)
    eqs_targ = Equations('''
    dV/dt = (-V+mu+sigmaI*I-g0/gm*gsyn*(V-Esyn))/tau : volt 
    I : 1 (linked)
    mu : volt (linked)
    #-Monosynaptic input
    dgsyn/dt = -gsyn/tauS : 1
    ''')
    # -- Input noise to target neuron (same for synapse on/off) 
    eqs_targnoise = Equations('''
    dx/dt = -x/tauI+(2/tauI)**.5*xi : 1
    ''')

    # 'Synapse on' model
    # -- Constrain the model
    reference = NeuronGroup(Ntrial,model=eqs_ref,threshold='V>Vt',reset='V=Vr',
                            refractory=refractory_period,method='euler')
    target = NeuronGroup(Ntrial,model=eqs_targ,threshold='V>Vt',reset='V=Vr',
                         refractory=refractory_period,method='euler')
    reference.run_regularly('''mu = xmin+(xmax-xmin)*rand()''',dt=period*ms)
    target.mu = linked_var(reference,'mu')
    ref_noise = NeuronGroup(Ntrial,model=eqs_refnoise,threshold='x>10**6',
                            reset='x=0',method='euler')
    targ_noise = NeuronGroup(Ntrial,model=eqs_targnoise,threshold='x>10**6',
                             reset='x=0',method='euler')
    reference.I = linked_var(ref_noise,'x')
    target.I = linked_var(targ_noise,'x')
    # -- Parameter initialization
    reference.V = (Vt-.1*mV-Vr)*rand(Ntrial)+Vr
    target.V = (Vt-.1*mV-Vr)*rand(Ntrial)+Vr
    target.gsyn = 0
    ref_noise.x = 2*rand(Ntrial)-1
    targ_noise.x = 2*rand(Ntrial)-1
    # -- Synaptic connection
    weight_value = np.random.permutation(linspace(wmin,wmax,Nphase))
    weight = TimedArray(weight_value,dt=phase*ms)
    synaptic = Synapses(reference,target,
                 '''w = weight(t) : 1''',
                 on_pre='''
                 gsyn += w
                 ''')
    synaptic.connect(i=arange(Ntrial),j=arange(Ntrial))
    synaptic.delay = latency
    #--Record variables
    Sref = SpikeMonitor(reference)
    Starg = SpikeMonitor(target)
    Msyn = StateMonitor(synaptic,'w',record=0)

    # 'Synapse off' model
    # -- Constrain the model
    reference0 = NeuronGroup(Ntrial,model=eqs_ref0,threshold='V>Vt',reset='V=Vr',
                             refractory=refractory_period,method='euler')
    target0 = NeuronGroup(Ntrial,model=eqs_targ,threshold='V>Vt',reset='V=Vr',
                          refractory=refractory_period,method='euler')
    reference0.mu = linked_var(reference,'mu')
    target0.mu = linked_var(reference,'mu')
    reference0.I = linked_var(ref_noise,'x')
    target0.I = linked_var(targ_noise,'x')
    # -- Parameter initialization
    reference0.V = reference.V
    target0.V = target.V
    target0.gsyn = 0
    #--Record variables
    Sref0 = SpikeMonitor(reference0)
    Starg0 = SpikeMonitor(target0)

    run(duration*ms)

    #--------------------------------------------------------------------------
    # Check the resulting spike trains 
    #--------------------------------------------------------------------------

    # Represent some of the recorded variables
    FigW = figure()
    xlabel('Time (ms)')
    ylabel('Synaptic weight')
    title('Target cell')
    plot(Msyn.t/ms,Msyn.w[0],'k')

    # Organize the spike times into two long spike trains
    # -- Synapse on
    train_ref = sort(Sref.t/ms + floor(Sref.t/(ms*phase))*(-1+Ntrial)*phase 
                     + Sref.i*phase)
    train_targ = sort(Starg.t/ms + floor(Starg.t/(ms*phase))*(-1+Ntrial)*phase
                      + Starg.i*phase)
    train = append(train_ref, train_targ)
    cell = int64(append(zeros(len(train_ref)), ones(len(train_targ))))
    # -- Synapse off
    train_ref0 = sort(Sref0.t/ms + floor(Sref0.t/(ms*phase))*(-1+Ntrial)*phase 
                      + Sref0.i*phase)
    train_targ0 = sort(Starg0.t/ms + floor(Starg0.t/(ms*phase))*(-1+Ntrial)*phase 
                       + Starg0.i*phase)
    train0 = append(train_ref0, train_targ0)
    cell0 = int64(append(zeros(len(train_ref0)), ones(len(train_targ0))))

    # Basic firing parameters
    print("SYNAPSE ON")
    print("-- Reference train:")
    print("# spikes: ",len(train_ref),
          "Average firing rate",len(train_ref)/(Ntrial*duration*1.),
          "CV",std(diff(train_ref))/mean(diff(train_ref)))
    print("-- Target train:")
    print("# spikes: ",len(train_targ),
          "Average firing rate",len(train_targ)/(Ntrial*duration*1.),
          "CV",std(diff(train_targ))/mean(diff(train_targ)))
    print("SYNAPSE OFF")
    print("-- Target train:")
    print("# spikes: ",len(train_targ0),
          "Average firing rate",len(train_targ0)/(Ntrial*duration*1.),
          "CV",std(diff(train_targ0))/mean(diff(train_targ0)))

    # Compute the correlogram matrix between the two long trains
    lagmax = 100.  # Correlogram window in (ms)
    bine = 1.      # Correlogram time bin in (ms)
    #--WITH SYNAPSE--
    ind_sort = np.argsort(train)
    st = train[ind_sort]*.001
    sc = cell[ind_sort]
    Craw = correlograms(st,sc,sample_rate=Fs,bin_size=bine/1000.,
                        window_size=lagmax/1000.)
    lag = (np.arange(len(Craw[0,1]))-len(Craw[0,1])/2)*bine
    #--WITHOUT SYNAPSE--
    ind_sort = np.argsort(train0)
    st = train0[ind_sort]*.001
    sc = cell0[ind_sort]
    Craw0 = correlograms(st,sc,sample_rate=Fs,bin_size=bine/1000.,
                         window_size=lagmax/1000.)

    # Represent the auto- and the cross-correlograms
    FigACG = figure()
    title('Auto-correlograms',fontsize=18)
    xlim(-lagmax/2.,lagmax/2.)
    xlabel('Time lag  (ms)',fontsize=18)
    ylabel('Firing rate (Hz)',fontsize=18)
    xticks(fontsize=18)
    yticks(fontsize=18)
    plot(lag,Craw[0,0]/(len(train_ref)*bine*.001),'.-k')
    plot(lag,Craw[1,1]/(len(train_targ)*bine*.001),'.-b')
    plot(lag,Craw0[1,1]/(len(train_targ0)*bine*.001),'.-c')
    FigCCG = figure()
    xlim(-lagmax/2.,lagmax/2.)
    title('Cross-correlograms',fontsize=18)
    xlabel('Time lag  (ms)',fontsize=18)
    ylabel('Firing rate (Hz)',fontsize=18)
    xticks(fontsize=18)
    yticks(fontsize=18)
    plot(lag,Craw[0,1]/(len(train_ref)*bine*.001),'.-k')
    plot(lag,Craw0[0,1]/(len(train_ref0)*bine*.001),'.-c')
    #show()

    # Save the relevant model parameters and the resulting spike trains
    parameters = np.array([Ntrial,duration,period,Fs,Nphase])

    return parameters,weight_value,train_ref,train_targ