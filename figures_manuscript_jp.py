'''
Functions to plot each figure of the manuscript. 

Jonathan Platkiewicz, 2017-11-26
'''

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

import generate_trains_monosynapse
import analyze_trains_monosynapse
import generate_timescale_classification
import analyze_timescale_classification 

def plot_figure(figure):
    '''
    Input:
        figure: Figure number in the manuscript to plot
    '''
    if figure == 1:  
        #-----------------------------------------------------------------------
        # Figure: Excess synchrony estimate - True synaptic weight relationship
        #-----------------------------------------------------------------------

        # Generate the spike data
        p,Tr0,Tt0,Tr,Tt = generate_trains_monosynapse.generate(Ntrial=100,
                                                               duration=300.)
        # Analyze the spike data
        analyze_trains_monosynapse.analyze(p,Tr0,Tt0,Tr,Tt)        
    elif figure == 2:    
        #-----------------------------------------------------------------------
        # Figure: Jitter interval length as classification parameter
        #-----------------------------------------------------------------------

        # Generate the spike data
        p,Tr ,Tt0 = generate_timescale_classification.generate(Ntrial=100,
                                                               duration=300.,
                                                               period=50.)
        # Analyze the spike data
        analyze_timescale_classification.analyze(p,Tr0,Tt0)
