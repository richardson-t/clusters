import numpy as np
from scipy.stats import kstest
from astropy.table import Table
from cluster import construct

#load data distribution

#set up parameter space
#mass, age, accretion history, IMF, SFH
m_cls = np.logspace(3,5,20)
histories = ['is','tc','ca']
imfs = ['salpeter','kroupa','chabrier']
sfhs = ['start','randomstart','end','randomend'] 

#sample parameter space
#for each set of parameters:
    #make a cluster
    #compare distributions using some method (KS test, etc.)
    #store likelihood value

#find extrema in likelihood, zoom in, iterate on this process

#questions: if this is iterative, how to cut off? can we find/pursue
#multiple extrema (and should we?)
