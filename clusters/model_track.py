import numpy as np
from astropy.table import Table
from astropy import units as u
from sedfitter.sed import SEDCube
from scipy.spatial import KDTree
from util import *
import os

datapath = os.path.dirname(__file__)

def metric_distance(mmod,fulldat,n_neighbors,norm):
    alpha = 0.02; beta = 0.275; gamma = 0.91
    l = mmod.shape[1]
    loc = np.zeros((l,n_neighbors))
    fulldat = fulldat.T
    for n in range(l):
        distdat = (fulldat-mmod[:,n])**2
        if norm == 'metric':
            distance = np.sqrt(distdat[:,0]**(1/alpha)+distdat[:,1]**(1/beta)+distdat[:,2]**(1/gamma))
        else:
            distance = np.sqrt(distdat[:,0]**alpha+distdat[:,1]**beta+distdat[:,2]**gamma)
        loc[n,:] = np.argsort(distance)[:n_neighbors]
    loc = loc.astype(int)
    return distance[loc],loc

def model_track(mf,name,history,ap=10,n_neighbors=5,tree_index=0,norm=None,return_distance=False):
    approved_histories = ['is','tc','ca']
    approved_norms = [None,'none','log','metric','unnormed_metric']
    if not history in approved_histories:
        raise RuntimeError('Accretion history not implemented')
    if (history == 'is'):
        prefix = 'isothermal'
    if (history == 'tc'):
        prefix = 'turbulentcore'
    if history == 'ca':
        prefix = 'competitive'
    if (tree_index >= n_neighbors):
        raise IndexError('Neighbor must be within n_neighbors of track')
    if not norm in approved_norms:
        if norm == 'none':
            norm = None
        else:
            raise RuntimeError('Norm argument not recognized')

    #model grid stuff

    #build table by model

    powerLaw = (name[2] == 'p')
    hasCavities = (name[3] == 'b')
    incs = 9
    if powerLaw and not hasCavities:
        incs = 1
    seds = SEDCube.read(f'{datapath}/../flux/robitaille_models-1.2/{name}/flux.fits')
    pars = Table.read(f'{datapath}/../flux/robitaille_models-1.2/{name}/augmented_pars.fits')
    pars.add_column(np.arange(0,len(pars)),name='Index')
    reduced_pars = pars[0::incs]
    env_masses = []; key = 'Sphere Masses'
    for row in range(len(reduced_pars)):
        env_masses.append(get_mass(reduced_pars,row,ap,key))
    env_masses = np.array(env_masses)

    #construct tree for queries
   
    if (norm == None) or (norm == 'unnormed_metric'):
        fulldat = np.array((np.log10(reduced_pars['star.temperature']),
                            np.log10(reduced_pars['Model Luminosity']),
                            np.log10(reduced_pars['Sphere Masses'][:,ap])))
    else:
        t_norm, l_norm, m_norm = log_range(reduced_pars['star.temperature']), log_range(reduced_pars['Model Luminosity']), log_range(reduced_pars['Sphere Masses'][:,ap])
        fulldat = np.array((np.log10(reduced_pars['star.temperature'])/t_norm,
                            np.log10(reduced_pars['Model Luminosity'])/l_norm,
                            np.log10(reduced_pars['Sphere Masses'][:,ap])/m_norm))

    valid_models = np.all(np.isfinite(fulldat), axis=0)
    selpars = reduced_pars[valid_models]
    fulldat = fulldat[:,valid_models]
    kt = KDTree(fulldat.T)

    #evolutionary track stuff

    #get time evolution of projected core mass
    q = 3
    mcore_0 = u.Quantity(q*mf,u.M_sun)
    evol = Table.read(f'{datapath}/protostar_tracks/{prefix}/protostellar_evolution_m={mf}.txt',format='ascii')
    rstar = evol['Stellar_Radius']*u.R_sun
    mstar = evol['Stellar_Mass']*u.M_sun
    lstar = evol['Intrinsic_Lum']*u.L_sun
    tstar = evol['Stellar_Temperature']*u.K
    evol.add_column(mcore_0-q*mstar,name='Projected Core Mass')

    #construct array for queries
    if (norm == None) or (norm == 'unnormed_metric'):
        mod = np.array([np.log10(tstar.value), np.log10(lstar.value), np.log10(evol['Projected Core Mass'])])
    else:
        mod = np.array([np.log10(tstar.value)/t_norm, np.log10(lstar.value)/l_norm, np.log10(evol['Projected Core Mass'])/m_norm])
    mmod = mod[:, np.all(np.isfinite(mod), axis=0)]
    mid_evol = evol[np.all(np.isfinite(mod),axis=0)]
    final_evol = Table(names=[key for key in mid_evol.keys()])
    for row in range(len(mid_evol['Time'])):
        for i in range(incs):
            final_evol.add_row(mid_evol[row])

    #time to bring 'em together

    #query and record initial positions of all nearest-neighbor inclinations

    if 'metric' in norm:
        dist,loc = metric_distance(mmod,fulldat,n_neighbors,norm)
    else:
        dist,loc = kt.query(mmod.T, n_neighbors)
    loc_indices = []
    for idx in loc[:,tree_index]:
        pars_index = selpars['Index'][idx]
        loc_indices.extend(np.arange(pars_index,pars_index+incs))
    all_models = pars[loc_indices[:]]
    all_models.remove_column('Index')
    if return_distance:
        dist_column = []
        for d in dist[:,tree_index]:
            dist_column.extend(d*np.ones(incs))
        all_models.add_column(dist_column,name='Metric Distance')

    #construct final table                                                                                                                              
    final_table = Table()
    for key in final_evol.keys():
        final_table.add_column(final_evol[key],name=key)
    for key in all_models.keys():
        final_table.add_column(all_models[key],name=key)
    return final_table,incs
