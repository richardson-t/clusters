import numpy as np
from astropy.table import Table
from astropy import constants,units as u
from sedfitter.sed import SEDCube
from scipy.spatial import KDTree
from sklearn import preprocessing
from util import *

model_dir = '/blue/adamginsburg/richardson.t/research/flux/robitaille_models-1.2'

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

def seedling(dimensions,norm='none'):
    if norm == 'metric':
        norms = [lin_range(array) for array in dimensions]
        fulldat = np.array([dimensions[i]/norms[i] for i in range(len(dimensions))])

    else:
        fulldat = np.array([array for array in dimensions])

    valid_models = np.all(np.isfinite(fulldat), axis=0)
    fulldat = fulldat[:,valid_models]

    if norm == 'metric':
        return fulldat,valid_models,norms
    else:
        return fulldat,valid_models

def fullgrid_seedling(mass_ap=10,norm='none'):
    pars = Table.read(f'{model_dir}/allgeos_props.fits')
    temp = pars['star.temperature']
    lum = pars['Model Luminosity']
    env_masses = []
    for row in range(len(pars)):
        env_masses.append(get_mass(pars,row,mass_ap,'Sphere Masses'))
    env_masses = np.array(env_masses)

    if norm == 'metric':
        fulldat, valid_models, norms = seedling([np.log10(temp),np.log10(lum),np.log10(env_masses)],norm=norm)
        valid_pars = pars[valid_models]
        return fulldat, valid_models, valid_pars, norms
    else:
        fulldat, valid_models = seedling([np.log10(temp),np.log10(lum),np.log10(env_masses)])
        fulldat = fulldat.T
        valid_pars = pars[valid_models]
        if norm == 'quant':
            transformer = preprocessing.QuantileTransformer()
            fulldat = transformer.fit_transform(fulldat)
            return fulldat, valid_models, valid_pars, transformer
        else:
            return fulldat, valid_models, valid_pars

def nearby_models(T_source,L_source,M_core,n_neighbors=10,mass_ap=10,norm='none'):
    norm_check(norm)

    if norm == 'metric':
        fulldat, valid_models, valid_pars, norms = fullgrid_seedling(norm=norm)
        mmod = np.array([np.log10(T_source)/norms[0],np.log10(L_source)/norms[1],np.log10(M_core)/norms[2]])
        dists,locs = metric_distance(mmod[:,None],fulldat,n_neighbors,norm)
        dists,locs = dists[0],locs[0]
    else:
        mmod = np.array([np.log10(T_source),np.log10(L_source),np.log10(M_core)])
        mmod = mmod[None,:]
        if norm == 'quant':
            fulldat, valid_models, valid_pars, transformer = fullgrid_seedling(norm=norm)
            mmod = transformer.transform(mmod)
        else:
            fulldat, valid_models, valid_pars = fullgrid_seedling(norm=norm)

        kt = KDTree(fulldat)
        dists,locs = kt.query(mmod,n_neighbors)
        dists,locs = dists[0],locs[0]

    selpars = valid_pars[locs]
    names = [f'''{selpars['Geometry'][idx]}/{selpars['Model Name'][idx]}''' for idx in range(len(selpars))]
    return names