import numpy as np
from astropy.table import Table
from astropy import units as u
from sedfitter.sed import SEDCube
from sklearn import preprocessing
from scipy.spatial import KDTree

from tqdm import tqdm
from glob import glob
import os

from track_functions import metric_distance,fullgrid_seedling
from util import geo_inc

def nearby_models(T_source,L_source,M_core,n_neighbors=10,norm='none'):
    if norm == 'metric':
        mmod = np.array([np.log10(T_source)/norms[0],np.log10(L_source)/norms[1],np.log10(M_core)/norms[2]])
        dists,locs = metric_distance(mmod[:,None],fulldat,n_neighbors,norm)
        dists,locs = dists[0],locs[0]
    else:
        mmod = np.array([np.log10(T_source),np.log10(L_source),np.log10(M_core)])
        mmod = mmod[None,:]
        if norm == 'quant':
            mmod = transformer.transform(mmod)
        kt = KDTree(fulldat)
        dists,locs = kt.query(mmod,n_neighbors)
        dists,locs = dists[0],locs[0]

    selpars = valid_pars[locs]
    names = [f'''{selpars['Geometry'][idx]}/{selpars['Model Name'][idx]}''' for idx in range(len(selpars))]
    return names

def sort_by_wavelength(vals,nwav):
    sort = []
    for i in range(nwav):
        sort.append(np.array([v[i] for v in vals]))
    return sort

def smooth_bin(data,nbins):
    smooth_data = np.zeros(nbins)
    bin_indices = np.linspace(0,len(data)-1,nbins+1).astype(int)
    for i in range(nbins):
        try:
            bin_data = np.concatenate(data[bin_indices[i]:bin_indices[i+1]])
        except(ValueError):
            bin_data = data[bin_indices[i]:bin_indices[i+1]]
        smooth_data[i] = np.nanmedian(bin_data)
    return smooth_data

datapath = os.path.dirname(__file__)

geometries = ['s-p-hmi','s-p-smi','s-pbhmi','s-pbsmi','s-u-hmi','s-u-smi',
              's-ubhmi','s-ubsmi','spu-hmi','spu-smi','spubhmi','spubsmi']
ap = 10
wavelengths = np.logspace(-2,3+np.log10(5),200)*u.um
history = 'tc'
dist = 'quant'
q = 3
nbins = 15

if history == 'is':
    filedir = 'isothermal'
elif history == 'tc':
    filedir = 'turbulentcore'
elif history == 'ca':
    filedir = 'competitive'

if dist == 'none':
    fulldat, valid_models, valid_pars = fullgrid_seedling()
elif dist == 'metric':
    fulldat, valid_models, valid_pars, norms = fullgrid_seedling(norm=dist)
elif dist == 'quant':
    fulldat, valid_models, valid_pars, transformer = fullgrid_seedling(norm=dist)

files = glob(f'protostar_tracks/{filedir}/*.txt')
files.sort()

#basenames = [os.path.basename(f).split('=')[-1] for f in files]
#masses = np.sort(np.array([float(name.split('.')[0]+'.'+name.split('.')[1]) for name in basenames]))

seddict = {}
incdict = {}

print('Loading SED tables...')
for g in geometries:
    seds = SEDCube.read(f'/blue/adamginsburg/richardson.t/research/flux/robitaille_models-1.2/{g}/flux.fits')
    seddict.update({g:seds})
    incdict.update({g:geo_inc(g)})

for f in files:
    basename = os.path.basename(f).split('=')[-1]
    mass = float(basename.split('.')[0]+'.'+basename.split('.')[1])
    track = Table()
    
    print(f'Setting up mf = {np.round(mass,3)}...')
    names = []
    evol = Table.read(f,format='ascii')
    valid_times = np.logical_and(mass-evol['Stellar_Mass'] > 0,evol['Intrinsic_Lum'] > 0)
    evol = evol[valid_times]

    time = evol['Time']
    for step in tqdm(range(len(time))):
        T = evol['Stellar_Temperature'][step]
        L = evol['Intrinsic_Lum'][step]
        M = q*(mass-evol['Stellar_Mass'][step])
        names.append(nearby_models(T,L,M))

    smooth_time = smooth_bin(time,nbins)
    track.add_column(time,name='Time')

    print('Retrieving fluxes...')
    flux_blocks = []
    for array in tqdm(names):
        vals = []
        for name in array:
            this_geo, this_model = name.split('/')
            if incdict[this_geo] == 1:
                sed = seddict[this_geo].get_sed(f'{this_model}_01')
                f = np.interp(wavelengths,sed.wav[::-1],sed.flux[ap,::-1].value)
                for inc in range(9):
                    vals.append(f)
            else:
                for inc in range(incdict[this_geo]):
                    sed = seddict[this_geo].get_sed(f'{this_model}_0{inc+1}')
                    f = np.interp(wavelengths,sed.wav[::-1],sed.flux[ap,::-1].value)
                    vals.append(f)
        vals = sort_by_wavelength(vals,len(wavelengths))
        flux_blocks.append(vals)

    print('Calculating trajectories...')
    for wav in range(len(wavelengths)):
        f_wav = []
        for step in range(len(time)):
            f_wav.append(flux_blocks[step][wav])
        f_wav = smooth_bin(f_wav,nbins)
        final_wav = np.interp(time,smooth_time,f_wav)
        track.add_column(final_wav,name=f'{wavelengths[wav].to(u.um).value}-micron flux')

    track.write(f'flux_tracks/{filedir}/{dist}/mf={mass}.fits',overwrite=True)
    print('Done.')
