import numpy as np
from astropy.table import Table
from astropy import units as u
from sedfitter.sed import SEDCube
from model_track import model_track
from tqdm import tqdm
from glob import glob
import os

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
wavelengths = np.logspace(-2,np.log10(5000),200)*u.um
history = 'is'
dist = 'none'
nbins = 15

if history == 'is':
    filedir = 'isothermal'
elif history == 'tc':
    filedir = 'turbulentcore'

files = glob(f'{datapath}/protostar_tracks/{filedir}/*.txt')
basenames = [os.path.basename(f).split('=')[-1] for f in files]
masses = np.sort(np.array([float(name.split('.')[0]+'.'+name.split('.')[1]) for name in basenames]))

trackdict = {}
seddict = {}
incdict = {}

for mass in masses:
    print(f'Setting up mf = {np.round(mass,3)}...')
    table = Table()
    for g in geometries:
        seds = SEDCube.read(f'{datapath}/../flux/robitaille_models-1.2/{g}/flux.fits')
        evol,incs = model_track(mass,g,history,ap=ap,norm=dist,return_distance=True)
        trackdict.update({g:evol})
        seddict.update({g:seds})
        incdict.update({g:incs})
    time = evol['Time'][::incs]
    smooth_time = smooth_bin(time,nbins)
    table.add_column(time,name='Time')

    print('Retrieving fluxes...')
    flux_blocks = []
    for step in tqdm(range(len(time))):
        dists = [trackdict[g]['Metric Distance'][step*incdict[g]] for g in geometries]
        vals = []
        for g in np.array(geometries)[dists < 10*min(dists)]:
            for i in range(incdict[g]):
                sed = seddict[g].get_sed(trackdict[g]['MODEL_NAME'][step*incdict[g]+i])
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
        table.add_column(final_wav,name=f'{wavelengths[wav].to(u.um).value}-micron flux')

    table.write(f'{datapath}/flux_tracks/{filedir}/{dist}/mf={mass}.fits',overwrite=True)
    print('Done.')
