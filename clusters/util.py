import numpy as np
from astropy.table import Table
from astropy import units as u
from astropy.modeling.models import BlackBody
from astroquery.svo_fps import SvoFps
from dust_emissivity.dust import kappa
from glob import glob
import os

def setup_templates(history,dist):
    datapath = os.path.dirname(__file__)
    if history == 'is':
        directory = 'isothermal'
    elif history == 'tc':
        directory = 'turbulentcore'
    elif history == 'ca':
        directory = 'competitive'
    else:
        raise ValueError('Accretion history not implemented')

    ev_files = glob(f'{datapath}/protostar_tracks/{directory}/*.txt')
    masses = [float(f.split('=')[-1].split('.')[0]+'.'+f.split('=')[-1].split('.')[1]) for f in ev_files]; masses = np.array(masses)
    indices = np.argsort(masses); masses = masses[indices]
    ev_tracks = {masses[pair[0]]:Table.read(ev_files[pair[1]],format='ascii') for pair in enumerate(indices)}
    flux_tracks = {masses[i]:Table.read(f'{datapath}/flux_tracks/{directory}/{dist}/mf={masses[i]}.fits') for i in range(len(masses))}

    # last timestep + temperature in which the star is still accreting
    last_times = {mass: tbl['Time'][tbl['Stellar_Mass'] == tbl['Stellar_Mass'][-1]][0] for mass,tbl in ev_tracks.items()}
    last_temps = {mass: tbl['Stellar_Temperature'][tbl['Stellar_Mass'] == tbl['Stellar_Mass'][-1]][0] for mass,tbl in ev_tracks.items()}
    return masses,ev_tracks,flux_tracks,last_times,last_temps

#radiation from the prestellar core prior to source ignition
def dust_sphere(mf,wav,distance=1*u.kpc):
    M = mf*3*u.M_sun
    bb = BlackBody(10*u.K); sed = bb(wav)*u.sr
    initial_d = 1*u.kpc
    F = (M*kappa(wav)*sed/initial_d**2).to(u.mJy)
    F = scale_distance(F,distance)
    return F

def geo_inc(geo):
    if (geo[2:4] == 'p-') or geo[:4] == 's---':
        return 1
    else:
        return 9

def scale_distance(fnu,d,initial_d=1*u.kpc):
    return fnu*(initial_d.to(u.kpc)/d.to(u.kpc))**2

def filter_flux(wav,sed,instrument,camera):
    filter_info = SvoFps.get_transmission_data(f'{instrument}/{camera}')
    filter_wav = (filter_info['Wavelength']).to(u.um)
    filter_response = filter_info['Transmission']
    interp_flux = np.interp(filter_wav,wav,sed)
    avresponse = (filter_response[:-1]+filter_response[1:])/2
    vals = interp_flux*filter_response; vals = (vals[:1]+vals[:-1])/2
    dlambda = filter_wav[1:]-filter_wav[:-1]
    flux = np.sum(vals*dlambda)/np.sum(avresponse*dlambda*u.um)

    table = SvoFps.get_filter_list(instrument)
    zeropoint = table['ZeroPoint'][table['filterID'] == f'{instrument}/{camera}'][0]*u.Jy
    return flux,zeropoint

def get_mass(table,row,ap,key):
    return np.nanmax(table[key][row,:ap])

def norm_check(norm):
    approved_norms = ['none','metric','quant']
    if norm in approved_norms:
        pass
    else:
        raise KeyError('Norm argument not recognized')
