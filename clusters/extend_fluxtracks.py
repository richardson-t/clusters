import numpy as np
from astropy.table import Table
from astropy import constants,units as u
from mysg.atmosphere import interp_atmos
from glob import glob

history = 'tc'
dist = 'metric'

if history == 'is':
    directory = 'isothermal'
    maxtime = 78*u.Myr
elif history == 'tc':
    directory = 'turbulentcore'
    maxtime = 1.354*u.Myr

proto_tracks = glob(f'protostar_tracks/{directory}/*.txt')
masses = [float(f.split('=')[1].split('.')[0]+f.split('=')[1].split('.')[1]) for f in files]
proto_tracks = proto_tracks[np.argsort(masses)][:-1]

flux_tracks = glob(f'flux_tracks/{directory}/{dist}/*.fits')
masses = [float(f.split('=')[1].split('.')[0]+f.split('=')[1].split('.')[1]) for f in files]
flux_tracks = flux_tracks[np.argsort(masses)][:-1]

model_wav = np.logspace(np.log10(0.01),np.log10(5000),200)

for n in range(len(flux_tracks)):
    proto = Table.read(proto_tracks[n],format='ascii')
    flux = Table.read(flux_tracks[n])
    
    acc_index = np.argmin(proto['Accretion_Rate'][proto['Stellar_Mass'] > 0])
    max_index = np.argmin(abs(proto['Time']-maxtime.value))
    sample_indices = np.linspace(acc_index,max_index,5).astype(int)
    
    for i in sample_indices:
        new_row = [proto['Time'][i]]
        rstar = proto['Stellar_Radius'][i]*u.R_star; tstar = proto['Stellar_Temperature'][i]

        nu,fnu = interp_atmos(tstar); nu = nu[::-1]*u.Hz
        wav = nu.to(u.um,equivalencies=u.spectral())
        fwav = np.interp(model_wav,wav.value,fnu[::-1])*u.erg/u.s/u.cm**2/u.Hz
        fwav = (fwav*(rstar/(1*u.kpc))**2).to(u.mJy)
        new_row.extend([val for val in fwav.value])

        flux.add_row(new_row)
    
    flux.write(flux_tracks[n],overwrite='True')
