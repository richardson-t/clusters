import numpy as np
from imf.imf import make_cluster
from astropy import units as u
from interpolation import interp_flux
from util import setup_templates,filter_flux,scale_distance
import matplotlib.pyplot as plt
from tqdm import tqdm

wav = np.logspace(-2,np.log10(5000),200)*u.um

#If the start times have been offset from zero, make sure the first star starts at zero whie preserving offsets
def zero_start(fluxdict):
    times = [np.min(tbl['Time']) for tbl in fluxdict.values()]
    mintime = np.min(times)
    for tbl in fluxdict.values():
        tbl['Time'] -= mintime

#Align the tracks such that accretion ends at the same time
def align_end(fluxdict):
    times = [np.max(tbl['Time'][-1]) for tbl in fluxdict.values()]
    maxtime = np.max(times)
    for tbl in fluxdict.values():
        tbl['Time'] += maxtime-tbl['Time'][-1]

#Add a random number pulled from a uniform distribution from zero to maxtime to the time of each track.
def add_random(fluxdict,maxtime):
    rng = np.random.default_rng()
    for tbl in fluxdict.values():
        tbl['Time'] += rng.uniform(0,maxtime,1)[0]

#Add a random number pulled from a normal distribution with a 1-sigma of interval to the time of each track.
def add_normal_random(fluxdict,interval):
    rng = np.random.default_rng()
    for tbl in fluxdict.values():
        tbl['Time'] += rng.normal(0,interval,1)[0]

def offset_time(fluxdict,method,scale=1*u.Myr,times=None):
    if method == 'custom':
        assert np.all(np.isfinite(times))
    if 'start' in method:
        if method == 'randomstart':
            add_normal_random(fluxdict,scale.value)
            zero_start(fluxdict)
    elif 'end' in method:
        align_end(fluxdict)
        if method == 'randomend':
            add_normal_random(fluxdict,scale.value)
            zero_start(fluxdict)
    elif method == 'random':
        add_random(fluxdict,scale.value)
        zero_start(fluxdict)

def construct(mtot,history,dist,method,timescale=0.2*u.Myr,imf='kroupa',stop_criterion='nearest'):
    masses = make_cluster(mtot,mmin=0.03,mmax=120,massfunc=imf,stop_criterion=stop_criterion)
    masses = masses[np.argsort(masses)]
    masses = masses[masses >= 0.2]
    print(f'Sampled {len(masses)} stars. Retrieving fluxes...')
    info = setup_templates(history,dist)
    flux_history = {}
    for m in tqdm(masses):
        flux_history.update({m:interp_flux(m,history,dist,*info)})
    offset_time(flux_history,method,scale=timescale)
    times = []
    for tbl in flux_history.values():
        times.append(tbl['Time'][-1])
    max_time = np.max(times)
    print(f'Accretion ends at t={max_time} Myr')
    return flux_history,max_time

def sample(flux_history,wavelength,time,distance=1*u.kpc):
    fluxes = []
    for tbl in flux_history.values():
        time_index = np.argmin(abs(time-tbl['Time']))
        tbl_flux = tbl[time_index]
        wav_index = np.searchsorted(wav,wavelength)
        frac = (wavelength-wav[wav_index-1])/(wav[wav_index]-wav[wav_index-1])
        fluxes.append((frac*tbl[f'{wav[wav_index-1].value}-micron flux'][time_index]+(1.-frac)*tbl[f'{wav[wav_index].value}-micron flux'][time_index]).value)
    return (scale_distance(np.array(fluxes),distance)).value

def cmd(flux_history,filter1,filter2,time,distance=1*u.kpc):
    inst1,cam1 = filter1.split('/'); inst2,cam2 = filter2.split('/')
    colors = []
    mags = []
    for tbl in flux_history.values():
        time_index = np.argmin(abs(time-tbl['Time']))
        tbl_flux = (np.array([val for val in tbl[time_index]])[1:])*u.mJy
        flux1,zeropoint1 = filter_flux(wav,tbl_flux,inst1,cam1)
        flux2,zeropoint2 = filter_flux(wav,tbl_flux,inst2,cam2)
        mag1 = -2.5*np.log10(flux1/zeropoint1)
        mag2 = -2.5*np.log10(flux2/zeropoint2)
        colors.append(mag1-mag2); mags.append(mag2)

    colors = np.array(colors); mags = np.array(mags)
    detectable = mags < 30
    colors = colors[detectable]; mags = mags[detectable]

    masses = np.array([float(key) for key in flux_history.keys()])

    plt.figure()
    plt.scatter(colors,mags,c=np.log10(masses[detectable]),cmap='viridis',marker='.')
    plt.gca().invert_yaxis()
    cb = plt.colorbar()
    cb.set_label(r'log Mass ($M_\odot$)')
    plt.title(f'{mtot} Msun cluster, IMF = {imf}, method = {method}')
    plt.xlabel(f'|{cam1}|-|{cam2}|')
    plt.ylabel(f'|{cam2}|')
    plt.savefig(f'plots/cmds/{cam1}_{cam2}_{imf}_{method}_{np.round(time,1)}.pdf')
    plt.close()

mtot = 1000
history = 'tc'
dist = 'metric'
imf = 'kroupa'
stop_criterion = 'nearest'
method = 'randomstart'

filter1 = 'JWST/NIRCam.F444W'
filter2 = 'JWST/NIRCam.F470N'

cluster,max_time = construct(mtot,history,dist,method,imf=imf,stop_criterion=stop_criterion)

sample_times = np.linspace(0,max_time,5)
samples = []
for i in range(len(sample_times)):
    fluxes = sample(cluster,1000*u.um,sample_times[i])
    samples.append(fluxes)
    cmd(cluster,filter1,filter2,sample_times[i])

'''
plt.figure()
plt.boxplot(samples)
plt.yscale('log')
plt.title(f'{mtot} Msun cluster, IMF = {imf}, random start')
plt.xlabel('Time (Myr)'); plt.ylabel('1-mm Flux (mJy)')
locs,labels = plt.xticks()
plt.xticks(locs,np.round(sample_times,2))
plt.savefig(f'plots/box_1mm_{imf}_{method}.pdf')
plt.close()
'''

'''
class Cluster(object):
    
    def __init__(self, m_tot, massfunc='kroupa', history='tc',norm='metric'):
        """
        Create a cluster with total mass m_tot distributed according to the provided IMF (options are 'kroupa' (default), 'chabrier', and 'salpeter'.)
        """
        self.m_tot = m_tot
        self.massfunc = massfunc
        self.history = history
        self.norm = norm
'''