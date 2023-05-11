import numpy as np
from astropy.table import Table
from astropy import units as u
from astropy.modeling.models import BlackBody
from util import dust_sphere

wav = np.logspace(-2,np.log10(5000),200)*u.um

# find rows from the beginning
def find_row(selected_time, tbl):
    times = tbl['Time']
    selected_row = np.argmin(np.abs(times - selected_time))    
    return selected_row

def standardize(m1,m2,ev_tracks,flux_tracks,last_times,interpTime):
    ev1,ev2 = ev_tracks[m1],ev_tracks[m2]
    fx1,fx2 = flux_tracks[m1],flux_tracks[m2]
    t1,t2 = last_times[m1],last_times[m2]

    overlap_12 = np.array([val in fx2['Time'] for val in fx1['Time']])
    overlap_21 = np.array([val in fx1['Time'] for val in fx2['Time']])
    
    #if table 1 starts first, add rows to table 2
    if np.argmin(overlap_12) < np.argmax(overlap_12):
        new_times = fx1['Time'][:np.argmax(overlap_12)][::-1]
        sed = dust_sphere(m2,wav).value
        fx2.reverse()
        for time in range(len(new_times)):
            row_to_add = [new_times[time]]
            row_to_add.extend(sed)
            fx2.add_row(row_to_add)
        fx2.reverse()

        #update overlap
        overlap_12 = np.array([val in fx2['Time'] for val in fx1['Time']])
    
    #else if table 2 starts first, add rows to table 1
    elif np.argmin(overlap_21) < np.argmax(overlap_21):
        new_times = fx2['Time'][:np.argmax(overlap_21)][::-1]
        sed = dust_sphere(m1,wav).value
        fx1.reverse()
        for time in range(len(new_times)):
            row_to_add = [new_times[time]]
            row_to_add.extend(sed)
            fx1.add_row(row_to_add)
        fx1.reverse()

        #update overlap
        overlap_21 = np.array([val in fx1['Time'] for val in fx2['Time']])

    #if table 1 ends last, add rows to table 2
    if np.argmin(overlap_12) > np.argmax(overlap_12):
        new_times = fx2['Time'][~overlap_12]
        last_temp = ev1['Stellar_Temperature'][find_row(t2,ev2)]*u.K
        last_rad = (ev1['Stellar_Radius'][find_row(t2,ev2)]*u.R_sun).to(u.kpc).value
        bb = BlackBody(last_temp); sed = (bb(wav)*u.sr).to(u.mJy)
        sed = (sed*last_rad**2).value
        for time in new_times:
            row_to_add = [time]
            row_to_add.extend(sed)
            fx2.add_row(row_to_add)

    #else if table 2 ends last, add rows to table 1
    elif np.argmin(overlap_21) > np.argmax(overlap_21):
        new_times = fx2['Time'][~overlap_21]
        last_temp = ev1['Stellar_Temperature'][find_row(t1,ev1)]*u.K
        last_rad = (ev1['Stellar_Radius'][find_row(t1,ev1)]*u.R_sun).to(u.kpc).value
        bb = BlackBody(last_temp); sed = (bb(wav)*u.sr).to(u.mJy)
        sed = (sed*last_rad**2).value
        for time in new_times:
            row_to_add = [time]
            row_to_add.extend(sed)
            fx1.add_row(row_to_add)

    return fx1,fx2

def interp_flux(mf,history,dist,masses,ev_tracks,flux_tracks,last_times,last_temps):
    if history == 'is':
        interpTime = True
    elif history == 'tc':
        interpTime = False
    else:
        raise ValueError('Accretion history not implemented')
    
    #masses,ev_tracks,flux_tracks,last_times,last_temps = setup_templates(directory,dist)

    # Retrieve the relevant tables (with modifications for interpolation)
    i = np.searchsorted(masses, mf)
    m1 = masses[i-1]; m2 = masses[i]
    fx1,fx2 = standardize(m1,m2,ev_tracks,flux_tracks,last_times,interpTime)
    interp = Table(); interp.add_column(fx1['Time'],name='Time')
    keys = fx1.keys()[1:]
    ev1 = ev_tracks[m1]; ev2 = ev_tracks[m2]

    # Interpolate
    frac = (mf-m1)/(m2-m1)
    for key in keys:
        interp.add_column(fx1[key]*(1.-frac)+fx2[key]*frac)
    tf = last_times[m1]*(1.-frac)+last_times[m2]*frac
    rf = (ev1['Stellar_Radius'][find_row(tf,ev1)]*(1.-frac)+ev2['Stellar_Radius'][find_row(tf,ev2)]*frac)*u.R_sun; rf = rf.to(u.kpc).value
    tempf = last_temps[m1]*(1.-frac)+last_temps[m2]*frac
    interp = interp[interp['Time'] < tf]
    
    interp.reverse()
    row_to_add = [0]
    row_to_add.extend(dust_sphere(mf,wav).value)
    interp.add_row(row_to_add)
    interp.reverse()

    row_to_add = [tf]
    bb = BlackBody(tempf*u.K); sed = (bb(wav)*u.sr).to(u.mJy).value
    sed = (sed*rf**2)
    row_to_add.extend(sed)
    interp.add_row(row_to_add)

    return interp
