"""
Explore the effect on the "true" PMF / CMF of different accretion histories.
Examples:
    (1) All stars start accreting at the same time
    (2) All stars finish accreting at the same time
    (3) Start time is Gaussian distributed
    (4) End time is Gaussian distributed
"""
from collections import defaultdict
import imf
import random
import numpy as np
from astropy import units as u
from astropy import constants
from astropy import table

protoev_history = {}

#align tracks by beginning
def find_row(selected_time, tbl):
    times = (tbl['Time']*u.s).to(u.yr)

    selected_row = np.argmin(np.abs(times - selected_time))

    return tbl[selected_row]

# "last_time" is the last timestep in which the star is still accreting
last_time = {mass: tbl['Time'][tbl['Stellar_Mass'] == tbl['Stellar_Mass'][-1]][0]
             for mass,tbl in protoev_history.items()}

#align tracks by end
def find_row_rtime(rewind_time, tbl):
    times = (tbl['Time']*u.s).to(u.yr)
    final_mass = tbl['Stellar_Mass'][-1]
    last_time = times[tbl['Stellar_Mass'] == final_mass][0]

    selected_time = last_time - rewind_time
    selected_row = np.argmin(np.abs(times - selected_time))

    return tbl[selected_row]


def make_plots(row_finder, time_factor=1,
               time_offsets=defaultdict(lambda: 0),
               title="All stars start accretion at the same time",
               sinceoruntil='since',
               startorend='start', figname="alpha_vs_time_same_start_times.pdf"):

    if not isinstance(time_offsets, defaultdict):
        time_offsets = {mass: np.random.randn() * time_offsets
                        for mass in protoev_history}
    alphafits = []
    for ii,time in enumerate(np.linspace(0,2,21)*u.Myr):

        masses = {stellar_mass:row_finder(time + time_offsets[stellar_mass], tbl)['Stellar_Mass']
                        for stellar_mass,tbl in protoev_history.items()}
        luminosities = {stellar_mass:row_finder(time + time_offsets[stellar_mass], tbl)['Total_Luminosity']
                        for stellar_mass,tbl in protoev_history.items()}
        temperatures = {stellar_mass:row_finder(time + time_offsets[stellar_mass], tbl)['Temperature']
                        for stellar_mass,tbl in protoev_history.items()}
        radii = {stellar_mass:row_finder(time + time_offsets[stellar_mass], tbl)['Stellar_Radius']
                        for stellar_mass,tbl in protoev_history.items()}

        lum_vals = u.Quantity([luminosities[k] for k in cluster], u.erg/u.s)
        tem_vals = u.Quantity([temperatures[k] for k in cluster], u.K)
        rad_vals = u.Quantity([radii[k] for k in cluster], u.cm)
        mass_vals = u.Quantity([masses[k] for k in cluster], u.M_sun)

        massfit = powerlaw.Fit(mass_vals)
        lumfit = powerlaw.Fit(lum_vals)
        print(f"Time {time}")
        print(f"Mass alpha={massfit.alpha} for M>{massfit.xmin}")
        print(f"Luminosity alpha={lumfit.alpha} for L>{lumfit.xmin}")

        alphafits.append([time_factor * time.value, massfit.alpha,
                          massfit.xmin, lumfit.alpha, lumfit.xmin])


### Example 1: All start at the same time ###
make_plots(row_finder=find_row)

### Example 2: All end at the same time ###
make_plots(row_finder=find_row_rtime, startorend='end', sinceoruntil='until',
           title="All stars end accretion at the same time",
           figname="alpha_vs_time_same_end_times.pdf",
           time_factor=-1,
          )

### Example 3: Start time is Gaussian distributed ###
make_plots(row_finder=find_row, startorend='start',
           time_offsets=0.3*u.Myr,
           title="Accretion start time is Gaussian distributed with width=0.3 Myr",
           figname="alpha_vs_time_gaussian0.3Myr_starts.pdf",
           time_factor=1,
          )

### Example 4: End time is Gaussian distributed ###
make_plots(row_finder=find_row_rtime, startorend='end',
           sinceoruntil='until',
           time_offsets=0.3*u.Myr,
           title="Accretion end time is Gaussian distributed with width=0.3 Myr",
           figname="alpha_vs_time_gaussian0.3Myr_ends.pdf",
           time_factor=-1,
          )
