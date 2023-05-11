import glob
import os

import numpy as np
from astropy.table import Table

# Set path with atmosphere templates
#DATA_PATH = os.path.join(os.path.dirname(__file__), 'data')

# Read in Kurucz atmospheres ready for interpolation
#MODELS_K = glob.glob(os.path.join('%s/atmos/kurucz/' % DATA_PATH, '*.hdf5'))
#TEFF_K = np.array([os.path.basename(model)[2:7] for model in MODELS_K],dtype=np.float32)
#TABLES_K = [Table.read(model, path="Table") for model in MODELS_K]

# Read in Phoenix atmospheres ready for interpolation
#MODELS_P = glob.glob(os.path.join('%s/atmos/phoenix/' % DATA_PATH, '*.hdf5'))
#TEFF_P = np.array([os.path.basename(model)[2:7] for model in MODELS_P],dtype=np.float32)
#TABLES_P = [Table.read(model, path="Table") for model in MODELS_P]

def interp_atmos(tval):
    
    datapath = os.path.dirname(__file__)
    
    # Read in Kurucz atmospheres ready for interpolation
    MODELS_K = glob.glob(f'{datapath}/data/atmos/kurucz/*.hdf5')
    TEFF_K = np.array([os.path.basename(model)[2:7] for model in MODELS_K],dtype=np.float32)
    TABLES_K = [Table.read(model, path="Table") for model in MODELS_K]

    # Read in Phoenix atmospheres ready for interpolation
    MODELS_P = glob.glob(f'{datapath}/data/atmos/phoenix/*.hdf5')
    TEFF_P = np.array([os.path.basename(model)[2:7] for model in MODELS_P],dtype=np.float32)
    TABLES_P = [Table.read(model, path="Table") for model in MODELS_P]

    # Decide which set of atmospheres to use
    if tval < 4000.:
        teff, tables = TEFF_P, TABLES_P
    else:
        teff, tables = TEFF_K, TABLES_K

    # We can't rely on the order being correct here
    order = np.argsort(teff)
    teff = teff[order]
    tables = [tables[i] for i in order]

    # Locate the temperature in the array
    i = np.searchsorted(teff, tval)
    
    # Retrieve two neighboring tables
    t1, t2 = tables[i - 1], tables[i]

    # Compute the interpolation fraction
    frac = (tval - teff[i - 1]) / (teff[i] - teff[i - 1])

    # Interpolate and return on the fly
    return t1['nu'], t1['fnu'] * (1. - frac) + t2['fnu'] * frac
