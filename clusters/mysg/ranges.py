from collections import OrderedDict


def _fixed_value(value):
    if type(value) is str:
        return {'sampling': 'str', 'value': value}
    else:
        return {'sampling': 'fixed', 'value': value}


def _log_range(lower, upper):
    return {'sampling': 'log10', 'lower': lower, 'upper': upper}


def _linear_range(lower, upper):
    return {'sampling': 'linear', 'lower': lower, 'upper': upper}


def _linked_value(parameter):
    return {'sampling': 'linked', 'parameter': parameter}


def write_ranges(filename, ranges):
    "Write a file containing information about ranges for each parameter"
    f = open(filename, 'w')
    for name in ranges:
        par = ranges[name]
        if par['sampling'] in ['log10', 'linear']:
            f.write("%-25s %-10s %10.3e  %10.3e\n"
                    % (name, par['sampling'], par['lower'], par['upper']))
        elif par['sampling'] in ['fixed']:
            f.write("%-25s %-10s %10.3e\n"
                    % (name, par['sampling'], par['value']))
        elif par['sampling'] in ['str']:
            f.write("%-25s %-10s  %s\n"
                    % (name, par['sampling'], par['value']))
        elif par['sampling'] in ['linked']:
            f.write("%-25s %-10s  %s\n"
                    % (name, par['sampling'], par['parameter']))
        else:
            raise Exception("Unknown sampling: %s" % par['sampling'])
    f.close()


def read_ranges(filename):
    "Read a file containing information about ranges for each parameter"
    ranges = OrderedDict()
    f = open(filename, 'r')
    for line in f.readlines():
        name, sampling, details = line.strip().split(None, 2)
        ranges[name] = OrderedDict()
        ranges[name]['sampling'] = sampling
        if sampling in ['log10', 'linear']:
            lower, upper = details.split()
            ranges[name]['lower'] = float(lower)
            ranges[name]['upper'] = float(upper)
        elif sampling in ['fixed']:
            ranges[name]['value'] = float(details)
        elif sampling in ['str']:
            ranges[name]['value'] = details
        elif sampling in ['linked']:
            ranges[name]['parameter'] = details
        else:
            raise Exception("Unknown sampling: %s" % sampling)
    f.close()
    return ranges


def select_required_ranges(set_name):

    ranges = OrderedDict()

    # Add parameters
    required = []

    # Source
    if set_name[0] == 's':  # Spherical source
        ranges['star.radius'] = _log_range(0.1, 100)  # R_sun
        ranges['star.temperature'] = _log_range(2000., 30000.)  # K

    # Disk
    if set_name[1] == 'p':  # Passive disk
        ranges['disk.mass'] = _log_range(1.e-8, 1.e-1)  # M_sun [dust]
        ranges['disk.rmax'] = _log_range(50., 5000.)  # AU
        ranges['disk.beta'] = _linear_range(1.0, 1.3)
        ranges['disk.p'] = _linear_range(-2.0, 0.0)
        ranges['disk.h100'] = _log_range(1., 20.)  # AU
    elif set_name[1] == 'a':  # Alpha-disk
        raise Exception("Not implemented: alpha-disk")

    # Envelope
    if set_name[2] == 'p':  # Power-law envelope
        ranges['envelope.rho_0'] = _log_range(1.e-24, 1.e-16)  # g/cm^3 [dust]
        ranges['envelope.power'] = _linear_range(-2.0, -1.0)
    elif set_name[2] == 'u':  # Ulrich envelope
        ranges['envelope.rho_0'] = _log_range(1.e-24, 1.e-16)  # g/cm^3 [dust]
        if 'disk.rmax' in ranges:
            ranges['envelope.rc'] = _linked_value('disk.rmax')
        else:
            ranges['envelope.rc'] = _log_range(50., 5000.)  # AU

    # Cavities
    if set_name[3] == 'b':  # Bipolar power-law cavities
        ranges['cavity.power'] = _linear_range(1.0, 2.0)
        ranges['cavity.theta_0'] = _linear_range(0., 60.)  # degrees
        ranges['cavity.rho_0'] = _log_range(1.e-23, 1.e-20)  # g/cm^3 [dust] Note: 1e-23 is ambient density

    # Inner holes
    if set_name[4] == 's':
        pass  # defaults to R_sub
    elif set_name[4] == 'h':
        if set_name[1] != '-':
            ranges['disk.rmin'] = _log_range(1., 1000.)  # R_sub
            if set_name[2] != '-':
                ranges['envelope.rmin'] = _linked_value('disk.rmin')
        elif set_name[2] != '-':
            ranges['envelope.rmin'] = _log_range(1., 1000.)  # R_sub

    # Ambient medium
    if set_name[5] == 'm':
        ranges['ambient.density'] = _fixed_value(1.e-23)  # [dust]
        ranges['ambient.temperature'] = _fixed_value(10.)

    # Dust
    if set_name[6] == 'i':
        dust_file = 'd03_5.5_3.0_A_sub.hdf5'
    else:
        raise Exception("No dust specified")

    if set_name[1] != '-':
        ranges['disk.dust'] = _fixed_value(dust_file)
    if set_name[2] != '-':
        ranges['envelope.dust'] = _fixed_value(dust_file)
    if set_name[3] != '-':
        ranges['cavity.dust'] = _fixed_value(dust_file)
    if set_name[5] != '-':
        ranges['ambient.dust'] = _fixed_value(dust_file)

    # PAH/VSGs
    if set_name[7] == 'p':
        raise Exception("Not implemented: PAH/VSGs")

    return ranges

# In future, can define function to find intersection of two ranges or
# compare the parameter space volume of two set of ranges.
