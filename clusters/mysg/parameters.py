import numpy as np

from collections import OrderedDict


def write_parfile(filename, parameters):
    f = open(filename, 'w')
    for name in parameters:
        if not isinstance(parameters[name], (str, np.string_, np.str_)):
            f.write("%s = %9.3e\n" % (name, parameters[name]))
        else:
            f.write("%s = %s\n" % (name, parameters[name].decode('ascii')))
    f.close()


def read_parfile(filename, nested=False):
    parameters = OrderedDict()
    f = open(filename, 'r')
    for line in f.readlines():
        key, value = line.strip().split('=')
        try:
            key, value = key.strip(), float(value)
        except ValueError:
            key, value = key.strip(), value.strip()
        if nested:
            p = parameters
            components = key.split('.')
            for component in components[:-1]:
                if component not in p:
                    p[component] = OrderedDict()
                p = p[component]
            p[components[-1]] = value
        else:
            parameters[key] = value
    f.close()
    return parameters
