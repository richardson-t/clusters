from __future__ import unicode_literals, division, print_function

import numpy as np

from hyperion.model import AnalyticalYSOModel
from hyperion.util.constants import msun, rsun, au, pi, sigma
from hyperion.util.convenience import OptThinRadius
from hyperion.dust import SphericalDust

from astropy import log

from .parameters import read_parfile
from .atmosphere import interp_atmos

# Set dust sublimation temperature
TSUB = 1600.

# Number of viewing angles (for 2D models)
NVIEW = 9


def setup_model(parfile, output, imaging=True):

    # Read in model parameters
    par = read_parfile(parfile, nested=True)

    # Find all dust files
    dust_files = {}
    for par_name in par:
        if 'dust' in par[par_name]:
            dust_file = par[par_name]['dust']
            dust_files[dust_file] = SphericalDust(dust_file)

    # Find dimensionality of problem:
    if 'disk' in par:
        ndim = 2
        optimize = False
    elif 'cavity' in par:
        ndim = 2
        optimize = False
    elif 'envelope' in par and 'rc' in par['envelope']:
        ndim = 2
        optimize = False
    else:
        ndim = 1
        optimize = True

    # Set up model
    m = AnalyticalYSOModel(output)

    if not 'star' in par:
        raise Exception("Cannot compute a model without a central source")

    # Set radius and luminosity
    m.star.radius = par['star']['radius'] * rsun
    m.star.luminosity = 4. * pi * (par['star']['radius'] * rsun) ** 2. \
                        * sigma * par['star']['temperature'] ** 4.

    # Interpolate and set spectrum
    nu, fnu = interp_atmos(par['star']['temperature'])
    m.star.spectrum = (nu, fnu)

    subtract_from_ambient = []

    if 'disk' in par:

        # Add the flared disk component
        disk = m.add_flared_disk()

        # Basic parameters
        disk.mass = par['disk']['mass'] * msun
        disk.rmax = par['disk']['rmax'] * au
        disk.p = par['disk']['p']
        disk.beta = par['disk']['beta']
        disk.h_0 = par['disk']['h100'] * au
        disk.r_0 = 100. * au

        # Set inner and outer walls to be spherical
        disk.cylindrical_inner_rim = False
        disk.cylindrical_outer_rim = False

        # Set dust
        disk.dust = dust_files[par['disk']['dust']]

        # Inner radius
        if 'rmin' in par['disk']:
            disk.rmin = par['disk']['rmin'] * OptThinRadius(TSUB)
        else:
            disk.rmin = OptThinRadius(TSUB)

        # Settling
        if 'eta' in par['disk']:
            raise Exception("Dust settling not implemented")

        # Accretion luminosity
        if 'lacc' in par['disk']:
            raise Exception("Accretion luminosity not implemented")
            # m.setup_magnetospheric_accretion(par['disk']['lacc'] * lsun,
            #                                  par['disk']['rtrunc'],
            #                                  par['star']['fspot'], disk)

        subtract_from_ambient.append(disk)

    if 'envelope' in par:

        if 'rc' in par['envelope']:  # Ulrich envelope

            envelope = m.add_ulrich_envelope()
            envelope.rho_0 = par['envelope']['rho_0']
            envelope.rc = par['envelope']['rc'] * au

        elif 'power' in par['envelope']:  # Power-law envelope

            envelope = m.add_power_law_envelope()
            envelope.power = par['envelope']['power']
            envelope.rho_0 = par['envelope']['rho_0']
            envelope.r_0 = 1000. * au

        # Set dust
        envelope.dust = dust_files[par['envelope']['dust']]

        # Inner radius
        if 'rmin' in par['envelope']:
            envelope.rmin = par['envelope']['rmin'] * OptThinRadius(TSUB)
        else:
            envelope.rmin = OptThinRadius(TSUB)

        subtract_from_ambient.append(envelope)

    if 'cavity' in par:

        if not 'envelope' in par:
            raise Exception("Can't have a bipolar cavity without an envelope")

        # Add the bipolar cavity component
        cavity = envelope.add_bipolar_cavity()

        # Basic parameters
        cavity.power = par['cavity']['power']
        cavity.r_0 = 10000 * au
        cavity.theta_0 = par['cavity']['theta_0']
        cavity.rho_0 = par['cavity']['rho_0']
        cavity.rho_exp = 0.

        # Very important is that the cavity density should not be *larger* than
        # the envelope density.
        cavity.cap_to_envelope_density = True

        # Set dust
        cavity.dust = dust_files[par['cavity']['dust']]

        subtract_from_ambient.append(cavity)

    if 'ambient' in par:

        # Add the ambient medium contribution
        ambient = m.add_ambient_medium(subtract=subtract_from_ambient)

        # Set the density, temperature, and dust properties
        ambient.rho = par['ambient']['density']
        ambient.temperature = par['ambient']['temperature']
        ambient.dust = dust_files[par['ambient']['dust']]

        # If there is an envelope, set the outer radius to where the
        # optically thin temperature would transition to the ambient medium
        # temperature
        if 'envelope' in par:

            # Find radius where the optically thin temperature drops to the
            # ambient temperature. We can do this only if we've already set
            # up all the sources of emission beforehand (which we have)
            rmax_temp = OptThinRadius(ambient.temperature).evaluate(m.star, envelope.dust)

            # Find radius where the envelope density drops to the ambient density
            rmax_dens = envelope.outermost_radius(ambient.rho)

            # If disk radius is larger than this, use that instead
            if 'disk' in par:
                if disk.rmax > rmax_dens:
                    rmax_dens = disk.rmax

            # Pick the largest
            if rmax_temp < rmax_dens:
                print("Setting envelope outer radius to that where rho(r) = rho_amb")
                envelope.rmax = rmax_dens
            else:
                print("Setting envelope outer radius to that where T_thin(r) = T_amb")
                envelope.rmax = OptThinRadius(ambient.temperature)

            ambient.rmax = envelope.rmax

        else:

            if 'disk' in par:

                # Find radius where the optically thin temperature drops to the
                # ambient temperature. We can do this only if we've already set
                # up all the sources of emission beforehand (which we have)
                rmax_temp = OptThinRadius(ambient.temperature).evaluate(m.star, ambient.dust)

                # Find outer disk radius
                rmax_dens = disk.rmax

                # Pick the largest
                if rmax_temp < rmax_dens:
                    print("Setting ambient outer radius to outer disk radius")
                    ambient.rmax = rmax_dens
                else:
                    print("Setting ambient outer radius to that where T_thin(r) = T_amb")
                    ambient.rmax = OptThinRadius(ambient.temperature)

            else:

                ambient.rmax = OptThinRadius(ambient.temperature)

        # The inner radius for the ambient medium should be the largest of
        # the inner radii for the disk and envelope
        if 'envelope' in par and 'rmin' in par['envelope']:
            if 'disk' in par and 'rmin' in par['disk']:
                ambient.rmin = max(par['disk']['rmin'], \
                                   par['envelope']['rmin']) \
                               * OptThinRadius(TSUB)
            else:
                ambient.rmin = par['envelope']['rmin'] * OptThinRadius(TSUB)
        elif 'disk' in par and 'rmin' in par['disk']:
            ambient.rmin = par['disk']['rmin'] * OptThinRadius(TSUB)
        else:
            ambient.rmin = OptThinRadius(TSUB)

        # The ambient medium needs to go out to sqrt(2.) times the envelope
        # radius to make sure the slab is full (don't need to do sqrt(3)
        # because we only need a cylinder along line of sight)
        ambient.rmax *= np.sqrt(2.)

        # Make sure that the temperature in the model is always at least
        # the ambient temperature
        m.set_minimum_temperature(ambient.temperature)

    else:

        # Make sure that the temperature in the model is always at least
        # the CMB temperature
        m.set_minimum_temperature(2.725)

        if 'envelope' in par:
            raise Exception("Can't have an envelope without an ambient medium")

    # Use raytracing to improve s/n of thermal/source emission
    m.set_raytracing(True)

    # Use the modified random walk
    m.set_mrw(True, gamma=2.)

    # Use the partial diffusion approximation
    m.set_pda(True)

    # Improve s/n of scattering by forcing the first interaction
    m.set_forced_first_scattering(True)

    # Set up grid.
    if ndim == 1:
        m.set_spherical_polar_grid_auto(400, 1, 1)
    else:
        m.set_spherical_polar_grid_auto(400, 300, 1)

    # Find the range of radii spanned by the grid
    rmin, rmax = m.radial_range()

    # Set up SEDs
    image = m.add_peeled_images(sed=True, image=False)
    image.set_wavelength_range(200, 0.01, 5000.)

    if 'ambient' in par:
        image.set_aperture_range(20, rmin, rmax / np.sqrt(2.))
    else:
        image.set_aperture_range(20, rmin, rmax)

    image.set_output_bytes(8)
    image.set_track_origin(True)
    image.set_uncertainties(True)
    image.set_stokes(True)

    if ndim == 1:

        # Viewing angle does not matter
        image.set_viewing_angles([45.], [45.])

    else:

        # Use stratified random sampling to ensure that all models
        # contain a viewing angle in each bin, but also ensure we have a
        # continuum of viewing angles over all models
        xi = np.random.uniform(0., 90./float(NVIEW), NVIEW)
        theta = xi + np.linspace(0., 90. * (1. - 1./float(NVIEW)), NVIEW)
        theta_even = np.linspace(0.,80.,9) + 5
        image.set_viewing_angles(theta_even, np.repeat(45., NVIEW))

    if 'ambient' in par:  # take a slab to avoid spherical geometrical effects
        w = ambient.rmax / np.sqrt(2.)
        image.set_depth(-w, w)
    else:  # don't need to take a slab, as no ambient material or envelope
        image.set_depth(-np.inf, np.inf)

    # Set number of photons

    if imaging:
        n_imaging=1e6
        n_raytracing_sources=10000
        n_raytracing_dust=1e6
    else:
        n_imaging=0
        n_raytracing_sources=0
        n_raytracing_dust=0

    if ndim == 1:
        m.set_n_photons(initial=100, imaging=n_imaging,
                        raytracing_sources=n_raytracing_sources,
                        raytracing_dust=n_raytracing_dust)
    else:
        m.set_n_photons(initial=1000000, imaging=n_imaging,
                        raytracing_sources=n_raytracing_sources,
                        raytracing_dust=n_raytracing_dust)

    # Set physical array output to 32-bit
    m.set_output_bytes(4)

    # Set maximum of 10^8 interactions per photon
    m.set_max_interactions(1e8)

    # Only request certain arrays to be output
    m.conf.output.output_density = 'none'
    m.conf.output.output_specific_energy = 'last'
    m.conf.output.output_n_photons = 'none'
    m.conf.output.output_density_diff = 'last'

    # Set number of temperature iterations and convergence criterion
    m.set_n_initial_iterations(10)
    m.set_convergence(True, percentile=99.0, absolute=2.0, relative=1.1)

    # Don't copy the full input into the output files
    m.set_copy_input(False)

    # Check whether the model is very optically thick

    mf = m.to_model()

    if 'envelope' in par:

        from hyperion.model.helpers import tau_to_radius
        surface = tau_to_radius(mf, tau=1., wav=5e3)
        rtau = np.min(surface)

        if rtau > rmin and optimize:
            log.warn("tau_5mm > 1 for all (theta,phi) values - truncating "
                     "inner envelope from {0:.3f}au to {1:.3f}au".format(mf.grid.r_wall[1] / au, rtau / au))
            for item in mf.grid['density']:
                item.array[mf.grid.gr < rtau] = 0.

    # Write out file
    mf.write(copy=False, absolute_paths=False,
             physics_dtype=np.float32, wall_dtype=float)
