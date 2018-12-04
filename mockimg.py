#!/usr/bin/env python3

from argparse import ArgumentParser, ArgumentTypeError
import os.path
from pkg_resources import resource_filename

from astropy import units, constants, wcs
from astropy.io import fits
import jsonlines
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.optimize import newton
from scipy.special import j1

# ---------------------------------------------------------------------------- #

parser = ArgumentParser()

def str_isfile(path):
    if not os.path.isfile(path):
        raise ArgumentTypeError("File not found.")
    return path

def float_isfraction(x):
    x = float(x)
    if (x < 0) or (x > 1):
         raise ArgumentTypeError("Fraction not between 0 and 1.")
    return x

parser.add_argument("--data", "-d",
                    type=str_isfile, default="stars.ndjson", metavar="string",
                    help="data file (default: stars.ndjson)")

parser.add_argument("--image_size", "-n",
                    type=int, default=100, metavar="int",
                    help="image size in pixels (default: 100)")
parser.add_argument("--pix_scale", "-s",
                    type=float, default=0.013, metavar="float",
                    help="pixel scale in arcsec per pixel (default: 0.013)")

parser.add_argument("--distance", "-R0", dest="R0",
                    type=float, default=8.34, metavar="float",
                    help="distance to the SMBH in kpc (default: 8.34)")
parser.add_argument("--mass", "-M0", dest="M0",
                    type=float, default=4.40, metavar="float",
                    help="mass of the SMBH in 10^6 M_sun (default: 4.40)")

parser.add_argument("--wavelength", "-w",
                    type=float, default=2e-6, metavar="float",
                    help="wavelength in meters (default: 2e-6)")
parser.add_argument("--aperture", "-D",
                    type=float, default=8, metavar="float",
                    help="aperture in meters (default: 8)")

parser.add_argument("--halo_frac",
                    type=float_isfraction, default=0, metavar="float",
                    help="strength of seeing halo as fraction (default: 0)")
parser.add_argument("--halo_fwhm",
                    type=float, default=0.5, metavar="float",
                    help="size of seeing halo in arcsec (default: 0.5)")

parser.add_argument("--zero_point", "-zp",
                    type=float, default=14, metavar="float",
                    help="magnitude zero point (default: 14)")

opts, _ = parser.parse_known_args()

# ---------------------------------------------------------------------------- #

m_unit = opts.M0*1e6*units.solMass
a_unit = units.arcsec
t_unit = units.yr

l_unit = a_unit.to(units.rad)*(opts.R0*units.kpc)
v_unit = l_unit/t_unit

G = float(constants.G.cgs * m_unit*t_unit**2/l_unit**3)

# ---------------------------------------------------------------------------- #

class Particle():
    def __init__(self, x=0, y=0, z=0, vx=0, vy=0, vz=0, mass=0, **kwargs):
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.mass = mass

class Orbit():
    def __init__(self, a, e, inc, Omega, omega, t0, mass=0, **kwargs):
        self.a = a
        self.e = e
        self.inc = inc
        self.Omega = Omega
        self.omega = omega
        self.t0 = t0
        self.mass = mass

class ProperMotion():
    def __init__(self, t0, x0, y0, vx, vy, ax=None, ay=None, **kwargs):
        self.t0 = t0
        self.x0 = x0
        self.y0 = y0
        self.vx = vx
        self.vy = vy
        self.ax = ax
        self.ay = ay

def is_flagged(d):
    return "flag" in d and d["flag"]

class Star():
    def __init__(self, primary, _id=None, orbit=[], proper_motion=[],
                 magnitude=[], mass=0, **kwargs):
        self.name = _id
        self.primary = primary
        self.mass = mass

        self.orbit = next((Orbit(mass=mass, **d) for d in orbit
                           if not is_flagged(d)), None)

        self.proper_motion = next((ProperMotion(**d) for d in proper_motion
                                  if not is_flagged(d)), None)   

        self.magnitude = next((d["value"] for d in magnitude
                               if d["band"] == "K" and not is_flagged(d)), None)

    def __repr__(self):
      return "<{self.__class__.__name__} {self.name}>"

    def locate(self, t):
        if self.orbit is not None:
            return eval_orbit(self.orbit, self.primary, t)
        elif self.proper_motion is not None:
            return eval_proper_motion(self.proper_motion, self.primary, t)
        else:
            raise Exception(f"Star {self.name} can not be located.")    

# ---------------------------------------------------------------------------- #

def mod2pi(x):
    return (x+np.pi)%(2*np.pi)-np.pi

def eccentric_anomaly(e, M, *args, **kwargs):
    if e < 1:
        f = lambda E: E - e*np.sin(E) - M
        fp = lambda E: 1 - e*np.cos(E)
        E0 = M if e < 0.8 else np.sign(M)*np.pi
        E = mod2pi(newton(f, E0, fp, *args, **kwargs))
    else:
        f = lambda E: E - e*np.sinh(E) - M
        fp = lambda E: 1 - e*np.cosh(E)
        E0 = np.sign(M) * np.log(2*np.fabs(M)/e+1.8)
        E = newton(f, E0, fp, *args, **kwargs)
    return E

def true_anomaly(e, M):
    E = eccentric_anomaly(e, M)
    if e > 1:
        return 2*np.arctan(np.sqrt((1+e)/(e-1))*np.tanh(E/2))
    else:
        return 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))        

def mean_motion(mu, a):
    return np.sign(a) * np.sqrt(np.fabs(mu/a**3))

# ---------------------------------------------------------------------------- #

def eval_orbit(orbit, primary, t):
    assert orbit.e != 1, "Can not initialize a radial orbit (e = 1)."
    assert orbit.e >= 0, "A valid orbit must have e >= 0."
    if orbit.e > 1:
        assert orbit.a < 0, "A bound orbit (a > 0) must have e < 1."
    else:
        assert orbit.a > 0, "An unbound orbit (a < 0) must have e > 1."

    mu = G*(orbit.mass+primary.mass)

    n = mean_motion(mu, orbit.a)
    M = mod2pi(n*(t-orbit.t0))
    f = true_anomaly(orbit.e, M)
    assert orbit.e*np.cos(f) > -1, "An unbound orbit can not have f set beyond \
        the range allowed by the parabolic asymptotes."

    r = orbit.a*(1-orbit.e*orbit.e)/(1+orbit.e*np.cos(f))
    v = np.sqrt(mu/orbit.a/(1-orbit.e**2))

    cO = np.cos(orbit.Omega)
    sO = np.sin(orbit.Omega)
    co = np.cos(orbit.omega)
    so = np.sin(orbit.omega)
    cf = np.cos(f)
    sf = np.sin(f)
    ci = np.cos(orbit.inc)
    si = np.sin(orbit.inc)

    x = primary.y+r*(sO*(co*cf-so*sf)+cO*(so*cf+co*sf)*ci)
    y = primary.x+r*(cO*(co*cf-so*sf)-sO*(so*cf+co*sf)*ci)
    z = primary.z+r*(so*cf+co*sf)*si

    vx = primary.vy+v*((orbit.e+cf)*(ci*co*cO-sO*so)-sf*(co*sO+ci*so*cO))
    vy = primary.vx+v*((orbit.e+cf)*(-ci*co*sO-cO*so)-sf*(co*cO-ci*so*sO))
    vz = primary.vz+v*((orbit.e+cf)*co*si-sf*si*so)

    return Particle(x, y, z, vx, vy, vz, orbit.mass)

def eval_proper_motion(promo, primary, t):
    vx = primary.vx+promo.vx
    vy = primary.vy+promo.vy
    
    x = primary.x+promo.x0+vx*(t-promo.t0)
    y = primary.y+promo.y0+vy*(t-promo.t0)
    
    if promo.ax:
        x += promo.ax/2*(t-promo.t0)**2
        vx += promo.ax*(t-promo.t0)
    if promo.ay:
        y += promo.ay/2*(t-promo.t0)**2
        vy += promo.ax*(t-promo.t0)
        
    return Particle(x, y, np.nan, vx, vy, np.nan)

# ---------------------------------------------------------------------------- #

def airy(r):
    x = np.pi*(opts.aperture/opts.wavelength)*np.deg2rad(opts.pix_scale*r/60**2)
    return (2*j1(x)/x)**2

def gauss(r):
    x = opts.pix_scale*r
    sigma = opts.halo_fwhm/np.sqrt(8*np.log(2))
    return np.exp(-x**2/(2*sigma**2))    

def psf(r):
    f = opts.halo_frac
    return (1-f)*airy(r) + f*gauss(r)

# ---------------------------------------------------------------------------- #    

black_hole = Particle(mass=1)

with jsonlines.open(resource_filename(__name__, opts.data)) as reader:
    stars = [Star(**star, primary=black_hole) for star in reader]

# ---------------------------------------------------------------------------- #

def gen_header(hdr=None, **kwargs):
    if hdr is None:
        hdr = fits.Header() # empty

    default_keys = ["R0", "M0", "pix_scale", "wavelength", "aperture",
                    "halo_frac", "halo_fwhm", "zero_point"]

    for k in default_keys:
        hdr[f"HIERARCH {k.upper()}"] = getattr(opts, k)

    for k, v in kwargs.items():
        hdr[f"HIERARCH {k.upper()}"] = v

    return hdr

def gen_image(t):    
    str_type = {"numpy": object, "fits": "4A"}
    float_type = {"numpy": float, "fits": "E"}
    dtype = {"name": str_type, "magnitude": float_type,
              "x_pix": float_type, "y_pix": float_type,
              "x": float_type, "y": float_type,
              "vx": float_type, "vy": float_type,
              "vz": float_type}

    table = np.empty(len(stars), dtype=[(k, dtype[k]["numpy"]) for k in dtype])

    w = wcs.WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [0, 0]
    w.wcs.crpix = [(opts.image_size+1)/2, (opts.image_size+1)/2]
    w.wcs.cdelt = [opts.pix_scale/60**2, opts.pix_scale/60**2]
    
    image = np.zeros((opts.image_size, opts.image_size))
    x_grid, y_grid = np.indices(image.shape)
    for i, star in enumerate(stars):
        p = star.locate(t)

        x_pix, y_pix = w.all_world2pix(p.y/60**2, -p.x/60**2, 0)
        f = 10**(-0.4*(star.magnitude-opts.zero_point))
        image += f * psf(np.sqrt((x_pix-x_grid)**2 + (y_pix-y_grid)**2))

        table["name"][i] = star.name
        table["magnitude"][i] = star.magnitude
        table["x_pix"][i] = y_pix
        table["y_pix"][i] = x_pix
        table["x"][i] = -p.x
        table["y"][i] = p.y
        table["vx"][i] = -p.vx
        table["vy"][i] = p.vy
        table["vz"][i] = (p.vz*v_unit).to(units.km/units.s).value

    w.wcs.crval = [266.4168262, -29.0077969] # Sgr A* (ICRS)

    image_hdr = gen_header(w.to_header(), epoch=t)
    image_hdu = fits.PrimaryHDU(image, header=image_hdr)
    
    table_hdr = gen_header(epoch=t)  
    table_hdu = fits.BinTableHDU.from_columns(fits.ColDefs([
        fits.Column(name=k, format=dtype[k]["fits"], array=table[k])
        for k in table.dtype.names]), header=table_hdr)

    return fits.HDUList([image_hdu, table_hdu])

def gen_plot(hdu_list, m0=20, m1=14, labels=True, show=True, save=None):
    image_hdu, table_hdu = hdu_list
    image_hdr = image_hdu.header
    image = image_hdu.data
    table = table_hdu.data

    w = wcs.WCS(image_hdr)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection=w)

    vmin = 10**(-0.4*(m0-opts.zero_point))
    vmax = 10**(-0.4*(m1-opts.zero_point))

    ax.imshow(image, origin="lower", interpolation="none",
              cmap="YlOrRd", norm=LogNorm(vmin=vmin, vmax=vmax))

    if labels:
        for label, x, y in zip(table["name"], table["x_pix"], table["y_pix"]):
            ax.text(x, y, label, ha="center", va="center",
                    clip_box=ax.bbox, clip_on=True)        
        ax.set_autoscale_on(False)
        ax.scatter(w.wcs.crval[0], w.wcs.crval[1], marker="x", color="k",
                   transform=ax.get_transform("world"))

    ra, dec = ax.coords
    ra.set_major_formatter("dd:mm:ss.s")
    dec.set_major_formatter("dd:mm:ss.s")
    ra.set_axislabel("RA")
    dec.set_axislabel("DEC")

    if save is not None:
        plt.savefig(save)
    if show:
        plt.show()

# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser.add_argument("--epoch", "-t",
                        type=float, required=True, metavar="float",
                        help="time of observation (year)")

    parser.add_argument("--output_fits", "-o",
                        type=str, required=False, metavar="string",
                        help="name of output file (fits)")  

    parser.add_argument("--preview", "-p",
                        action="store_true", default=False,
                        help="show an interactive preview image")

    parser.add_argument("--nolabels", "-nl", dest="labels",
                        action="store_false", default=True,
                        help="do not label stars on the preview image")

    parser.add_argument("--output_preview", "-op",
                        type=str, required=False, metavar="string",
                        help="name of output file (image format)")    

    opts = parser.parse_args()

    hdu_list = gen_image(opts.epoch)
    
    if opts.output_fits is not None:
        hdu_list.writeto(opts.output_fits, overwrite=True)

    if opts.preview or opts.output_preview is not None:
        gen_plot(hdu_list, labels=opts.labels,
                 show=opts.preview,
                 save=opts.output_preview)
