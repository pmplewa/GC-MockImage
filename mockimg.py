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
        raise ArgumentTypeError("Data file not found.")
    return path

def float_isfraction(x):
    x = float(x)
    if (x < 0) or (x > 1):
         raise ArgumentTypeError("Choose a fraction between 0 and 1.")
    return x

parser.add_argument("--data", "-d",
                    type=str_isfile, default="stars.ndjson", metavar="string",
                    help="data file (default: stars.ndjson)")

parser.add_argument("--image_size", "-n", dest="n",
                    type=int, default=256, metavar="int",
                    help="image size in pixels (default: 256)")
parser.add_argument("--pixel_scale", "-s", dest="s",
                    type=float, default=0.013, metavar="float",
                    help="pixel scale in arcsec per pixel (default: 0.013)")

parser.add_argument("-R0", "--distance", dest="R0",
                    type=float, default=8.34, metavar="float",
                    help="distance to the SMBH in kpc (default: 8.34)")
parser.add_argument("-M0", "--mass", dest="M0",
                    type=float, default=4.40, metavar="float",
                    help="mass of the SMBH in 10^6 M_sun (default: 4.40)")

parser.add_argument("--wavelength", "-w", dest="w",
                    type=float, default=2e-6, metavar="float",
                    help="wavelength in meters (default: 2e-6)")
parser.add_argument("--aperture", "-D", dest="D",
                    type=float, default=8, metavar="float",
                    help="aperture in meters (default: 8)")

parser.add_argument("--seeing", "-f", dest="f",
                    type=float_isfraction, default=0, metavar="float",
                    help="strength of seeing halo as fraction (default: 0)")
parser.add_argument("--fwhm", "-fwhm", dest="fwhm",
                    type=float, default=0.5, metavar="float",
                    help="size of seeing halo in arcsec (default: 0.5)")

parser.add_argument("--zero_point", "-zp", dest="zp",
                    type=float, default=14, metavar="float",
                    help="magnitude zero point (default: 14)")

opts, _ = parser.parse_known_args()

# ---------------------------------------------------------------------------- #

m_unit = opts.M0*1e6*units.solMass
a_unit = units.arcsec
t_unit = units.yr

l_unit = a_unit.to(units.rad)*(opts.R0*units.kpc)

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
        
class Star():
    def __init__(self, primary, _id=None, orbit=[], proper_motion=[],
                 magnitude=[], mass=0, **kwargs):
        self.name = _id
        self.primary = primary
        self.mass = mass

        self.orbit = next((Orbit(mass=mass, **d) for d in orbit
                           if not("flag" in d and d["flag"])), None)
        self.promo = next((ProperMotion(**d) for d in proper_motion
                           if not ("flag" in d and d["flag"])), None)        
        self.magnitude = next((d["value"] for d in magnitude if d["band"] == "K"
                               and not ("flag" in d and d["flag"])), None)

    def __repr__(self):
      return f"<{self.__class__.__name__} {self.name}>"

    def locate(self, t):
        if self.orbit is not None:
            return orbit_to_particle(self.orbit, self.primary, t)
        elif self.promo is not None:
            return promo_to_particle(self.promo, self.primary, t)
        else:
            raise Exception(f"Star {self.name} can not be located.")    

# ---------------------------------------------------------------------------- #

def mod2pi(x):
    return (x+np.pi)%(2*np.pi)-np.pi

def eccentric_anomaly(e, M, *args, **kwargs):
    if e < 1:
        f = lambda E: E-e*np.sin(E)-M
        fp = lambda E: 1-e*np.cos(E)
        E0 = M if e < 0.8 else np.sign(M)*np.pi
        E = mod2pi(newton(f, E0, fp, *args, **kwargs))
    else:
        f = lambda E: E-e*np.sinh(E)-M
        fp = lambda E: 1-e*np.cosh(E)
        E0 = np.sign(M)*np.log(2*np.fabs(M)/e+1.8)
        E = newton(f, E0, fp, *args, **kwargs)
    return E

def true_anomaly(e, M):
    E = eccentric_anomaly(e, M)
    if e > 1:
        return 2*np.arctan(np.sqrt((1+e)/(e-1))*np.tanh(E/2))
    else:
        return 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))        

# ---------------------------------------------------------------------------- #            

def orbit_to_particle(orbit, primary, t):
    assert orbit.e != 1, "Can nott initialize a radial orbit."
    assert orbit.e >= 0, "Eccentricity must be greater or equal to zero."
    if orbit.e > 1:
        assert orbit.a < 0, "Bound orbit (a > 0) must have e < 1."
    else:
        assert orbit.a > 0, "Unbound orbit (a < 0) must have e > 1."

    mu = G*(orbit.mass+primary.mass)

    n = orbit.a/np.fabs(orbit.a)*np.sqrt(np.fabs(mu/orbit.a**3))
    M = mod2pi(n*(t-orbit.t0))
    f = true_anomaly(orbit.e, M)
    assert orbit.e*np.cos(f) > -1, "Unbound orbit can not have f set beyond \
        the range allowed by the asymptotes set by the parabola."

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

def promo_to_particle(promo, primary, t):
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
        
    return Particle(x, y, None, vx, vy, None)

# ---------------------------------------------------------------------------- #

def airy(r, aperture=opts.D, wavelength=opts.w, pixel_scale=opts.s):
    x = np.pi*(aperture/wavelength)*np.deg2rad(pixel_scale/60**2)*r
    return (2*j1(x)/x)**2

def gauss(r, fwhm=opts.fwhm, pixel_scale=opts.s):
    x = pixel_scale*r
    sigma = fwhm/np.sqrt(8*np.log(2))
    return np.exp(-x**2/(2*sigma**2))    

def psf(r, f=opts.f):
    return (1-f)*airy(r)+f*gauss(r)

# ---------------------------------------------------------------------------- #    

black_hole = Particle(mass=1)
with jsonlines.open(resource_filename(__name__, "stars.ndjson")) as reader:
    stars = [Star(**star, primary=black_hole) for star in reader]

# ---------------------------------------------------------------------------- #

def gen_hdu(t, stars=stars, image_size=opts.n, pixel_scale=opts.s,
            zero_point=opts.zp):
    image = np.zeros((image_size, image_size))
    xi, yi = np.indices(image.shape)

    w = wcs.WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [0, 0]
    w.wcs.crpix = [(image_size+1)/2, (image_size+1)/2]
    w.wcs.cdelt = [pixel_scale/60**2, pixel_scale/60**2]
    
    for star in stars:
        p = star.locate(t)
        x, y = w.all_world2pix(p.x/60**2, p.y/60**2, 0)
        f = 10**(-0.4*(star.magnitude-zero_point))
        image += f*psf(np.sqrt((x-xi)**2+(y-yi)**2))

    w.wcs.crval = [266.41681662, -29.00782497]
    hdr = w.to_header()
    hdr["EPOCH"] = t
    hdu = fits.PrimaryHDU(np.fliplr(np.transpose(image)), hdr)
    return hdu

def gen_plot(hdu, stars=stars, m0=20, m1=14, zero_point=opts.zp, labels=False,
             output=None):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection=wcs.WCS(hdu.header))
    vmin = 10**(-0.4*(m0-zero_point))
    vmax = 10**(-0.4*(m1-zero_point))
    ax.imshow(hdu.data, origin="lower", cmap="YlOrRd", interpolation="none",
              norm=LogNorm(vmin=vmin, vmax=vmax))
    if labels:
        hdr = hdu.header
        w = wcs.WCS(hdr)
        w.wcs.crval = [0, 0]
        for star in stars:
            p = star.locate(hdr["EPOCH"])
            x, y = w.all_world2pix(-p.x/60**2, p.y/60**2, 0)
            ax.text(x, y, f"{star.name}", ha="center", va="center",
                    clip_box=ax.bbox, clip_on=True)
    ra, dec = ax.coords
    ra.set_major_formatter("dd:mm:ss.s")
    dec.set_major_formatter("dd:mm:ss.s")
    ra.set_axislabel("RA")
    dec.set_axislabel("DEC")
    if output is not None:
        plt.savefig(output)
    plt.show()     

# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser.add_argument("--epoch", "-t", dest="t",
                        type=float, required=True, metavar="float",
                        help="time of observation (year)")

    parser.add_argument("--output_fits", "-o",
                        type=str, required=False, metavar="string",
                        help="name of output file (fits)")

    parser.add_argument("--preview", "-p", action="store_true", default=False,
                        help="make a preview image")

    parser.add_argument("--labels", "-l", action="store_true", default=False,
                        help="label stars on the preview image")

    parser.add_argument("--output_preview", "-op",
                        type=str, required=False, metavar="string",
                        help="name of output file (image format)")    

    opts = parser.parse_args()

    hdu = gen_hdu(opts.t)
    fits.HDUList([hdu]).writeto(opts.output_fits, overwrite=True)

    if opts.preview or opts.output_preview is not None:
        gen_plot(hdu, labels=opts.labels, output=opts.output_preview)
