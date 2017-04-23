# Galactic Center Mock Images

Generate mock images of the S-star cluster at any particular time, based on the long term-monitoring of individual stellar motions.

Example ([data file](https://github.com/pmplewa/GCdb/blob/master/data/stars.ndjson)):
```
python mockimg.py -h
```
```
python mockimg.py -t 2018.3 -o image.fits -d stars.ndjson
```
```
>>> from mockimg import gen_hdu
>>> hdu = gen_hdu(2018.3)
```
