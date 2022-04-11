import torch
from torch import nn
from torch.autograd import Variable
import torchvision
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.functional import avg_pool2d, interpolate
import numpy as np
import math
from torch.utils.data import Dataset, DataLoader
import glob
import math
from functools import partial
from torchvision import transforms, utils
import random
import os
from datetime import datetime
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
    
import numpy as np
import xarray as xr
import requests
from os import mkdir
from os.path import isdir
import os.path
from os import path
import matplotlib.pyplot as plt
from skimage import exposure,io
import math
from datetime import datetime
import glob
import warnings
warnings.filterwarnings('ignore')

def _cloudmask(QA_PIXEL, cirrus_flag=True, dilate_flag=True,
                             shadow_flag=True, snow_flag=True,
                             ):
    """Extract cloud mask from the Landsat Collection 2 SR QA_PIXEL band

    Parameters
    ----------
    img : ee.Image
        Image from a Landsat Collection 2 SR image collection with a QA_PIXEL
        band (e.g. LANDSAT/LC08/C02/T1_L2).
    cirrus_flag : bool
        If true, mask cirrus pixels (the default is False).
        Note, cirrus bits are only set for Landsat 8 (OLI) images.
    dilate_flag : bool
        If true, mask dilated cloud pixels (the default is False).
    shadow_flag : bool
        If true, mask shadow pixels (the default is True).
    snow_flag : bool
        If true, mask snow pixels (the default is False).

    Returns
    -------
    ee.Image

    Notes
    -----
    Output image is structured to be applied directly with updateMask()
        i.e. 0 is cloud/masked, 1 is clear/unmasked

    Assuming Cloud must be set to check Cloud Confidence

    Bits
        0: Fill
            0 for image data
            1 for fill data
        1: Dilated Cloud
            0 for cloud is not dilated or no cloud
            1 for cloud dilation
        2: Cirrus
            0 for no confidence level set or low confidence
            1 for high confidence cirrus
        3: Cloud
            0 for cloud confidence is not high
            1 for high confidence cloud
        4: Cloud Shadow
            0 for Cloud Shadow Confidence is not high
            1 for high confidence cloud shadow
        5: Snow
            0 for Snow/Ice Confidence is not high
            1 for high confidence snow cover
        6: Clear
            0 if Cloud or Dilated Cloud bits are set
            1 if Cloud and Dilated Cloud bits are not set
        7: Water
            0 for land or cloud
            1 for water
        8-9: Cloud Confidence
        10-11: Cloud Shadow Confidence
        12-13: Snow/Ice Confidence
        14-15: Cirrus Confidence

    Confidence values
        00: "No confidence level set"
        01: "Low confidence"
        10: "Medium confidence" (for Cloud Confidence only, otherwise "Reserved")
        11: "High confidence"

    References
    ----------
    https://prd-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/atoms/files/LSDS-1328_Landsat8-9-OLI-TIRS-C2-L2-DFCB-v6.pdf

    """
    def Or(input1, input2):
        return (input1 != 0) | (input2 != 0).astype(int)
    
    def Not(input1):
        return (input1 == 0).astype(int)
    
    def RightShift(input1, input2):
        return (input1.astype(int) >> input2)

    
    qa_img = QA_PIXEL
    cloud_mask = np.not_equal(np.bitwise_and(RightShift(qa_img, 3), 1), 0).astype(int)
#     cloud_mask = qa_img.rightShift(3).bitwiseAnd(1).neq(0)
    #     .And(qa_img.rightShift(6).bitwiseAnd(3).gte(cloud_confidence))
    if cirrus_flag:
        cloud_mask = Or(cloud_mask, np.not_equal(np.bitwise_and(RightShift(qa_img, 2), 1), 0).astype(int))
#         cloud_mask = cloud_mask.Or(qa_img.rightShift(2).bitwiseAnd(1).neq(0))
    if dilate_flag:
        cloud_mask = Or(cloud_mask, np.not_equal(np.bitwise_and(RightShift(qa_img, 1), 1), 0).astype(int))
#         cloud_mask = cloud_mask.Or(qa_img.rightShift(1).bitwiseAnd(1).neq(0))
    if shadow_flag:
        cloud_mask = Or(cloud_mask, np.not_equal(np.bitwise_and(RightShift(qa_img, 4), 1), 0).astype(int))
#         cloud_mask = cloud_mask.Or(qa_img.rightShift(4).bitwiseAnd(1).neq(0))
    if snow_flag:
        cloud_mask = Or(cloud_mask, np.not_equal(np.bitwise_and(RightShift(qa_img, 5), 1), 0).astype(int))
#         cloud_mask = cloud_mask.Or(qa_img.rightShift(5).bitwiseAnd(1).neq(0))

    return Not(cloud_mask)

def _ndvi(nir, red):
    return ((((nir-red)/(nir+red)) + 1.0)/2.0)

def _lst(band10):  
    return (band10 * 0.00341802) + 149.0
    k1 = 774.89
    k2 = 1321.08
    l1 = ((band10*0.0003342) + 0.1)
    return k2/(np.log((k1/l1) + 1))

def _interpolate(data, data_name, reference):
    if data_name == "":
        interpolated_data = reference.assign(
            interpolate=lambda each_data: data.interp(lat=reference.lat,
                                                      lon=reference.lon))
    else:
        interpolated_data = reference.assign(
            interpolate=lambda each_data: data[data_name].interp(lat=reference.lat,
                                                                 lon=reference.lon))
#       data=np.flipud(interpolated_data.interpolate.values),  
    interpolated_data = xr.DataArray(
        data=interpolated_data.interpolate.values,
        dims=["y", "x"],
        coords=dict(
            lon=(["y", "x"], interpolated_data.lon),
            lat=(["y", "x"], interpolated_data.lat),
        ),
    )
    return interpolated_data

def _gridmet2landsat(band10, lat, lon):
        
    return  xr.Dataset(
        data_vars=dict(
            lst=(["y", "x"], band10)
        ),
        coords=dict(
            lon=(["y", "x"], lon),
            lat=(["y", "x"], lat),
        ),
    )
    
def _tmax(date, band10, lat, lon):
    os.makedirs("tmax", exist_ok=True)
    date_split = date.split("-")
    year = date_split[0]
    name = "tmax/" + str(date) + "_" + str(float(lat.min())) + "_" + str(float(lat.max())) + "_" + str(float(lon.min())) + "_" + str(float(lon.max())) + ".npy"
    if path.exists(name):
        return np.load(name)
    if not path.exists("tmax_" + year + ".nc"):
        url = 'https://www.northwestknowledge.net/metdata/data/tmmx_' + year + '.nc'
        r = requests.get(url)
        with open('tmax_' + year + '.nc', 'wb') as f:
            f.write(r.content)
    tmax_data = xr.open_dataset('tmax_' + year + '.nc').sel(day=date)
    tmax_data["tmax"] = tmax_data["air_temperature"]
    tmax_data = tmax_data.drop(["air_temperature"])
    tmax_data = tmax_data.drop_dims("crs")
    tmax = _interpolate(tmax_data, "tmax", _gridmet2landsat(band10, lat, lon))
    np.save(name, tmax.data)
    return tmax.data

def _tmin(date, band10, lat, lon): 
    os.makedirs("tmin", exist_ok=True)
    date_split = date.split("-")
    year = date_split[0]
    name = "tmin/" + str(date) + "_" + str(float(lat.min())) + "_" + str(float(lat.max())) + "_" + str(float(lon.min())) + "_" + str(float(lon.max())) + ".npy"
    if path.exists(name):
        return np.load(name)
    if not path.exists("tmin_" + year + ".nc"):
        url = 'https://www.northwestknowledge.net/metdata/data/tmmn_' + year + '.nc'
        r = requests.get(url)
        with open('tmmn_' + year + '.nc', 'wb') as f:
            f.write(r.content)
    tmin_data = xr.open_dataset('tmmn_' + year + '.nc').sel(day=date)
    tmin_data["tmin"] = tmin_data["air_temperature"]
    tmin_data = tmin_data.drop(["air_temperature"])
    tmin_data = tmin_data.drop_dims("crs")
    tmin = _interpolate(tmin_data, "tmin", _gridmet2landsat(band10, lat, lon))
    np.save(name, tmin.data)
    return tmin.data

def _srad(date, band10, lat, lon): 
    os.makedirs("srad", exist_ok=True)
    date_split = date.split("-")
    year = date_split[0]
    name = "srad/" + str(date) + "_" + str(float(lat.min())) + "_" + str(float(lat.max())) + "_" + str(float(lon.min())) + "_" + str(float(lon.max())) + ".npy"
    if path.exists(name):
        return np.load(name)
    if not path.exists("srad_" + year + ".nc"):
        url = 'https://www.northwestknowledge.net/metdata/data/srad_' + year + '.nc'
        r = requests.get(url)
        with open('srad_' + year + '.nc', 'wb') as f:
            f.write(r.content)
    srad_data = xr.open_dataset('srad_' + year + '.nc').sel(day=date)
    srad_data["srad"] = srad_data["surface_downwelling_shortwave_flux_in_air"]
    srad_data = srad_data.drop(["surface_downwelling_shortwave_flux_in_air"])
    srad_data = srad_data.drop_dims("crs")
    srad = _interpolate(srad_data, "srad", _gridmet2landsat(band10, lat, lon))
    np.save(name, srad.data)
    return srad.data

def _sph(date, band10, lat, lon): 
    os.makedirs("sph", exist_ok=True)
    date_split = date.split("-")
    year = date_split[0]
    name = "sph/" + str(date) + "_" + str(float(lat.min())) + "_" + str(float(lat.max())) + "_" + str(float(lon.min())) + "_" + str(float(lon.max())) + ".npy"
    if path.exists(name):
        return np.load(name)
    if not path.exists("sph_" + year + ".nc"):
        url = 'https://www.northwestknowledge.net/metdata/data/sph_' + year + '.nc'
        r = requests.get(url)
        with open('sph_' + year + '.nc', 'wb') as f:
            f.write(r.content)
    sph_data = xr.open_dataset('sph_' + year + '.nc').sel(day=date)
    sph_data["sph"] = sph_data["specific_humidity"]
    sph_data = sph_data.drop(["specific_humidity"])
    sph_data = sph_data.drop_dims("crs")
    sph = _interpolate(sph_data, "sph", _gridmet2landsat(band10, lat, lon))
    np.save(name, sph.data)
    return sph.data

def _etr(date, band10, lat, lon): 
    os.makedirs("etr", exist_ok=True)
    date_split = date.split("-")
    year = date_split[0]
    name = "etr/" + str(date) + "_" + str(float(lat.min())) + "_" + str(float(lat.max())) + "_" + str(float(lon.min())) + "_" + str(float(lon.max())) + ".npy"
    if path.exists(name):
        return np.load(name)
    if not path.exists("etr_" + year + ".nc"):
        url = 'https://www.northwestknowledge.net/metdata/data/etr_' + year + '.nc'
        r = requests.get(url)
        with open('etr_' + year + '.nc', 'wb') as f:
            f.write(r.content)
    etr_data = xr.open_dataset('etr_' + year + '.nc').sel(day=date)
    etr_data["etr"] = etr_data["potential_evapotranspiration"]
    etr_data = etr_data.drop(["potential_evapotranspiration"])
    etr_data = etr_data.drop_dims("crs")
    etr = _interpolate(etr_data, "etr", _gridmet2landsat(band10, lat, lon))
    np.save(name, etr.data)
    return etr.data

def _elev(band10, lat, lon): 
    os.makedirs("elev", exist_ok=True)
    name = "elev/" + str(float(lat.min())) + "_" + str(float(lat.max())) + "_" + str(float(lon.min())) + "_" + str(float(lon.max())) + ".npy"
    if path.exists(name):
        return np.load(name)
    elev_data = xr.open_dataset("elev.nc").sel(day=1)
    elev = _interpolate(elev_data, "elev", _gridmet2landsat(band10, lat, lon))
    np.save(name, elev.data)
    return elev.data

def _focalStatistics(data, diameter):
    from scipy import signal
    
    arr = np.asarray(data,float)
    ones = (~np.isnan(arr)).astype(float)
    out = np.nan_to_num(arr)

    kernel = np.ones((diameter,diameter))


    # Perform 2D convolution with input data and kernel 
    out2 = signal.convolve2d(out, kernel, boundary='fill', mode='same')
    ones2 = signal.convolve2d(ones, kernel, boundary='fill', mode='same')
    return out2/ones2

def _nearestNeighbor(values, newsize):
    from scipy.interpolate import griddata
    x,y = np.mgrid[0:(values.shape[1]), 0:(values.shape[0])]
    
    x1 = np.linspace(0, values.shape[1]-1, values.shape[1])
    y1 = np.linspace(0, values.shape[0]-1, values.shape[0])
    x, y = np.meshgrid(x1, y1)
    
    points = np.stack((x.flatten(), y.flatten()), axis=1)
    values2 = values.flatten()

    X1 = np.linspace(0, values.shape[1]-1, newsize[1])
    Y1 = np.linspace(0, values.shape[0]-1, newsize[0])
    X, Y = np.meshgrid(X1, Y1)
    
    final = griddata(points, values2, (X, Y), method='nearest')
    return final

def _zonalStatistics(values, shape, function="mean", output_maps=False):
    values = np.asarray(values,float)

    fs = np.asarray(np.arange(shape[1]*shape[0]).reshape((shape[1], shape[0])),float)
    zones = _nearestNeighbor(fs, values.shape)
    tzf = zones.flatten().astype(int)
    tvf = values.flatten()
    
    keep1 = ~np.isnan(tvf)
    data = np.bincount(tzf[keep1],tvf[keep1])/np.bincount(tzf[keep1])
    if(function=="std"):
        data = np.vectorize(dict(enumerate(data.flatten())).get)(zones)
        data = np.square(values - data).flatten()
        keep2 = ~np.isnan(data)
        data = np.sqrt(np.bincount(tzf[keep2],data[keep2])/np.bincount(tzf[keep2]))
        
    elif(function!="std" and function!="mean"):
        print("Function Doesn't Exist")
        return
    
    indices = np.unique(tzf)
    np.put(fs, indices, data)
    
    if output_maps:
        return fs, zones
    
    return fs

def _fillna(values):
    from scipy import interpolate
    from scipy.interpolate import griddata
    
#     grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
    px = np.linspace(0,1,values.shape[1])
    py =  np.linspace(0,1,values.shape[0])
    pX, pY = np.meshgrid(px,py)

    keep = ~np.isnan(values.flatten())
    data = griddata((pX.flatten()[keep], pY.flatten()[keep]), values.flatten()[keep], (pX, pY), method='nearest')

    return data

def _bilinear(values, newsize):
    from scipy import interpolate
    from scipy.interpolate import griddata
    
#     grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear')
    px = np.linspace(0,1,values.shape[1])
    py =  np.linspace(0,1,values.shape[0])
    pX, pY = np.meshgrid(px,py)
    
    
    x = np.linspace(0,1,newsize[1])
    y =  np.linspace(0,1,newsize[0])
    X, Y = np.meshgrid(x,y)

    keep = ~np.isnan(values.flatten())
    data = griddata((pX.flatten()[keep], pY.flatten()[keep]), values.flatten()[keep], (X, Y), method='linear')

    return _fillna(data)

def _mosaic_map(matrices, out_map):
    fs = np.empty(matrices[0].shape)
    fs[:] = np.NaN
    
    for matrix in reversed(matrices):
        fs[~np.isnan(matrix)] = matrix[~np.isnan(matrix)]
    
    return np.vectorize(dict(enumerate(fs.flatten())).get)(out_map)

def _mosaic(matrices):
    fs = np.empty(matrices[0].shape)
    fs[:] = np.NaN
    
    for matrix in reversed(matrices):
        fs[~np.isnan(matrix)] = matrix[~np.isnan(matrix)]
        
    return fs

def _dt(tmax, tmin, rs, ea, elev, doy, lat):
    phi = (lat * (math.pi / 180))
    doy = ((tmax * 0) + doy)
    delta = (np.sin((doy * (2 * math.pi / 365)) - 1.39) * 0.409)
    ws = np.arccos((np.tan(phi) * -1) * np.tan(delta))
    dr = ((np.cos(doy * (2 * math.pi / 365)) * 0.033) + 1)
    ra = (((((ws * np.sin(phi)) * np.sin(delta)) + (np.cos(phi) * np.cos(delta) * np.sin(ws))) * dr) * ((1367.0 / math.pi) * 0.0820))
    rso =  (((elev * 2E-5) + 0.75) * ra)
    fcd =  ((np.clip((rs / rso), 0.3, 1.0) * 1.35) - 0.35)
    rns = (rs * (1 - 0.23)) 
    rnl = (((((tmax**4) + (tmin**4)) * (((np.sqrt(ea) * -0.14)) + 0.34)) * (4.901E-9 * 0.5)) * fcd)
    rn = (rns - rnl)
    pair = (((((elev * -0.0065) + 293.0) / 293.0) ** 5.26) * 101.3)
    den = (((((tmax + tmin) * 0.5) ** -1) * pair) * (3.486 / 1.01))
    dt = ((rn/den) * (110.0 / ((1.013 / 1000) * 86400)))

    return dt

def get_nc_files(year):
    if not isdir("distributed_ssebop\\data"):
        mkdir("distributed_ssebop\\data")

    if not isdir("distributed_ssebop\\data\\gridmet"):
        mkdir("distributed_ssebop\\data\\gridmet")

    mkdir("distributed_ssebop\\data\\gridmet\\" + year)

    url = 'https://www.northwestknowledge.net/metdata/data/tmmn_' + year + '.nc'
    r = requests.get(url)
    with open('distributed_ssebop\\data\\gridmet\\' + year + '\\tmin.nc', 'wb') as f:
        f.write(r.content)

    url = 'https://www.northwestknowledge.net/metdata/data/tmmx_' + year + '.nc'
    r = requests.get(url)
    with open('distributed_ssebop\\data\\gridmet\\' + year + '\\tmax.nc', 'wb') as f:
        f.write(r.content)

    url = 'https://www.northwestknowledge.net/metdata/data/etr_' + year + '.nc'
    r = requests.get(url)
    with open('distributed_ssebop\\data\\gridmet\\' + year + '\\etr.nc', 'wb') as f:
        f.write(r.content)
        
    
def _gridc(lst,ndvi,tmax):
    def _con1(lst, focalNDVI, coldTcorr):
        
        fs = np.empty(coldTcorr.shape)
        fs[:] = np.NaN
        fs[((lst>270) & (focalNDVI>=0.7))] = coldTcorr[((lst>270) & (focalNDVI>=0.7))]
        return fs
                   
    focalNDVI = _focalStatistics(ndvi,4)
    filtTc = lst/tmax
    
#     filtTc = _con1(lst, focalNDVI, filtTc)
    if np.nanmean(filtTc) > 0:
        gridsize = filtTc.shape
        gridsize = (16, 16)
        zonalmean, out_map = _zonalStatistics(filtTc, gridsize, function="mean", output_maps=True)
        zonalstd = _zonalStatistics(filtTc, gridsize ,function="std")
        zonalC = zonalmean - (zonalstd * 2)
        coldgrid = _bilinear(zonalC, gridsize)
        cold1 = _focalStatistics(coldgrid,3)
        cold2 = _focalStatistics(coldgrid,5)
        cold3 = _focalStatistics(coldgrid,7)
        cold4 = _focalStatistics(coldgrid,9)
        cold5 = _focalStatistics(coldgrid,11)
        raslist = [coldgrid,cold1,cold2,cold3,cold4,cold5]
        mosaic = _mosaic_map(raslist, out_map)
        mos2 = _focalStatistics(mosaic,4)
        mos4 = _focalStatistics(mosaic,8)
        mos16 = _focalStatistics(mosaic,12)
        mos40 = _focalStatistics(mosaic,16) 
        weightmean1 = (mosaic * 0.4) + (mos2 * 0.3) + (mos4 * 0.2) + (mos16 * 0.1)
        weightmean2 = (mos2 * 0.5) + (mos4 * 0.33) + (mos16 * 0.17)
        weightmean3 = (mos4 * 0.67) + (mos16 * 0.33)
        raslist2 = [weightmean1,weightmean2,weightmean3,mos40]
        mosaic2 = _mosaic(raslist2)
        final = _focalStatistics(mosaic2, 2)
        finalc = _bilinear(final, final.shape)
        return finalc
    
def ET(filename):
    data = np.load(filename)
    SR_B2 = data[0,1]
    SR_B3 = data[0,2]
    SR_B4 = data[0,3]
    SR_B5 = data[0,4]
    SR_B6 = data[0,5]
    SR_B7 = data[0,6]
    ST_B10 = data[0,7]
    QA_PIXEL = data[0,17]
    LAT = data[0,20]
    LON = data[0,19]
    year, month, day = filename.split("_")[-7:-4]
    DATE = year + "-" + month + "-" + day
#     print("Date: " + DATE)
    doy =  datetime(int(year), int(month), int(day)).timetuple().tm_yday
    blue = (SR_B2*0.0000275) - 0.2
    green = (SR_B3*0.0000275) - 0.2
    red = (SR_B4*0.0000275) - 0.2
    nir = (SR_B5*0.0000275) - 0.2
    swir1 = (SR_B6*0.0000275) - 0.2
    swir2 = (SR_B7*0.0000275) - 0.2
    tir = (ST_B10*0.00341802) + 149.0
    lst = (ST_B10 * 0.00341802) + 149.0
    QA_PIXEL = (QA_PIXEL*1)
#     cloudmask = _cloudmask(QA_PIXEL)
    ndvi = _ndvi(nir, red)
    tmax = _tmax(DATE, red, nir, tir, LAT, LON)
    tmin = _tmin(DATE, red, nir, tir, LAT, LON)
    srad = _srad(DATE, red, nir, tir, LAT, LON)
    sph = _sph(DATE, red, nir, tir, LAT, LON)
    ETr = _etr(DATE, red, nir, tir, LAT, LON)
    elev = _elev(red, nir, tir, LAT, LON)
    pair = ((((elev * -0.0065) + 293.0)/293.0)**5.26 * 101.3)
    rs = srad * 0.0864
    ea = (((((sph * 0.378) + 0.622) ** -1) * sph) * pair)
    dt = _dt(tmax, tmin, rs, ea, elev, doy, LAT)
    cfactor = _gridc(lst,ndvi,tmax)
    
    Tcold = tmax * cfactor
    Thot = Tcold + dt
    ETf = (Thot - lst) / dt
#     ETf[(ETf > 1.0) & (ETf < 2.0)] = 1.0
    ETf[(ETf > 1.05)] = 1.05
    #     ETf[ETf >= 2.0] = np.NaN
    ETf[ETf > 1.3] = np.NaN
    ETf[ETf < 0.0] = 0.0

#     print(cloudmask)
#     ETf[cloudmask == 0.0] = np.NaN
    k = 0.85
    ETr = ETr * k
    ETa = ETf * ETr

    #     ETa[cloudmask == 0.0] = np.NaN
    ETa[ETa < 0.0] == 0.0
    return ETa

class Encoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.in_planes = [32, 64, 128, 256]
        
        self.range1, self.range2 = 0, 2

        self.convolutions = nn.Sequential(nn.Conv2d(2, self.in_planes[0], kernel_size=5, stride=1, padding=2),
#                                         nn.BatchNorm2d(self.in_planes[0]),
                                        nn.LeakyReLU(),
                                        nn.AvgPool2d(kernel_size=2, stride=2),
                                        nn.Conv2d(self.in_planes[0], self.in_planes[1], kernel_size=5, stride=1, padding=2),
#                                         nn.BatchNorm2d(self.in_planes[1]),
                                        nn.LeakyReLU(),
                                        nn.AvgPool2d(kernel_size=2, stride=2),
                                        nn.Conv2d(self.in_planes[1], self.in_planes[2], kernel_size=5, stride=1, padding=2),
#                                         nn.BatchNorm2d(self.in_planes[2]),
                                        nn.LeakyReLU(),
                                        nn.AvgPool2d(kernel_size=2, stride=2),
                                        nn.Conv2d(self.in_planes[2], self.in_planes[3], kernel_size=5, stride=1, padding=2),
#                                         nn.BatchNorm2d(self.in_planes[3]),
                                        nn.LeakyReLU(),
                                        nn.AvgPool2d(kernel_size=2, stride=2),
        
        )
        
        self.metadata_network = torch.nn.Sequential(
            torch.nn.Linear(25, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 64)
        )
                                
        self.fc = nn.Linear(1024 + 64, 1)
#         self.fc = torch.nn.Sequential(
#             torch.nn.Linear(1024 + 64, 256),
#             torch.nn.LeakyReLU(),
#             torch.nn.Linear(256, 1)
#         )


    def forward(self, x, metadata):
        out = self.convolutions(x)
        out = out.view(x.size(0), -1)
        out_metadata = self.metadata_network(metadata)
        
        out = ((torch.sigmoid(self.fc(torch.cat((out, out_metadata), dim=1))) * (self.range2 - self.range1)) + self.range1)
        return out
    
def normalize(input_mat):
    return ((input_mat - input_mat.min())/(input_mat.max() - input_mat.min()))

def SR_B1_Norm(input_mat):
    return ((input_mat - 0)/(50325 - 0))

def SR_B2_Norm(input_mat):
    return ((input_mat - 0)/(52519 - 0))

def SR_B3_Norm(input_mat):
    return ((input_mat - 0)/(51698 - 0))

def SR_B4_Norm(input_mat):
    return ((input_mat - 0)/(53826 - 0))

def SR_B5_Norm(input_mat):
    return ((input_mat - 0)/(54923 - 0))

def SR_B6_Norm(input_mat):
    return ((input_mat - 0)/(27972 - 0))

def SR_B7_Norm(input_mat):
    return ((input_mat - 0)/(25281 - 0))

def ST_B10_Norm(input_mat):
    return ((input_mat - 0)/(55076 - 0))

def LST_TMAX_Norm(input_mat):
    return ((input_mat - 0.48200147021378864)/(1.0916192689820599 - 0.48200147021378864))

class CNN(nn.Module):

    def __init__(self, device):
        super().__init__()
        
        self.Encoder = Encoder()
        self.device=device
        
#     seq_len, batch, input_size
    def forward(self, data, date, lat, lon, metadata):
        
        data = data.cpu().numpy()
        batch_size = data.shape[0]
        
        SR_B1 = data[:,0]
        SR_B2 = data[:,1]
        SR_B3 = data[:,2]
        SR_B4 = data[:,3]
        SR_B5 = data[:,4]
        SR_B6 = data[:,5]
        SR_B7 = data[:,6]
        ST_B10 = data[:,7]
        QA_PIXEL = data[:,17]
        LON = data[:,19]
        LAT = data[:,20]
        
        lat = lat.cpu().numpy().reshape(batch_size, 1, 1, 1)
        lon = lon.cpu().numpy().reshape(batch_size, 1, 1, 1)
    
        lon_d = np.abs(LON.reshape(batch_size, 1, 32, 32)-lon)
        lat_d = np.abs(LAT.reshape(batch_size, 1, 32, 32)-lat)
        dif = lon_d + lat_d
        et_x = np.array([np.where( mat == mat.min())[1][0] for mat in dif])
        et_y = np.array([np.where( mat == mat.min())[2][0] for mat in dif])
        
        year = [date1.split("_")[0] for date1 in date]
        month = [date1.split("_")[1] for date1 in date]
        day = [date1.split("_")[2] for date1 in date]
        
        date_str = [dstr.replace("_", "-") for dstr in date]
        doy = [datetime(int(year[i]), int(month[i]), int(day[i])).timetuple().tm_yday for i in range(batch_size)]

        aerosol = (SR_B1*0.0000275) - 0.2
        blue = (SR_B2*0.0000275) - 0.2
        green = (SR_B3*0.0000275) - 0.2
        red = (SR_B4*0.0000275) - 0.2
        nir = (SR_B5*0.0000275) - 0.2
        swir1 = (SR_B6*0.0000275) - 0.2
        swir2 = (SR_B7*0.0000275) - 0.2
        lst = (ST_B10 * 0.00341802) + 149.0
        
        ndvi = _ndvi(nir, red)
        
        tmax = np.array([_tmax(date_str[i], lst[i], LAT[i], LON[i]) for i in range(batch_size)])
        tmin = np.array([_tmin(date_str[i], lst[i], LAT[i], LON[i]) for i in range(batch_size)])
        srad = np.array([_srad(date_str[i], lst[i], LAT[i], LON[i]) for i in range(batch_size)])
        sph = np.array([_sph(date_str[i], lst[i], LAT[i], LON[i]) for i in range(batch_size)])
        ETr = np.array([_etr(date_str[i], lst[i], LAT[i], LON[i]) for i in range(batch_size)])
        elev = np.array([_elev(lst[i], LAT[i], LON[i]) for i in range(batch_size)])
        pair = ((((elev * -0.0065) + 293.0)/293.0)**5.26 * 101.3)
        rs = srad * 0.0864
        ea = (((((sph * 0.378) + 0.622) ** -1) * sph) * pair)
        dt = np.array([_dt(tmax[i], tmin[i], rs[i], ea[i], elev[i], doy[i], LAT[i]) for i in range(batch_size)])
        
        SR_B1 = torch.from_numpy(SR_B1).to(device=self.device, dtype=torch.float32)
        SR_B2 = torch.from_numpy(SR_B2).to(device=self.device, dtype=torch.float32)
        SR_B3 = torch.from_numpy(SR_B3).to(device=self.device, dtype=torch.float32)
        SR_B4 = torch.from_numpy(SR_B4).to(device=self.device, dtype=torch.float32)
        SR_B5 = torch.from_numpy(SR_B5).to(device=self.device, dtype=torch.float32)
        SR_B6 = torch.from_numpy(SR_B6).to(device=self.device, dtype=torch.float32)
        SR_B7 = torch.from_numpy(SR_B7).to(device=self.device, dtype=torch.float32)
        ST_B10 = torch.from_numpy(ST_B10).to(device=self.device, dtype=torch.float32)
        
        lst = torch.from_numpy(lst).to(device=self.device, dtype=torch.float32)
        ndvi = torch.from_numpy(ndvi).to(device=self.device, dtype=torch.float32)
        np.save("NDVI_" + date[0] + ".npy", ndvi.cpu().detach().numpy())
        np.save("LST_" + date[0] + ".npy", lst.cpu().detach().numpy())
        tmax = torch.from_numpy(tmax).to(device=self.device, dtype=torch.float32)
        tmin = torch.from_numpy(tmin).to(device=self.device, dtype=torch.float32)
        aerosol = torch.from_numpy(aerosol).to(device=self.device, dtype=torch.float32)
        blue = torch.from_numpy(blue).to(device=self.device, dtype=torch.float32)
        green = torch.from_numpy(green).to(device=self.device, dtype=torch.float32)
        red = torch.from_numpy(red).to(device=self.device, dtype=torch.float32)
        nir = torch.from_numpy(nir).to(device=self.device, dtype=torch.float32)
        swir1 = torch.from_numpy(swir1).to(device=self.device, dtype=torch.float32)
        swir2 = torch.from_numpy(swir2).to(device=self.device, dtype=torch.float32)
        
        dt = torch.from_numpy(dt).to(device=self.device, dtype=torch.float32)
        ETr = torch.from_numpy(ETr).to(device=self.device, dtype=torch.float32)

        cfactor_input = torch.cat([
                                    LST_TMAX_Norm(lst/tmax).view(batch_size, 1, 32, 32),
                                    ndvi.view(batch_size, 1, 32, 32),
#                                     SR_B1_Norm(SR_B1).view(batch_size, 1, 32, 32),
#                                     SR_B2_Norm(SR_B2).view(batch_size, 1, 32, 32),
#                                     SR_B3_Norm(SR_B3).view(batch_size, 1, 32, 32),
#                                     SR_B4_Norm(SR_B4).view(batch_size, 1, 32, 32),
#                                     SR_B5_Norm(SR_B5).view(batch_size, 1, 32, 32),
#                                     SR_B6_Norm(SR_B6).view(batch_size, 1, 32, 32),
#                                     SR_B7_Norm(SR_B7).view(batch_size, 1, 32, 32),
#                                     ST_B10_Norm(ST_B10).view(batch_size, 1, 32, 32)
                                       ], dim=1)
        
        batch_list = list(range(batch_size))

        cfactor = self.Encoder(cfactor_input, metadata).view(-1)
        
        np.save("CFactor_" + date[0] + ".npy", cfactor.cpu().detach().numpy())
    
        self.tmax = tmax[batch_list, et_x, et_y].view(-1)
        self.Tcold = self.tmax * cfactor      
        
        self.dt = dt[batch_list, et_x, et_y].view(-1)
        self.Thot = self.Tcold + self.dt
        
        
        self.lst = lst[batch_list, et_x, et_y].view(-1)
        
        self.ETf = (self.Thot - self.lst) / self.dt
        
        np.save("ETF_" + date[0] + ".npy", self.ETf.cpu().detach().numpy())
        
        self.k = 0.85
        
        self.ETr = ETr[batch_list, et_x, et_y].view(-1)
        
        np.save("ETR_" + date[0] + ".npy", self.ETr.cpu().detach().numpy())
        
        ETa = self.ETf * self.ETr * self.k
        
        np.save("ET_" + date[0] + ".npy", ETa.cpu().detach().numpy())
        
        return ETa, cfactor
    
    def reverse_Cfactor(self, ET):
        ETf = ET/(self.ETr * self.k)
        Thot = (ETf * self.dt) + self.lst
        Tcold = Thot - self.dt
        cfactor = Tcold/self.tmax
        return cfactor
        
class LysiDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self):
        
        self.file_names = glob.glob("lysidata/*")
        self.vegs = ['DBF', 'OSH', 'GRA', 'WET', 'SAV', 'ENF', 'MF', 'WSA', 'CRO']
        self.clims = ['Dfb', 'Bwk', 'Cfa', 'Cwa', 'Dwb', 'Dfc', 'Dfa', 'Bsk', 'Csa', 'Bsh']
        
        print("Dataset Length " + str(len(self.file_names)))
        

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        lon, lat, elev, veg, clim, geohash, year, month, day, cloud_coverage, pixel_coverage, true_et, pred_et = self.file_names[idx].split("_")[-13:]
        img = torch.from_numpy(np.load(self.file_names[idx]).astype(float))
        et = float(self.file_names[idx].split("_")[-1].replace(".npy", ""))
        date = "_".join(self.file_names[idx].split("_")[-7:-4])
        lat = float(self.file_names[idx].split("_")[-12])
        lon = float(self.file_names[idx].split("_")[-13])
        elev = np.array([float(self.file_names[idx].split("_")[-11])/8848.0])
        veg = torch.nn.functional.one_hot(torch.tensor(self.vegs.index(self.file_names[idx].split("_")[-10])), num_classes=len(self.vegs))
        clim = torch.nn.functional.one_hot(torch.tensor(self.clims.index(self.file_names[idx].split("_")[-9])), num_classes=len(self.clims))
        year = self.file_names[idx].split("_")[-7]
        month = self.file_names[idx].split("_")[-6]
        day = self.file_names[idx].split("_")[-5]
        
        date_time_obj = datetime.strptime(year + '_' + month + '_' + day, '%Y_%m_%d')
        day_of_year = date_time_obj.timetuple().tm_yday
        day_sin = torch.tensor([np.sin(2 * np.pi * day_of_year/364.0)])
        day_cos = torch.tensor([np.cos(2 * np.pi * day_of_year/364.0)])
        
        x_coord = torch.tensor([np.sin(math.pi/2-np.deg2rad(lat)) * np.cos(np.deg2rad(lon))])
        y_coord = torch.tensor([np.sin(math.pi/2-np.deg2rad(lat)) * np.sin(np.deg2rad(lon))])
        z_coord = torch.tensor([np.cos(math.pi/2-np.deg2rad(lat))])
        
        img = interpolate(img , size=32)[0]
        if img[20].mean() < 0:
            lon_img = img[20].clone()
            lat_img = img[19].clone()
        else:
            lat_img = img[20].clone()
            lon_img = img[19].clone()
            
        img[19] = lon_img
        img[20] = lat_img
            
        return img, et, date, lon, lat, veg, clim, day_sin, day_cos, x_coord, y_coord, z_coord, elev       
    
class EvapoDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, train=True, split=1):
        
        
        self.file_names = []
        for i in range(1, 6):
            if (train and (i != split)):
                self.file_names = self.file_names + glob.glob("split_data/" + str(split) + "/*")
                
            if (train is False and (i == split)):
                self.file_names = self.file_names + glob.glob("split_data/" + str(split) + "/*")
              
        self.vegs = ['DBF', 'OSH', 'GRA', 'WET', 'SAV', 'ENF', 'MF', 'WSA', 'CRO']
        self.clims = ['Dfb', 'Bwk', 'Cfa', 'Cwa', 'Dwb', 'Dfc', 'Dfa', 'Bsk', 'Csa', 'Bsh']
        
        print("Dataset Length " + str(len(self.file_names)))
        

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        lon, lat, elev, veg, clim, geohash, year, month, day, cloud_coverage, pixel_coverage, true_et, pred_et = self.file_names[idx].split("_")[-13:]
        img = torch.from_numpy(np.load(self.file_names[idx]).astype(float))
        et = float(self.file_names[idx].split("_")[-1].replace(".npy", ""))
        date = "_".join(self.file_names[idx].split("_")[-7:-4])
        lat = float(self.file_names[idx].split("_")[-12])
        lon = float(self.file_names[idx].split("_")[-13])
        elev = np.array([float(self.file_names[idx].split("_")[-11])/8848.0])
        veg = torch.nn.functional.one_hot(torch.tensor(self.vegs.index(self.file_names[idx].split("_")[-10])), num_classes=len(self.vegs))
        clim = torch.nn.functional.one_hot(torch.tensor(self.clims.index(self.file_names[idx].split("_")[-9])), num_classes=len(self.clims))
        year = self.file_names[idx].split("_")[-7]
        month = self.file_names[idx].split("_")[-6]
        day = self.file_names[idx].split("_")[-5]
        
        date_time_obj = datetime.strptime(year + '_' + month + '_' + day, '%Y_%m_%d')
        day_of_year = date_time_obj.timetuple().tm_yday
        day_sin = torch.tensor([np.sin(2 * np.pi * day_of_year/364.0)])
        day_cos = torch.tensor([np.cos(2 * np.pi * day_of_year/364.0)])
        
        x_coord = torch.tensor([np.sin(math.pi/2-np.deg2rad(lat)) * np.cos(np.deg2rad(lon))])
        y_coord = torch.tensor([np.sin(math.pi/2-np.deg2rad(lat)) * np.sin(np.deg2rad(lon))])
        z_coord = torch.tensor([np.cos(math.pi/2-np.deg2rad(lat))])
        
        img = interpolate(img , size=32)[0]
        
        if img[20].mean() < 0:
            lon_img = img[20].clone()
            lat_img = img[19].clone()
        else:
            lat_img = img[20].clone()
            lon_img = img[19].clone()
            
        img[19] = lon_img
        img[20] = lat_img
            
        return img, et, date, lon, lat, veg, clim, day_sin, day_cos, x_coord, y_coord, z_coord, elev, self.file_names[idx]       
    
class TrainCNN():

    def __init__(self, epochs=500, batch_size=32, torch_type=torch.float32, random_seed=18, split=1):
        super(TrainCNN, self).__init__()
        
        torch.cuda.manual_seed(18)
        torch.cuda.manual_seed_all(18)
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = "cpu"
        if torch.cuda.is_available(): self.device = "cuda"
        self.torch_type = torch_type
        self.split = split
        
        self.mse = torch.nn.MSELoss()
        self.l1 = torch.nn.L1Loss()
        self.model = CNN(self.device).to(self.device, dtype=torch.float32)
        
        self.train_dataset = EvapoDataset(split=self.split, train=True)
        self.test_dataset = EvapoDataset(split=self.split, train=False)
        self.lysidataset = LysiDataset()
        self.len = len(self.train_dataset)

        self.train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=5)
        
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset,batch_size=32, num_workers=5)
        self.final_accuracy_loader = torch.utils.data.DataLoader(self.test_dataset)
        self.lysi_loader = torch.utils.data.DataLoader(self.lysidataset,batch_size=8)
        
        #self.opt = torch.optim.AdamW(self.model.parameters(), lr=0.00001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        #self.sched = torch.optim.lr_scheduler.StepLR(self.opt, step_size=10, gamma=0.1)
#         self.opt = torch.optim.Adagrad(self.model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
#         self.sched = torch.optim.lr_scheduler.StepLR(self.opt, step_size=25, gamma=0.5)
#         self.opt = torch.optim.SGD(self.model.Encoder.parameters(), lr=0.01)
#         self.opt = torch.optim.AdamW(self.model.Encoder.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
        self.opt = torch.optim.Adagrad(self.model.Encoder.parameters(), lr=0.001, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10)
        self.sched = torch.optim.lr_scheduler.StepLR(self.opt, step_size=25, gamma=0.5)
        self.best_lysi = 9999
    
    def train(self):
        for epoch in range(self.epochs):
            for ind, (img_seq, et, date, lat, lon, veg, clim, day_sin, day_cos, x_coord, y_coord, z_coord, elev, filename) in enumerate(self.train_loader):
                img_seq = img_seq.to(device=self.device, dtype=torch.float32)
                et = et.to(device=self.device, dtype=torch.float32)
                lat = lat.to(device=self.device, dtype=torch.float32)
                lon = lon.to(device=self.device, dtype=torch.float32)
                veg = veg.to(device=self.device, dtype=torch.float32)
                clim = clim.to(device=self.device, dtype=torch.float32)
                day_sin = day_sin.to(device=self.device, dtype=torch.float32)
                day_cos = day_cos.to(device=self.device, dtype=torch.float32)
                x_coord = x_coord.to(device=self.device, dtype=torch.float32)
                y_coord = y_coord.to(device=self.device, dtype=torch.float32)
                z_coord = z_coord.to(device=self.device, dtype=torch.float32)
                elev = elev.to(device=self.device, dtype=torch.float32)
                output_ET, output_cfactor = self.model(img_seq, date, lat, lon, torch.cat((veg, clim, day_sin, day_cos, elev, x_coord, y_coord, z_coord), dim=1))
                true_cfactor = self.model.reverse_Cfactor(et)
                loss_et = self.l1(output_ET, et) / 100
                loss_cfactor = self.mse(output_cfactor, true_cfactor) * 100
                loss = loss_cfactor + loss_et
                loss.backward()
                self.opt.step()
            self.sched.step()
#                 print("===> " + str(ind + 1) + "/" + str(int(self.len/self.batch_size)) + ", Loss: " + str(float(loss.data)) + ", Predicted ET: " + str(output_cfactor.data) + ", True ET: " + str(true_cfactor.data))
#             print("===> " + str(epoch + 1)  + ", Loss: " + str(round(float(loss.data), 4)) + "(" + str(round(float(loss_cfactor.data), 4)) + "+" + str(round(float(loss_et.data), 4)) + "), Predicted ET: " + str(output_cfactor.data) + ", True ET: " + str(true_cfactor.data))
                
            self.test_accuracy_et, self.test_accuracy_cfactor = self.test(epoch)
            self.lysi_accuracy_et, self.lysi_accuracy_cfactor = self.lysi(epoch)
            print("Epoch " + str(epoch + 1) + ", Test(" + self.test_accuracy_et + "/" + self.test_accuracy_cfactor + "), LYSI(" + self.lysi_accuracy_et + "/" + self.lysi_accuracy_cfactor + ")")
            
#             if (self.best_lysi > float(self.lysi_accuracy_et)):
#                 torch.save(self.model, "MLBOP_LYSI.pt" )
#                 with open('lysiError.txt', "w") as myfile:
#                     myfile.write("TEST(" + self.test_accuracy_et + "/" + self.test_accuracy_cfactor + "), LYSI(" + self.lysi_accuracy_et + "/" + self.lysi_accuracy_cfactor)
#                 self.best_lysi = float(self.lysi_accuracy_et)
        self.final_accuracy(epoch)       
        torch.save(self.model, "MLBOP_" + str(self.split) + ".pt" )
    
    def test(self, epoch):
        with torch.no_grad():
            et_correct = 0
            cfactor_correct = 0
            counter = 0
            for img_seq, et, date, lat, lon, veg, clim, day_sin, day_cos, x_coord, y_coord, z_coord, elev, filename in self.test_loader:
                img_seq = img_seq.to(device=self.device, dtype=torch.float32)
                et = et.to(device=self.device, dtype=torch.float32)
                lat = lat.to(device=self.device, dtype=torch.float32)
                lon = lon.to(device=self.device, dtype=torch.float32)
                veg = veg.to(device=self.device, dtype=torch.float32)
                clim = clim.to(device=self.device, dtype=torch.float32)
                day_sin = day_sin.to(device=self.device, dtype=torch.float32)
                day_cos = day_cos.to(device=self.device, dtype=torch.float32)
                x_coord = x_coord.to(device=self.device, dtype=torch.float32)
                y_coord = y_coord.to(device=self.device, dtype=torch.float32)
                z_coord = z_coord.to(device=self.device, dtype=torch.float32)
                elev = elev.to(device=self.device, dtype=torch.float32)
                output_ET, output_cfactor = self.model(img_seq, date, lat, lon, torch.cat((veg, clim, day_sin, day_cos, elev, x_coord, y_coord, z_coord), dim=1))
                true_cfactor = self.model.reverse_Cfactor(et)
                et_correct += (torch.sum(torch.abs((output_ET-et))))
                cfactor_correct += (torch.sum(torch.abs((output_cfactor-true_cfactor))))
                counter += output_ET.shape[0]
            
            return str(round(float(et_correct.sum() / counter), 4)), str(round(float(cfactor_correct.sum() / counter), 4))
        
    def lysi(self, epoch):
        with torch.no_grad():
            et_correct = 0
            cfactor_correct = 0
            counter = 0
            for img_seq, et, date, lat, lon, veg, clim, day_sin, day_cos, x_coord, y_coord, z_coord, elev in self.lysi_loader:
                img_seq = img_seq.to(device=self.device, dtype=torch.float32)
                et = et.to(device=self.device, dtype=torch.float32)
                lat = lat.to(device=self.device, dtype=torch.float32)
                lon = lon.to(device=self.device, dtype=torch.float32)
                veg = veg.to(device=self.device, dtype=torch.float32)
                clim = clim.to(device=self.device, dtype=torch.float32)
                day_sin = day_sin.to(device=self.device, dtype=torch.float32)
                day_cos = day_cos.to(device=self.device, dtype=torch.float32)
                x_coord = x_coord.to(device=self.device, dtype=torch.float32)
                y_coord = y_coord.to(device=self.device, dtype=torch.float32)
                z_coord = z_coord.to(device=self.device, dtype=torch.float32)
                elev = elev.to(device=self.device, dtype=torch.float32)
                output_ET, output_cfactor = self.model(img_seq, date, lat, lon, torch.cat((veg, clim, day_sin, day_cos, elev, x_coord, y_coord, z_coord), dim=1))
                true_cfactor = self.model.reverse_Cfactor(et)
                et_correct += (torch.sum(torch.abs((output_ET-et))))
                cfactor_correct += (torch.sum(torch.abs((output_cfactor-true_cfactor))))
                counter += output_ET.shape[0]
                
            return str(round(float(et_correct.sum() / counter), 4)), str(round(float(cfactor_correct.sum() / counter), 4))
        
    def final_accuracy(self, epoch):
        with open(str(self.split) + '.txt', 'w') as f:
            with torch.no_grad():
                et_correct = 0
                cfactor_correct = 0
                counter = 0
                for img_seq, et, date, lat, lon, veg, clim, day_sin, day_cos, x_coord, y_coord, z_coord, elev, filename in self.final_accuracy_loader:
                    img_seq = img_seq.to(device=self.device, dtype=torch.float32)
                    et = et.to(device=self.device, dtype=torch.float32)
                    lat = lat.to(device=self.device, dtype=torch.float32)
                    lon = lon.to(device=self.device, dtype=torch.float32)
                    veg = veg.to(device=self.device, dtype=torch.float32)
                    clim = clim.to(device=self.device, dtype=torch.float32)
                    day_sin = day_sin.to(device=self.device, dtype=torch.float32)
                    day_cos = day_cos.to(device=self.device, dtype=torch.float32)
                    x_coord = x_coord.to(device=self.device, dtype=torch.float32)
                    y_coord = y_coord.to(device=self.device, dtype=torch.float32)
                    z_coord = z_coord.to(device=self.device, dtype=torch.float32)
                    elev = elev.to(device=self.device, dtype=torch.float32)
                    output_ET, output_cfactor = self.model(img_seq, date, lat, lon, torch.cat((veg, clim, day_sin, day_cos, elev, x_coord, y_coord, z_coord), dim=1))
                    true_cfactor = self.model.reverse_Cfactor(et)
                    et_correct += (torch.sum(torch.abs((output_ET-et))))
                    cfactor_correct += (torch.sum(torch.abs((output_cfactor-true_cfactor))))
                    counter += output_ET.shape[0]
                    
                    
                    f.write(filename[0].split("/")[-1] + ', ' + str(float(output_ET[0])) + ', ' + str(float(output_cfactor[0])) + ', ' + str(float(et[0])) + ', ' + str(float(true_cfactor[0])) + ', ' + str(float(torch.abs(output_ET[0]-et[0]))) + ', '+ str(float(torch.abs(output_cfactor[0]-true_cfactor[0]))) + "\n")
            f.close()
            
            
# if __name__ == '__main__':
#     for r in range(1, 6):
#         trainer = TrainCNN(split=r)
#         trainer.train()
#         print("DONE " + str(r) + " SPLIT")
        
        
if __name__ == '__main__':
    for r in range(1, 6):
        model  = torch.load("MLBOP_" + str(r) + ".pt")
        
        lysi_loader = torch.utils.data.DataLoader(EvapoDataset(split=r, train=False))
        with torch.no_grad():
            et_correct = 0
            cfactor_correct = 0
            counter = 0
            for img_seq, et, date, lat, lon, veg, clim, day_sin, day_cos, x_coord, y_coord, z_coord, elev, filename in lysi_loader:
                if not ("US-An1" in filename[0]):
                    continue
                    
                img_seq = img_seq.to(device="cuda", dtype=torch.float32)
                et = et.to(device="cuda", dtype=torch.float32)
                lat = lat.to(device="cuda", dtype=torch.float32)
                lon = lon.to(device="cuda", dtype=torch.float32)
                veg = veg.to(device="cuda", dtype=torch.float32)
                clim = clim.to(device="cuda", dtype=torch.float32)
                day_sin = day_sin.to(device="cuda", dtype=torch.float32)
                day_cos = day_cos.to(device="cuda", dtype=torch.float32)
                x_coord = x_coord.to(device="cuda", dtype=torch.float32)
                y_coord = y_coord.to(device="cuda", dtype=torch.float32)
                z_coord = z_coord.to(device="cuda", dtype=torch.float32)
                elev = elev.to(device="cuda", dtype=torch.float32)
                output_ET, output_cfactor = model(img_seq, date, lat, lon, torch.cat((veg, clim, day_sin, day_cos, elev, x_coord, y_coord, z_coord), dim=1))
                true_cfactor = model.reverse_Cfactor(et)
                et_correct += (torch.sum(torch.abs((output_ET-et))))
                cfactor_correct += (torch.sum(torch.abs((output_cfactor-true_cfactor))))
                counter += output_ET.shape[0]
                print("HERE")
        print("DONE " + str(r) + " SPLIT")
        