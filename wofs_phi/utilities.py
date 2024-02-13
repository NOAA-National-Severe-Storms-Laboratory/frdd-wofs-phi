import os
#import paramiko
#from scp import SCPClient
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import pyproj
from pyproj.crs import ProjectedCRS
from pyproj.crs.coordinate_operation import AzimuthalEquidistantConversion
from shapely.ops import transform
from shapely.geometry import Point
import geopandas as gpd

def copy_torp(write_dir):
    ssh_client=paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh_client.connect(hostname='myrorss2',username='ryan.martz',password='jjBinKs-4801')
    sftp = ssh_client.open_sftp()

    for date in sftp.listdir('/work/thea.sandmael/radar'):
        sftp.chdir('/work/thea.sandmael/radar/' + date)
        for site in sftp.listdir('/work/thea.sandmael/radar/' + date):
            csv_dir = '/work/thea.sandmael/radar/' + date + '/' + site + '/netcdf/torp/TORPcsv/00.50/'
            try:
                sftp.chdir(csv_dir)
            except:
                continue

            for file in sftp.listdir():
                if os.path.exists(write_dir):
                    sftp.get(csv_dir + file, write_dir + file)
                else:
                    os.mkdir(write_dir)
                    sftp.get(csv_dir + file, write_dir + file)

def geodesic_point_buffer(lon, lat, km):
    proj_crs = ProjectedCRS(conversion = AzimuthalEquidistantConversion(lat, lon))
    proj_wgs84 = pyproj.Proj('EPSG:4326')
    Trans = pyproj.Transformer.from_proj(proj_crs,proj_wgs84,always_xy=True).transform

    return transform(Trans, Point(0, 0).buffer(km * 1000))