import xml
import xml.etree.ElementTree as ET
from os import listdir
from os.path import isfile, join
import pickle
from datetime import datetime

def deg2dec(deg_dict):
    ans = int(deg_dict['degrees']) + int(deg_dict['minutes'])/60.0
    if 'seconds' in deg_dict:
        ans = ans + int(deg_dict['seconds'])/3600.0
    dir = deg_dict['direction']
    if dir == 'SOUTH' or dir == 'WEST':
        ans = ans * -1.0
    return ans

def str2time(s):
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")


def fl2meters(alt):
    return alt*100.0 * 0.3048

class Flight(object):

    def __init__(self, speed, lat, lon, alt, next_lat, next_lon, tim, arr, dep, ref):

        self.speed = [speed]
        self.latitude =[lat]
        self.longitude = [lon]
        self.altitude = [alt]
        self.time = [tim]
        self.arrival = arr
        self.departure = dep
        self.ref = ref
        self.next_latitude = [next_lat]
        self.next_longtiude = [next_lon]

    def update(self,speed, lat, lon, alt, next_lat, next_lon, tim):
        self.speed.append(speed)
        self.latitude.append(lat)
        self.longitude.append(lon)
        self.altitude.append(alt)
        self.time.append(tim)
        self.next_latitude.append(next_lat)
        self.next_longtiude.append(next_lon)