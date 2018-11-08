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


def main():

    ns = {'ds': "urn:us:gov:dot:faa:atm:tfm:tfmdataservice",
          'xmlns': "urn:us:gov:dot:faa:atm:tfm:tfmdataservice",
          'fdm': "urn:us:gov:dot:faa:atm:tfm:flightdata",
          'nxce': "urn:us:gov:dot:faa:atm:tfm:tfmdatacoreelements",
          'nxcm': "urn:us:gov:dot:faa:atm:tfm:flightdatacommonmessages",
          'xsi': "http://www.w3.org/2001/XMLSchema-instance"}

    #mypath = './20160101/'
    #mypath = '//adaptshare/SWIM/public/tfms-data/2016/20160130'

    year = '2016'
    day_str = year + '0113'
    mypath = '//adaptshare/SWIM/public/tfms-data/' + year + '/' + day_str
    save_fname = 'flights' + day_str + '.pkl'

    txtfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

    flights = dict()
    filt_airport = 'KSEA'

    for i,f in enumerate(txtfiles):
        if i % 1000 == 0:
            print (f)

        try:
            tree = ET.parse(join(mypath, f))
            root = tree.getroot()

            for child1 in root.getchildren():  # tfmdataservice
                for child2 in child1.getchildren():  # fltdMessage
                    if child2.tag == '{urn:us:gov:dot:faa:atm:tfm:flightdata}fltdMessage':

                        next_lat = 0.0
                        next_lon = 0.0

                        ref = int(child2.get('flightRef'))
                        tim = str2time(child2.get('sourceTimeStamp'))

                        n_speed = child2.find("./fdm:trackInformation/nxcm:speed", ns)
                        n_dep = child2.find("./fdm:trackInformation/nxcm:qualifiedAircraftId/nxce:departurePoint/nxce:airport", ns)
                        n_arr = child2.find(
                                "./fdm:trackInformation/nxcm:qualifiedAircraftId/nxce:arrivalPoint/nxce:airport", ns)
                        n_lat = child2.find(
                                "./fdm:trackInformation/nxcm:position/nxce:latitude/nxce:latitudeDMS", ns)
                        n_alt =  child2.find(
                                "./fdm:trackInformation/nxcm:reportedAltitude/nxce:assignedAltitude/nxce:simpleAltitude", ns)
                        n_lon = child2.find(
                                "./fdm:trackInformation/nxcm:position/nxce:longitude/nxce:longitudeDMS", ns)

                        if n_speed is None or n_dep is None or n_arr is None or n_lat is None or n_alt is None or n_lon is None:
                            continue

                        speed = int(n_speed.text)
                        dep = n_dep.text
                        arr = n_arr.text
                        lat = deg2dec(dict(n_lat.items()))
                        lon = deg2dec(dict(n_lon.items()))
                        alt = fl2meters(int(n_alt.text[0:3]))

                        for info in child2.findall(
                                "./fdm:trackInformation/nxcm:ncsmTrackData/nxcm:nextEvent", ns):
                            next = dict(info.items())
                            next_lat = float(next['latitudeDecimal'])
                            next_lon = float(next['longitudeDecimal'])

                        if arr == filt_airport or dep == filt_airport:  # arrive or depart SEA
                            flight = flights.get(ref)
                            if flight is None:
                                flight = Flight(speed, lat, lon, alt, next_lat, next_lon, tim, arr, dep, ref)
                                flights[ref] = flight
                            else:
                                flight.update(speed, lat, lon, alt, next_lat, next_lon, tim)


        except xml.etree.ElementTree.ParseError:
            pass


    print('saving')
    pickle.dump(flights, open(save_fname,'wb') )


if __name__== "__main__":
  main()