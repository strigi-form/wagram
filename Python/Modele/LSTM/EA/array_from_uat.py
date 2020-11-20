"""Request MeteoNet single timeserie, return as a numpy array"""
import numpy as np
import lisptick

HOST = "uat.lisptick.org"
PORT = 12006

DIRECTIONVENT = '@"dd"'
VITESSEVENT = '@"ff"'
PRECIPITATION = '@"precip"'
HUMIDITE = '@"hu"'
POINTDEROSEE = '@"td"'
TEMPERATURE = '@"t"'
PRESSION = '@"psl"'

def get_array(field, station, start, stop):
    """retreive meteonet timeserie values (no timestamp) as a numpy array"""
    conn = lisptick.Socket(HOST, PORT)
    request = " ".join(["(timeserie", field, '"meteonet"', '"'+str(station)+'"', start, stop, ")"])
    array = []
    def inner_append(_, __, point):
        """append value to local to local array"""
        array.append(point.i)
    conn.walk_result(request, inner_append)
    return np.array([array])

def main():
    """simple temprature request"""
    serie = get_array(TEMPERATURE, 14578001, "2016-01-01", "2016-01-12")
    print(serie)

if __name__ == "__main__":
    main()
