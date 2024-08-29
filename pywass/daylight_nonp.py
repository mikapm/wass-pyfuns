#!/usr/bin/env python3
"""
Compute sunrise and sunset hours for given location. 
This version does not use the numpy library.

Following example in
https://en.wikipedia.org/wiki/Sunrise_equation
"""
import sys
from math import pi, sin, cos, ceil, asin, acos, radians, degrees
from datetime import datetime as DT
from datetime import timedelta as TD
from datetime import timezone

def julian_day_number_to_gregorian(jdn):
    """
    Convert the Julian Day Number to the proleptic Gregorian 
    Year, Month, Day.
    Borrowed from
    https://orbital-mechanics.space/reference/julian-date.html
    """
    L = jdn + 68569
    N = int(4 * L / 146_097)
    L = L - int((146097 * N + 3) / 4)
    I = int(4000 * (L + 1) / 1_461_001)
    L = L - int(1461 * I / 4) + 31
    J = int(80 * L / 2447)
    day = L - int(2447 * J / 80)
    L = int(J / 11)
    month = J + 2 - 12 * L
    year = 100 * (N - 49) + I + L
    return year, month, day

def julian_date_to_gregorian(jd):
    """
    Convert a decimal Julian Date to the equivalent proleptic 
    Gregorian date and time (UTC).
    Borrowed from
    https://orbital-mechanics.space/reference/julian-date.html
    """
    jdn = int(jd)
    year, month, day = julian_day_number_to_gregorian(jdn)
    offset = TD(days=(jd % 1), hours=+12)
    dt = DT(year=year, month=month, day=day, tzinfo=timezone.utc)
    return dt + offset

def sunrise_sunset(date, lat=56.549197, lon=3.209986, utc=True):
    """
    Compute sunrise/set hours (UTC) for date + location given 
    by lat, lon (in decimal degrees) following example in
    https://en.wikipedia.org/wiki/Sunrise_equation

    Parameters:
        date - datetime.datetime object
        lat - float; latitude of location (decimal deg)
                     (north is positive, south is negative)
                     Default: Ekofisk lat.
        lon - float; longitude of location (decimal deg)
                     (west is negative, east is positive)
                     Default: Ekofisk lon.
        utc - bool; if True returns UTC times, otherwise 
                    returns local times based on lat, lon
    Returns:
        sunrise - datetime.datetime obj. (timezone UTC)
        sunset - datetime.datetime obj. (timezone UTC)
    """
    latr = lat * (pi/180) # Latitude in radians
    # Current Julian day
    jdate = date.toordinal() + 1721424.5
    # Calculate number of days since Jan 1st, 2000 12:00
    n = ceil(jdate - 2451545.0 + 0.0008)
    # Approximation of mean solar time at integer n expressed 
    # as a Julian day
    js = n - lon/360 # deg
    # Solar mean anomaly
    Md = (357.5291 + 0.98560028 * js) % 360 # Degrees
    Mr = Md * (pi/180) # Radians
    # Equation of the center
    C = 1.9148*sin(Mr) + 0.02*sin(2*Mr) + 0.0003*sin(3*Mr)
    # Ecliptic longitude
    ld = (Md + C + 180 + 102.9372) % 360 # deg
    lr = ld * (pi/180) # rad
    # Solar transit (deg)
    jt = (2451545 + js + 0.0053*sin(Mr) - 0.0069*sin(2*lr))
    # Declination of the Sun (rad)
    delta = asin(sin(lr) * sin(23.4397*(pi/180)))
    # Hour angle omega_0 (deg)
    nom = sin(-0.833*(pi/180)) - sin(latr)*sin(delta)
    denom = cos(latr) * cos(delta)
    omega_0 = degrees(acos(nom / denom))
    # Calculate sunrise and sunset
    jrise = jt - omega_0/360 # Julian date
    jset = jt + omega_0/360 # Julian date
    # Convert to UTC 
    sunrise = (julian_date_to_gregorian(jrise))
    sunset = (julian_date_to_gregorian(jset))
    # Convert to local timezone?
    if not utc:
        import pytz
        from timezonefinder import TimezoneFinder
        # Get timezone for requested location
        tf = TimezoneFinder() # Init class
        tzn = tf.timezone_at(lng=lon, lat=lat)
        tz = pytz.timezone(tzn)
        # Get timezone offset (seconds) from UTC
        # utc_offset = tz.utcoffset(sunrise).seconds
        utc_offset = sunrise.replace(tzinfo=pytz.utc).astimezone(tz).utcoffset().total_seconds()
        # Apply offset to sunrise + sunset times
        sunrise += TD(seconds=utc_offset)
        sunset += TD(seconds=utc_offset)

    return sunrise, sunset

# Main script
if __name__ == '__main__':
    from argparse import ArgumentParser
    # Input arguments
    def parse_args(**kwargs):
        parser = ArgumentParser()
        parser.add_argument("-date", 
                help=("Date string, format yyyymmdd"),
                type=str,
                )
        parser.add_argument("-lat", 
                help=("Latitude (decimal deg.)"),
                type=float,
                default=56.549197, # Ekofisk
                )
        parser.add_argument("-lon", 
                help=("Longitude (decimal deg.)"),
                type=float,
                default=3.209986, # Ekofisk
                )
        parser.add_argument("--local", 
                help=("Return time in local time zone"),
                action='store_true',
                )
        return parser.parse_args(**kwargs)

    # Call args parser to create variables out of input arguments
    args = parse_args(args=sys.argv[1:])

    # If no date is given, use today's date
    if args.date is None:
        date = DT.now()
    else:
        # Convert date string to datetime
        date = DT.strptime(args.date, '%Y%m%d')

    # UTC or local time output
    if args.local:
        utc=False
    else:
        utc=True

    # Get sunrise + sunset times
    sunrise, sunset = sunrise_sunset(date=date, lat=args.lat, lon=args.lon, utc=utc)
    print('sunrise: {}, sunset: {}'.format(sunrise, sunset))

    # Make list of daylight hours (for stereo video scheduling)
    daylight_hours = [item for item in range(sunrise.hour+1, sunset.hour)]
