#!/usr/bin/env python3
"""
Compute sunrise and sunset hours for given location.

Following example in
https://en.wikipedia.org/wiki/Sunrise_equation
"""
import sys
import numpy as np
from datetime import datetime as DT
from datetime import timedelta as TD
from datetime import timezone

def julian_day_number_to_gregorian(jdn):
    """
    Convert the Julian Day Number to the proleptic Gregorian 
    Year, Month, Day.
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
    """
    jdn = int(jd)
    year, month, day = julian_day_number_to_gregorian(jdn)
    offset = TD(days=(jd % 1), hours=+12)
    dt = DT(year=year, month=month, day=day, tzinfo=timezone.utc)
    return dt + offset

def sunrise_sunset(date, lat=56.549197, lon=3.209986):
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
    Returns:
        sunrise - datetime.datetime obj. (timezone UTC)
        sunset - datetime.datetime obj. (timezone UTC)
    """
    latr = np.deg2rad(lat) # Latitude in radians
    # Current Julian day
    jdate = date.toordinal() + 1721424.5
    # Calculate number of days since Jan 1st, 2000 12:00
    n = np.ceil(jdate - 2451545.0 + 0.0008)
    # Approximation of mean solar time at integer n expressed 
    # as a Julian day
    js = n - lon/360 # deg
    # Solar mean anomaly
    Md = (357.5291 + 0.98560028 * js) % 360 # Degrees
    Mr = np.deg2rad(Md) # Radians
    # Equation of the center
    C = 1.9148*np.sin(Mr) + 0.02*np.sin(2*Mr) + 0.0003*np.sin(3*Mr)
    # Ecliptic longitude
    ld = (Md + C + 180 + 102.9372) % 360 # deg
    lr = np.deg2rad(ld) # rad
    # Solar transit (deg)
    jt = (2451545 + js + 0.0053*np.sin(Mr) - 0.0069*np.sin(2*lr))
    # Declination of the Sun (rad)
    delta = np.arcsin(np.sin(lr) * np.sin(np.deg2rad(23.4397)))
    # Hour angle omega_0 (deg)
    nom = np.sin(np.deg2rad(-0.833)) - np.sin(latr) * np.sin(delta)
    denom = np.cos(latr) * np.cos(delta)
    omega_0 = np.rad2deg(np.arccos(nom / denom))
    # Calculate sunrise and sunset
    jrise = jt - omega_0/360 # Julian date
    jset = jt + omega_0/360 # Julian date
    # Convert to UTC 
    sunrise = (julian_date_to_gregorian(jrise))
    sunset = (julian_date_to_gregorian(jset))

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
        return parser.parse_args(**kwargs)

    # Call args parser to create variables out of input arguments
    args = parse_args(args=sys.argv[1:])

    # If no date is given, use today's date
    if args.date is None:
        date = DT.now()
    else:
        # Convert date string to datetime
        date = DT.strptime(args.date, '%Y%m%d')

    # Get sunrise + sunset times
    sunrise, sunset = sunrise_sunset(date, args.lat, args.lon)
    print('sunrise: {}, sunset: {}'.format(sunrise, sunset))

    # Make list of daylight hours (for stereo video scheduling)
    daylight_hours = np.arange(sunrise.hour+1, sunset.hour)
