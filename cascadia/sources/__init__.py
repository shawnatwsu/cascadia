"""Open-data source adapters.

Each adapter pulls one open feed and returns a normalized pandas DataFrame of
*observations* with at least: lat, lon, time (UTC), value, and a `kind` tag.
All MVP feeds are free; the no-key feeds run with zero signup.
"""
from .usgs_quakes import USGSQuakes
from .nws_alerts import NWSAlerts
from .open_meteo import OpenMeteo
from .usgs_water import USGSWater
from .firms import FIRMS

__all__ = ["USGSQuakes", "NWSAlerts", "OpenMeteo", "USGSWater", "FIRMS"]
