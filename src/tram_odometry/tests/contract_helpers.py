"""Small synthetic fixtures using the portable public map and Bridge APIs."""
import math

from tram_estimator import Drive
from tram_odometry.bridge import Bridge, BridgeConfig
from tram_position import CheckedRouteMap
from tram_position.route_localizer import Calibration, DirectedRoute, RouteLocalizer

BASE = 1_700_000_000_000_000_000
CLOCK_OFFSET = 900_000_000_000


class RecordingDrive:
    def __init__(self):
        self.calls = []

    def predict(self, command, speed):
        self.calls.append((command, speed))
        return Drive(command * .02, .1)


def checked_map(*, two_directions=False, length=1000.):
    calibration = Calibration((0., 0., 0.), (0., 0.), 0., (0., 0.), 0.)
    routes = [DirectedRoute("east", [(0., 0., 0.), (length, 0., 0.)])]
    if two_directions:
        routes.append(DirectedRoute("west", [(length, 0., 0.), (0., 0., 0.)]))
    return CheckedRouteMap(RouteLocalizer(calibration, tuple(routes)))


def make_bridge(config=None, *, two_directions=False, length=1000.):
    checked = checked_map(two_directions=two_directions, length=length)
    drive = RecordingDrive()
    config = config or BridgeConfig(wheel_input_unit="mps", start_mode="relative")
    return Bridge(drive, checked.as_indexed_localizer(), config), drive


def lla(calibration, x, y=0., z=0.):
    """Invert only our small fixture's calibration, without research helpers."""
    latitude, longitude = calibration.origin_lla[:2]
    altitude = z + calibration.gnss_alt_minus_map_z_m
    delta = 1e-5
    for _ in range(8):
        point = calibration.gnss_to_map(latitude, longitude, altitude)
        ex, ey = x-point[0], y-point[1]
        if math.hypot(ex, ey) < 1e-7:
            return latitude, longitude, altitude
        north = calibration.gnss_to_map(latitude+delta, longitude, altitude)
        east = calibration.gnss_to_map(latitude, longitude+delta, altitude)
        a, b = (north[0]-point[0])/delta, (east[0]-point[0])/delta
        c, d = (north[1]-point[1])/delta, (east[1]-point[1])/delta
        determinant = a*d-b*c
        latitude += (d*ex-b*ey)/determinant
        longitude += (-c*ex+a*ey)/determinant
    raise AssertionError("fixture coordinate inversion failed")


def stamp(seconds):
    return BASE + round(seconds*1e9)


def fix_at(bridge, seconds, x, y=0., z=0., *, source="master", receipt_seconds=None, **kwargs):
    header = stamp(seconds)
    clock = stamp(seconds if receipt_seconds is None else receipt_seconds) + CLOCK_OFFSET
    bridge.receive_fix(header, *lla(bridge.localizer.calibration, x, y, z),
                       source=source, clock_ns=clock, **kwargs)


def step(bridge, seconds, speed=0., command=0):
    header = stamp(seconds)
    clock = header+CLOCK_OFFSET
    for name in ("front", "rear"):
        bridge.receive_wheel(name, header, speed, clock)
    return bridge.receive_command(header, command, clock)
