# simulate.py

from sgp4.api import Satrec, jday
import datetime
from fetch_tle import get_satellites


def get_positions(satellites):
    now = datetime.datetime.utcnow()

    results = {}

    for sat in satellites:
        s = Satrec.twoline2rv(sat["line1"], sat["line2"])

        jd, fr = jday(
            now.year, now.month, now.day,
            now.hour, now.minute, now.second
        )

        e, r, v = s.sgp4(jd, fr)

        if e == 0:
            results[sat["name"]] = r

    return results


def get_trajectory(sat, minutes=120, step=0.25):
    s = Satrec.twoline2rv(sat["line1"], sat["line2"])

    now = datetime.datetime.utcnow()
    trajectory = []

    steps = int(minutes / step)

    for i in range(steps):
        m = i * step
        t = now + datetime.timedelta(minutes=m)

        jd, fr = jday(
            t.year, t.month, t.day,
            t.hour, t.minute, t.second
        )

        e, r, v = s.sgp4(jd, fr)

        if e == 0:
            trajectory.append(r)

    return trajectory


# test run
if __name__ == "__main__":
    satellites = get_satellites(5)

    print("=== CURRENT POSITIONS ===")
    positions = get_positions(satellites)

    for name, pos in positions.items():
        print(name, pos)

    print("\n=== TRAJECTORY SAMPLE ===")
    traj = get_trajectory(satellites[0])

    print("Points:", len(traj))
    print("First:", traj[0])
    print("Last:", traj[-1])