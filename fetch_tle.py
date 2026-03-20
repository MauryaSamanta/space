# fetch_tle.py

import requests

def get_satellites(limit=10):
    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
    
    response = requests.get(url)
    data = response.text

    lines = data.strip().split("\n")

    satellites = []

    for i in range(0, len(lines), 3):
        if i + 2 < len(lines):
            satellites.append({
                "name": lines[i].strip(),
                "line1": lines[i+1].strip(),
                "line2": lines[i+2].strip()
            })

    return satellites[:limit]


# test run
if __name__ == "__main__":
    sats = get_satellites(5)
    print("Satellites loaded:", len(sats))
    print(sats[0])