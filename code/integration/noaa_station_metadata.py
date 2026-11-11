"""Join the challenge stations with NOAA station metadata and check the detected level shifts against it.

Run from the repository root, after the notebook has written data/level_shifts_at_2018.csv:
    python code/integration/noaa_station_metadata.py

Sources (station metadata only, no weather observations; only events dated <= 2018-12-31 are kept):
    NOAA GHCN-Daily station list   https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt
    NOAA HOMR station histories    https://www.ncei.noaa.gov/access/homr/services/station/...
This script does not feed the forecasts; it audits the data and the station-change adjustment.

Writes:
    data/ghcnd_station_crosswalk.csv      challenge station vs. GHCN-Daily entry (name, location, elevation, networks)
    data/noaa_homr_events_to_2018.csv     dated relocations and temperature-equipment changes per station
    data/station_break_attribution.csv    each detected level shift, matched to city_info and HOMR events
"""
import io
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
REPO, REVISION = "CMU-OSPO/protostar-climate-train-to-2018", "bbbdebdab2a18e00f97ef87bd3f35a71e222180b"
GHCND_STATIONS = "https://www.ncei.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt"
HOMR = "https://www.ncei.noaa.gov/access/homr/services/station"
CUTOFF = pd.Timestamp("2018-12-31")
WINDOW = 1  # years: an event this close to a detected break counts as a possible cause


def get(url, tries=5, as_json=False):
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                body = r.read()
            return json.loads(body) if as_json else body
        except Exception:  # network errors and truncated JSON are retried
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def load_challenge():
    from datasets import load_dataset
    annual = load_dataset(REPO, "annual_indices", split="train", revision=REVISION).to_pandas()
    info = load_dataset(REPO, "city_info", split="train", revision=REVISION).to_pandas()
    cities = annual.sort_values("Year").groupby("ID").tail(1).set_index("ID")[["Name", "Lat", "Lon"]]
    return cities, info


def ghcnd_crosswalk(cities):
    txt = get(GHCND_STATIONS).decode()
    rows = [{"ID": l[0:11], "ghcnd_lat": float(l[12:20]), "ghcnd_lon": float(l[21:30]), "elevation_m": float(l[31:37]),
             "ghcnd_state": l[38:40].strip(), "ghcnd_name": l[41:71].strip(), "gsn": l[72:75].strip(),
             "hcn_crn": l[76:79].strip(), "wmo_id": l[80:85].strip()} for l in txt.splitlines() if len(l) >= 71]
    g = pd.DataFrame(rows).set_index("ID")
    x = cities.join(g, how="left")
    la1, lo1, la2, lo2 = map(np.radians, [x.Lat, x.Lon, x.ghcnd_lat, x.ghcnd_lon])
    x["offset_km"] = 6371 * 2 * np.arcsin(np.sqrt(np.sin((la2 - la1) / 2) ** 2 +
                                                   np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2))
    return x.reset_index()


def homr_events(ghcnd_id):
    """Dated relocations and temperature-equipment changes for one station, up to the cutoff."""
    head = get(f"{HOMR}/search?qid=GHCND:{ghcnd_id}&headersOnly=true", as_json=True)
    stations = head["stationCollection"].get("stations") or []
    if not stations:
        return ghcnd_id, None, []
    stn = stations[0]["ncdcStnId"]
    full = get(f"{HOMR}/{stn}?date=all", as_json=True)["stationCollection"]["stations"][0]
    ev = []
    for r in full.get("relocations") or []:
        ev.append({"date": r.get("date", "")[:10], "event": "relocation", "detail": r.get("relocation", "")})
    temp = sorted(((e.get("date") or {}).get("beginDate", "")[:10], (e.get("equipment") or {}).get("equipment", "UNKNOWN"))
                  for e in full.get("elements") or [] if e.get("element") == "TEMP")
    prev = None
    for begin, eq in temp:
        if prev is not None and eq != prev:
            ev.append({"date": begin, "event": "temp_equipment", "detail": f"{prev} -> {eq}"})
        prev = eq
    ev = [dict(e, ID=ghcnd_id, ncdc_stn_id=stn) for e in ev if e["date"] and pd.Timestamp(e["date"]) <= CUTOFF]
    return ghcnd_id, stn, ev


def main():
    cities, info = load_challenge()
    xw = ghcnd_crosswalk(cities)
    xw.round(4).to_csv(os.path.join(DATA, "ghcnd_station_crosswalk.csv"), index=False)
    print(f"GHCN-Daily: {xw.ghcnd_name.notna().sum()} of {len(xw)} challenge IDs found; "
          f"median location offset {xw.offset_km.median():.2f} km, max {xw.offset_km.max():.1f} km "
          f"({xw.loc[xw.offset_km.idxmax(), 'Name']}); HCN/CRN stations: {(xw.hcn_crn != '').sum()}")

    with ThreadPoolExecutor(3) as pool:
        res = list(pool.map(homr_events, cities.index))
    found = sum(stn is not None for _, stn, _ in res)
    ev = pd.DataFrame([e for _, _, es in res for e in es], columns=["ID", "ncdc_stn_id", "date", "event", "detail"])
    ev = ev.sort_values(["ID", "date"])
    ev.to_csv(os.path.join(DATA, "noaa_homr_events_to_2018.csv"), index=False)
    print(f"HOMR: {found} of {len(cities)} stations found; {len(ev)} events up to 2018 "
          f"({(ev.event == 'relocation').sum()} relocations, {(ev.event == 'temp_equipment').sum()} temperature-equipment changes)")

    br = pd.read_csv(os.path.join(DATA, "level_shifts_at_2018.csv"))
    info = info.copy()
    info["start"] = pd.to_datetime(info["Stn.stDate"].astype(str), errors="coerce")
    tr = info.sort_values(["ID", "start"])
    tr = tr[tr.groupby("ID").cumcount() > 0]
    ev["year"] = pd.to_datetime(ev.date).dt.year

    def near(df, ident, year, col):
        m = df[(df.ID == ident) & ((df[col] - year).abs() <= WINDOW)]
        return m

    rows = []
    for b in br.itertuples():
        ci = near(tr.assign(year=tr.start.dt.year), b.ID, b.break_year, "year")
        hm = near(ev, b.ID, b.break_year, "year")
        rows.append({"target": b.target, "ID": b.ID, "Name": cities.loc[b.ID, "Name"], "break_year": b.break_year,
                     "shift": b.shift, "t": b.t,
                     "city_info_change": "; ".join(ci["Stn.Name"].astype(str)),
                     "homr_relocation": "; ".join(f"{r.date} {r.detail}" for r in hm[hm.event == "relocation"].itertuples()),
                     "homr_temp_equipment": "; ".join(f"{r.date} {r.detail}" for r in hm[hm.event == "temp_equipment"].itertuples())})
    att = pd.DataFrame(rows)
    att["explained_by"] = np.select(
        [att.city_info_change != "", (att.homr_relocation != "") | (att.homr_temp_equipment != "")],
        ["city_info", "noaa_homr_only"], "none")
    att.to_csv(os.path.join(DATA, "station_break_attribution.csv"), index=False)

    print(f"\nDetected level shifts matched to a documented change within ±{WINDOW} year:")
    print(att.groupby("target").explained_by.value_counts().unstack(fill_value=0).to_string())
    print("\nLargest Avg Temp shifts:")
    print(att[att.target == "Avg Temp"].reindex(att[att.target == "Avg Temp"]["shift"].abs().sort_values(ascending=False).index)
          .head(10)[["Name", "break_year", "shift", "city_info_change", "homr_relocation", "homr_temp_equipment"]].to_string(index=False))


if __name__ == "__main__":
    main()
