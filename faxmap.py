
import argparse
import datetime
import math
import random
from io import StringIO
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import metpy.calc as mpcalc
from metpy.calc import find_peaks
from metpy.plots import StationPlot, scattertext
from metpy.plots.wx_symbols import sky_cover
from metpy.units import units
import numpy as np
import pandas as pd
import pygrib
import requests
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
from scipy.spatial import cKDTree
from skimage.restoration import denoise_bilateral
import xarray as xr


# Wyoming大学への観測データ取得は、並列処理中でもリクエスト開始を一定間隔にする。
_wyoming_request_lock = threading.Lock()
_wyoming_last_request = 0.0
WYOMING_MIN_REQUEST_INTERVAL = 30


def wait_request_interval(min_interval):
    """スレッド間で共有するリクエスト開始間隔制御。"""
    global _wyoming_last_request
    if min_interval <= 0:
        return
    with _wyoming_request_lock:
        now = time.monotonic()
        wait = min_interval - (now - _wyoming_last_request)
        if wait > 0:
            time.sleep(wait)
        _wyoming_last_request = time.monotonic()


def get_with_retry(url, timeout=60, max_wait=300, description="HTTP request", min_interval=0, max_retries=8):
    """通信障害・HTTP 429・サーバー側5xxのみ有限回再試行する。恒久的な4xxは再試行しない。"""
    retry_statuses = {429, 500, 502, 503, 504}

    for attempt in range(max_retries + 1):
        try:
            wait_request_interval(min_interval)
            response = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 meteorological-data-client"})

            if response.status_code in (200, 404):
                return response

            if response.status_code not in retry_statuses:
                body = response.text[:1500].replace("\n", " ")
                print(f"{description}: HTTP {response.status_code}; 再試行しません。応答: {body}")
                return response

            if attempt >= max_retries:
                print(f"{description}: HTTP {response.status_code}; 最大再試行回数に到達")
                return response

            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    wait = min(float(retry_after), max_wait)
                except ValueError:
                    wait = min(15 * (2 ** min(attempt, 5)), max_wait)
            else:
                wait = min(15 * (2 ** min(attempt, 5)), max_wait)

            wait += random.uniform(0, min(5, wait * 0.1))
            print(f"{description}: HTTP {response.status_code}; {wait:.1f}秒後に再試行 ({attempt + 1}回目)")
            response.close()
            time.sleep(wait)

        except requests.exceptions.RequestException as e:
            if attempt >= max_retries:
                print(f"{description}: 通信エラー、最大再試行回数に到達: {e}")
                raise

            wait = min(5 * (2 ** min(attempt, 6)), max_wait)
            print(f"{description}: 通信エラー ({e}); {wait}秒後に再試行 ({attempt + 1}回目)")
            time.sleep(wait)

    raise RuntimeError(f"{description}: 通信処理が終了しました")


def transform_lonlat_to_figure(lonlat, ax, proj):
    point_proj = proj.transform_point(*lonlat, ccrs.PlateCarree())
    point_pix = ax.transData.transform(point_proj)
    point_fig = ax.transAxes.inverted().transform(point_pix)
    return point_fig, point_pix, point_proj


## 極大/極小ピーク検出関数
def detect_peaks(image, filter_size, dist_cut, flag=0):
    if flag == 0:
        local_extreme = maximum_filter(image, footprint=np.ones((filter_size, filter_size)), mode='constant')
    else:
        local_extreme = minimum_filter(image, footprint=np.ones((filter_size, filter_size)), mode='constant')

    peaks_mask = image == local_extreme
    x, y = np.where(peaks_mask)

    if len(x) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    coords = np.column_stack((x, y))
    tree = cKDTree(coords)
    neighbors = tree.query_ball_point(coords, r=dist_cut)
    dSum = np.zeros(len(x), dtype=float)
    neighbor_distances = []

    for i, indices in enumerate(neighbors):
        tmp = []
        for k in indices:
            if i == k:
                continue
            d = np.linalg.norm(coords[i] - coords[k])
            if d < dist_cut:
                tmp.append((k, d))
                dSum[i] += d
        neighbor_distances.append(tmp)

    cutPoint = np.zeros(len(x), dtype=bool)

    for i in range(len(x)):
        val_i = image[x[i], y[i]]
        for k, d in neighbor_distances[i]:
            val_k = image[x[k], y[k]]
            if flag == 0 and val_i < val_k:
                cutPoint[i] = True
                break
            if flag != 0 and val_i > val_k:
                cutPoint[i] = True
                break
            if dSum[i] > dSum[k]:
                cutPoint[i] = True
                break
            if dSum[i] == dSum[k] and i > k:
                cutPoint[i] = True
                break

    return x[~cutPoint], y[~cutPoint]


def _trace_candidate_axis(candidate_columns, score_columns, ncols, max_step=5, max_gap=3, distance_penalty=0.12):
    """最も強い候補を起点に、左右へ連続する候補を追跡する。"""
    seed_col = None
    seed_idx = None
    seed_score = -np.inf

    for j in range(ncols):
        if len(candidate_columns[j]) == 0:
            continue
        k = int(np.argmax(score_columns[j]))
        if score_columns[j][k] > seed_score:
            seed_score = float(score_columns[j][k])
            seed_col = j
            seed_idx = int(candidate_columns[j][k])

    axis = np.full(ncols, np.nan)
    if seed_col is None:
        return axis

    axis[seed_col] = seed_idx

    for direction in (-1, 1):
        previous_idx = seed_idx
        gap = 0
        j = seed_col + direction

        while 0 <= j < ncols:
            candidates = candidate_columns[j]
            scores = score_columns[j]

            if len(candidates) == 0:
                gap += 1
                if gap > max_gap:
                    break
                j += direction
                continue

            distances = np.abs(candidates - previous_idx)
            valid = distances <= max_step * (gap + 1)

            if not np.any(valid):
                gap += 1
                if gap > max_gap:
                    break
                j += direction
                continue

            candidate_ids = np.where(valid)[0]
            rank = scores[candidate_ids] - distance_penalty * distances[candidate_ids]
            chosen = candidate_ids[int(np.argmax(rank))]
            chosen_idx = int(candidates[chosen])

            axis[j] = chosen_idx
            previous_idx = chosen_idx
            gap = 0
            j += direction

    return axis


# 300hPa強風軸の抽出
def detect_jet_axis(wind_speed, lats, lons, threshold=50.0):
    """50kt以上の局地的な風速極大を抽出し、経度方向に連続追跡する。"""
    speed = np.asarray(wind_speed, dtype=float)
    finite = np.isfinite(speed)
    weighted = ndimage.gaussian_filter(np.where(finite, speed, 0.0), sigma=1.0)
    weight = ndimage.gaussian_filter(finite.astype(float), sigma=1.0)
    smooth = np.divide(weighted, weight, out=np.full_like(weighted, np.nan), where=weight > 0.5)
    local_max = maximum_filter(np.where(np.isfinite(smooth), smooth, -np.inf), size=(5, 5), mode='nearest')
    candidate_mask = np.isfinite(smooth) & (smooth >= threshold) & (smooth == local_max)

    candidate_columns, score_columns = [], []
    for j in range(speed.shape[1]):
        rows = np.where(candidate_mask[:, j])[0]
        candidate_columns.append(rows)
        score_columns.append(smooth[rows, j] if len(rows) else np.array([], dtype=float))

    axis_index = _trace_candidate_axis(candidate_columns, score_columns, speed.shape[1], max_step=6, max_gap=2, distance_penalty=0.20)
    axis_lats = np.full(len(lons), np.nan)
    axis_speeds = np.full(len(lons), np.nan)
    valid = np.isfinite(axis_index)
    if np.any(valid):
        rows = axis_index[valid].astype(int)
        cols = np.where(valid)[0]
        axis_lats[valid] = np.asarray(lats)[rows]
        axis_speeds[valid] = smooth[rows, cols]

    valid_cols = np.where(valid)[0]
    if len(valid_cols) >= 3:
        segments = np.split(valid_cols, np.where(np.diff(valid_cols) > 1)[0] + 1)
        for segment in segments:
            if len(segment) >= 3:
                axis_lats[segment] = ndimage.gaussian_filter1d(axis_lats[segment], sigma=1.0)
    return np.asarray(lons), axis_lats, axis_speeds, valid


# 500hPaトラフ・リッジ軸の抽出
def detect_trough_ridge(height, lats, lons):
    """各緯度の東西断面で高度の極小(トラフ)・極大(リッジ)を検出し、南北に追跡する。

    トラフ・リッジは東西方向の高度分布の谷・山として抽出する。
    欠測を含む断面は補間せず、その断面の候補を作らない。
    """
    z = np.asarray(height, dtype=float)
    # 緯度・経度の両方向を軽く平滑化し、格子ノイズによる過剰検出を抑える。
    finite = np.isfinite(z)
    weighted = ndimage.gaussian_filter(np.where(finite, z, 0.0), sigma=(1.0, 1.5))
    weight = ndimage.gaussian_filter(finite.astype(float), sigma=(1.0, 1.5))
    smooth = np.divide(weighted, weight, out=np.full_like(weighted, np.nan), where=weight > 0.8)

    def make_candidates(is_trough):
        candidates_by_lat, scores_by_lat = [], []
        for i in range(smooth.shape[0]):
            row = smooth[i, :]
            if not np.all(np.isfinite(row)):
                candidates_by_lat.append(np.array([], dtype=int))
                scores_by_lat.append(np.array([], dtype=float))
                continue
            if is_trough:
                idx = np.where((row[1:-1] < row[:-2]) & (row[1:-1] < row[2:]))[0] + 1
                strength = row[idx - 1] + row[idx + 1] - 2.0 * row[idx]
            else:
                idx = np.where((row[1:-1] > row[:-2]) & (row[1:-1] > row[2:]))[0] + 1
                strength = 2.0 * row[idx] - row[idx - 1] - row[idx + 1]
            # 断面ごとの相対基準で弱い曲率極値を除外する。
            if len(strength):
                limit = max(1.0, float(np.nanmax(strength)) * 0.05)
                keep = strength >= limit
                idx, strength = idx[keep], strength[keep]
            candidates_by_lat.append(idx.astype(int))
            scores_by_lat.append(np.asarray(strength, dtype=float))
        return candidates_by_lat, scores_by_lat

    trough_candidates, trough_scores = make_candidates(True)
    ridge_candidates, ridge_scores = make_candidates(False)

    # 追跡軸の横方向は経度、追跡方向は緯度。
    trough_axis = _trace_candidate_axis(trough_candidates, trough_scores, len(lats), max_step=8, max_gap=3, distance_penalty=0.25)
    ridge_axis = _trace_candidate_axis(ridge_candidates, ridge_scores, len(lats), max_step=8, max_gap=3, distance_penalty=0.25)

    trough_lons = np.full(len(lats), np.nan)
    ridge_lons = np.full(len(lats), np.nan)
    trough_valid = np.isfinite(trough_axis)
    ridge_valid = np.isfinite(ridge_axis)
    trough_lons[trough_valid] = np.asarray(lons)[trough_axis[trough_valid].astype(int)]
    ridge_lons[ridge_valid] = np.asarray(lons)[ridge_axis[ridge_valid].astype(int)]

    for values, valid in ((trough_lons, trough_valid), (ridge_lons, ridge_valid)):
        rows = np.where(valid)[0]
        if len(rows) >= 3:
            segments = np.split(rows, np.where(np.diff(rows) > 1)[0] + 1)
            for segment in segments:
                if len(segment) >= 3:
                    values[segment] = ndimage.gaussian_filter1d(values[segment], sigma=1.0)

    latitudes = np.asarray(lats)
    return trough_lons, latitudes, ridge_lons, latitudes


# 高層気象観測データ取得
def fetch_wyoming_data_all(wmo, dt):
    url = f"https://weather.uwyo.edu/wsgi/sounding?datetime={dt.year}-{dt.month:02d}-{dt.day:02d}%20{dt.hour:02d}:00:00&id={wmo}&type=TEXT:LIST"

    try:
        res = get_with_retry(url, timeout=60, description=f"Wyoming {wmo}", min_interval=WYOMING_MIN_REQUEST_INTERVAL, max_retries=3)

        if res.status_code == 404:
            print(f"Wyoming {wmo}: HTTP 404（対象時刻の観測データなし）")
            return "no_data", None

        if res.status_code == 400:
            return "request_error", f"HTTP 400: {res.text[:1000]}"

        if res.status_code != 200:
            return "http_error", f"HTTP {res.status_code}"

        lines = res.text.splitlines()
        data_started = False
        data = {}

        for line in lines:
            if "PRES" in line and "TEMP" in line:
                data_started = True
                continue

            if data_started:
                cols = line.split()
                if len(cols) < 8:
                    continue

                try:
                    pres = int(float(cols[0]))
                    temp = float(cols[2])
                    ttd = float(cols[2]) - float(cols[3])
                    u_wind = -math.sin(math.radians(float(cols[6]))) * float(cols[7])
                    v_wind = -math.cos(math.radians(float(cols[6]))) * float(cols[7])
                    data[pres] = (temp, ttd, u_wind, v_wind)
                except (ValueError, IndexError):
                    continue

        if not data_started or not data:
            debug_name = f"wyoming_parse_error_{wmo}_{dt:%Y%m%d%H}.txt"
            with open(debug_name, "w", encoding="utf-8") as f:
                f.write(res.text)
            return "parse_error", f"HTTP 200応答を解析できません。応答を {debug_name} に保存しました"

        return "success", data

    except requests.exceptions.RequestException as e:
        return "communication_error", str(e)


# 現在のUTC時刻を取得
now_utc = datetime.datetime.now(datetime.UTC)
adjusted_time = now_utc - datetime.timedelta(hours=12) - datetime.timedelta(minutes=80)
truncated_hour = (adjusted_time.hour // 12) * 12
dt = adjusted_time.replace(hour=truncated_hour, minute=0, second=0, microsecond=0)
vt = dt + datetime.timedelta(hours=12)

i_year = dt.year
i_month = dt.month
i_day = dt.day
i_hourZ = dt.hour

print("IT(UTC):", dt)

fig_fld = "{0:4d}{1:02d}{2:02d}{3:02d}".format(i_year, i_month, i_day, i_hourZ)
if not os.path.exists(fig_fld):
    os.makedirs(fig_fld)

i_area = [115, 155, 20, 50]
str_area = "jp"
latS = 10
latN = 60
lonW = 105
lonE = 165

# ECMWF 0.1度格子データ（oper-fc-pl / oper-fc-sfc）のダウンロード処理
data_fld = "https://storage.googleapis.com/ecmwf-open-data/{0:04d}{1:02d}{2:02d}/{3:02d}z/ifs/0p1/oper/"
dat_fld = data_fld.format(i_year, i_month, i_day, i_hourZ)

fname_pl = "{0:4d}{1:02d}{2:02d}{3:02d}0000-12h-oper-fc-pl.grib2".format(i_year, i_month, i_day, i_hourZ)
fname_sfc = "{0:4d}{1:02d}{2:02d}{3:02d}0000-12h-oper-fc-sfc.grib2".format(i_year, i_month, i_day, i_hourZ)

for fname in [fname_pl, fname_sfc]:
    if not os.path.exists(fname):
        response = get_with_retry(dat_fld + fname, timeout=180, description=f"ECMWF {fname}")
        response.raise_for_status()
        with open(fname, 'wb') as f:
            f.write(response.content)


# GRIB2データの読み込み
grbs_pl = pygrib.open(fname_pl)
grbs_sfc = pygrib.open(fname_sfc)

grbTm = grbs_pl(shortName="t", typeOfLevel='isobaricInhPa', level=lambda l: l >= 300)
valHt, latHt, lonHt = grbTm[0].data(lat1=latS, lat2=latN, lon1=lonW, lon2=lonE)

levels = np.sort(np.array([g['level'] for g in grbTm]))
l_size = levels.size
(lat_size, lon_size) = valHt.shape
lats = latHt[:, 0]
lons = lonHt[0, :]

aryHt = np.zeros([l_size, lat_size, lon_size])
aryTm = np.zeros([l_size, lat_size, lon_size])
aryRh = np.zeros([l_size, lat_size, lon_size])
aryWu = np.zeros([l_size, lat_size, lon_size])
aryWv = np.zeros([l_size, lat_size, lon_size])
aryOmg = np.zeros([l_size, lat_size, lon_size])

grbHt = sorted(grbs_pl.select(shortName="gh", typeOfLevel='isobaricInhPa', level=lambda l: l >= 300), key=lambda g: g.level, reverse=False)
grbTm = sorted(grbs_pl.select(shortName="t", typeOfLevel='isobaricInhPa', level=lambda l: l >= 300), key=lambda g: g.level, reverse=False)
grbWu = sorted(grbs_pl.select(shortName="u", typeOfLevel='isobaricInhPa', level=lambda l: l >= 300), key=lambda g: g.level, reverse=False)
grbWv = sorted(grbs_pl.select(shortName="v", typeOfLevel='isobaricInhPa', level=lambda l: l >= 300), key=lambda g: g.level, reverse=False)
grbRh = sorted(grbs_pl.select(shortName="r", typeOfLevel='isobaricInhPa', level=lambda l: l >= 300), key=lambda g: g.level, reverse=False)
grbOmg = sorted(grbs_pl.select(shortName="w", typeOfLevel='isobaricInhPa', level=lambda l: l >= 300), key=lambda g: g.level, reverse=False)

valSlp, _, _ = grbs_sfc.select(name='Mean sea level pressure')[0].data(lat1=latS, lat2=latN, lon1=lonW, lon2=lonE)
valHgt, _, _ = grbs_sfc.select(name='Surface pressure')[0].data(lat1=latS, lat2=latN, lon1=lonW, lon2=lonE)

for l in range(l_size):
    valHt, _, _ = grbHt[l].data(lat1=latS, lat2=latN, lon1=lonW, lon2=lonE)
    valTm, _, _ = grbTm[l].data(lat1=latS, lat2=latN, lon1=lonW, lon2=lonE)
    valWu, _, _ = grbWu[l].data(lat1=latS, lat2=latN, lon1=lonW, lon2=lonE)
    valWv, _, _ = grbWv[l].data(lat1=latS, lat2=latN, lon1=lonW, lon2=lonE)
    valRh, _, _ = grbRh[l].data(lat1=latS, lat2=latN, lon1=lonW, lon2=lonE)
    valOmg, _, _ = grbOmg[l].data(lat1=latS, lat2=latN, lon1=lonW, lon2=lonE)
    aryHt[l] = valHt
    aryTm[l] = valTm
    aryWu[l] = valWu
    aryWv[l] = valWv
    aryRh[l] = valRh
    aryOmg[l] = valOmg

arySlp = valSlp
aryHgt = valHgt

ds = xr.Dataset(
    {
        "Geopotential_height": (["level", "lat", "lon"], aryHt * units.meter),
        "temperature": (["level", "lat", "lon"], aryTm * units('K')),
        "relative_humidity": (["level", "lat", "lon"], aryRh * units('%')),
        "u_wind": (["level", "lat", "lon"], aryWu * units('m/s')),
        "v_wind": (["level", "lat", "lon"], aryWv * units('m/s')),
        "omega": (["level", "lat", "lon"], aryOmg / 100 * 3600 * units('Pa/s')),
        "mslp": (["lat", "lon"], arySlp / 100 * units('hPa')),
        "height": (["lat", "lon"], aryHgt / 100 * units('hPa'))
    },
    coords={"level": levels, "lat": lats, "lon": lons}
)

ds['Geopotential_height'].attrs['units'] = 'm'
ds['u_wind'].attrs['units'] = 'm/s'
ds['v_wind'].attrs['units'] = 'm/s'
ds['omega'].attrs['units'] = 'hPa/h'
ds['level'].attrs['units'] = 'hPa'
ds['lat'].attrs['units'] = 'degrees_north'
ds['lon'].attrs['units'] = 'degrees_east'

dsp = ds.metpy.parse_cf()
dsp['wind_speed'] = mpcalc.wind_speed(dsp['u_wind'], dsp['v_wind'])
dsp['conv'] = mpcalc.divergence(dsp['u_wind'], dsp['v_wind'])
dsp['u_wind'] = dsp['u_wind'].metpy.convert_units('knots')
dsp['v_wind'] = dsp['v_wind'].metpy.convert_units('knots')
dsp['wind_speed'] = dsp['wind_speed'].metpy.convert_units('knots')
dsp['vorticity'] = mpcalc.vorticity(dsp['u_wind'], dsp['v_wind'])
dsp['temperature'] = dsp['temperature'].metpy.convert_units(units.degC)

# 相対湿度の異常値対策：露点温度計算に使う値だけ1～100%に制限
rh_for_dewpoint = dsp['relative_humidity'].copy()
rh_values = np.asarray(rh_for_dewpoint.data.magnitude, dtype=float)
rh_values = np.where(np.isfinite(rh_values), np.clip(rh_values, 1.0, 100.0), np.nan)
rh_for_dewpoint.data = rh_values * units.percent

dsp['dewpoint_temperature'] = mpcalc.dewpoint_from_relative_humidity(dsp['temperature'], rh_for_dewpoint)
dsp['ttd'] = dsp['temperature'] - dsp['dewpoint_temperature']
dsp['Equivalent_Potential_temperature'] = mpcalc.equivalent_potential_temperature(dsp['level'], dsp['temperature'], dsp['dewpoint_temperature'])

p_925 = 925 * units.hPa
p_700 = 700 * units.hPa
p_500 = 500 * units.hPa

idx_925 = np.where(levels == 925)[0][0]
idx_700 = np.where(levels == 700)[0][0]
idx_500 = np.where(levels == 500)[0][0]

theta_e_925 = mpcalc.equivalent_potential_temperature(p_925, dsp['temperature'][idx_925], dsp['dewpoint_temperature'][idx_925])
theta_e_700 = mpcalc.equivalent_potential_temperature(p_700, dsp['temperature'][idx_700], dsp['dewpoint_temperature'][idx_700])
theta_es_700 = mpcalc.equivalent_potential_temperature(p_700, dsp['temperature'][idx_700], dsp['temperature'][idx_700])
theta_es_500 = mpcalc.equivalent_potential_temperature(p_500, dsp['temperature'][idx_500], dsp['temperature'][idx_500])

dsp['NewIndex'] = theta_e_925 - theta_es_500 - (theta_es_700 - theta_e_700)

dsp['Geopotential_height'].data = ndimage.gaussian_filter(dsp['Geopotential_height'].data, sigma=(0, 1, 1))
dsp['temperature'].data = ndimage.gaussian_filter(dsp['temperature'].data, sigma=(0, 4, 4))
dsp['Equivalent_Potential_temperature'].data = ndimage.gaussian_filter(dsp['Equivalent_Potential_temperature'].data, sigma=(0, 1, 1))
dsp['mslp'].data = ndimage.gaussian_filter(dsp['mslp'].data.magnitude, sigma=3) * dsp['mslp'].data.units


url = "https://www.ncei.noaa.gov/pub/data/igra/igra2-station-list.txt"
res = get_with_retry(url, timeout=120, description="NOAA IGRA station list")
res.raise_for_status()

colspecs = [(0, 11), (12, 20), (21, 30), (31, 36), (36, 72), (77, 81)]
names = ['station_id', 'lat', 'lon', 'elev', 'name', 'year']
df = pd.read_fwf(StringIO(res.text), colspecs=colspecs, names=names)

df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
df['elev'] = pd.to_numeric(df['elev'], errors='coerce')
df['wmo'] = pd.to_numeric(df['station_id'].astype(str).str[-5:], errors='coerce')
df['year'] = pd.to_numeric(df['year'], errors='coerce')

df_sel = df[(df['lon'] >= i_area[0]) & (df['lon'] <= i_area[1]) & (df['lat'] >= i_area[2]) & (df['lat'] <= i_area[3]) & (df['wmo'].notnull()) & (df['year'] > 2025)]

dt_str = (vt.strftime("%Y%m%d%HUTC")).upper()
dt_str2 = vt.strftime("%Y%m%d%H")

dlon, dlat = 10, 10
proj = ccrs.PlateCarree()


# --- 850hPa EPT 図 ---
fig3 = plt.figure(figsize=(10, 8))
plt.subplots_adjust(left=0, right=1, bottom=0.06, top=0.98)
ax = fig3.add_subplot(1, 1, 1, projection=proj)
ax.set_extent(i_area, proj)

cnf_ept = ax.contourf(dsp['lon'], dsp['lat'], dsp['Equivalent_Potential_temperature'][np.where(levels == 850)[0][0], :, :], np.arange(255, 375, 3), cmap="jet", extend='both', transform=proj)
cn_ept0 = ax.contour(dsp['lon'], dsp['lat'], dsp['Equivalent_Potential_temperature'][np.where(levels == 850)[0][0], :, :], colors='black', linewidths=0.3, levels=np.arange(255, 375, 3), transform=proj)
cn_ept1 = ax.contour(dsp['lon'], dsp['lat'], dsp['Equivalent_Potential_temperature'][np.where(levels == 850)[0][0], :, :], colors='black', linewidths=1.0, levels=np.arange(255, 375, 15), transform=proj)

ax.clabel(cn_ept1, fontsize=12, inline=True, inline_spacing=5, fmt='%i', rightside_up=True, colors='black')
ax.coastlines(resolution='50m', linewidth=1.6)

xticks = np.arange(0, 360, dlon)
yticks = np.arange(-90, 90.1, dlat)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, alpha=0.8)
gl.xlocator = mticker.FixedLocator(xticks)
gl.ylocator = mticker.FixedLocator(yticks)

wind_slice = (slice(None, None, 10), slice(None, None, 10))
ax.barbs(dsp['lon'][wind_slice[1]], dsp['lat'][wind_slice[0]], dsp['u_wind'][np.where(levels == 850)[0][0], wind_slice[0], wind_slice[1]].values, dsp['v_wind'][np.where(levels == 850)[0][0], wind_slice[0], wind_slice[1]].values, length=5.5, pivot='middle', color='black', transform=proj)

fig3.text(0.5, 0.01, dt_str + " 850hPa EPT(K), Wind", ha='center', va='bottom', size=20)
output_dir = os.path.join("Data/", dt_str)
os.makedirs(output_dir, exist_ok=True)
output_fig_nm = "{0}_850ept.png".format(dt_str)
out_path = os.path.join(output_dir, output_fig_nm)
plt.savefig(out_path)
print("output:{}".format(output_fig_nm))
plt.show()


# --- 地上天気図 ---
fig3 = plt.figure(figsize=(10, 8))
plt.subplots_adjust(left=0, right=1, bottom=0.06, top=0.98)
ax = fig3.add_subplot(1, 1, 1, projection=proj)
ax.set_extent(i_area, proj)

clevs_mslp = np.arange(800, 1200, 4)
ax.contour(dsp['lon'], dsp['lat'], dsp['mslp'], clevs_mslp, colors='black', linestyles='solid', linewidths=[1.25, 0.75, 0.75, 0.75, 0.75], transform=proj)
ax.contourf(dsp['lon'], dsp['lat'], dsp['NewIndex'].values, [-3, 0], colors=['yellow', 'red'], extend='max', transform=proj, alpha=0.5)

# 地上気圧の極大
maxid = detect_peaks(dsp['mslp'].values, filter_size=50, dist_cut=25.0)
for j in range(len(maxid[0])):
    wlon = dsp['lon'][maxid[1][j]]
    wlat = dsp['lat'][maxid[0][j]]
    fig_z, _, _ = transform_lonlat_to_figure((wlon, wlat), ax, proj)
    if fig_z[0] > 0.05 and fig_z[0] < 0.95 and fig_z[1] > 0.05 and fig_z[1] < 0.95:
        ax.plot(wlon, wlat, marker='x', markersize=4, color="blue", transform=proj)
        ax.text(wlon, wlat + 0.5, 'H', size=24, color="blue", transform=proj)
        val = dsp['mslp'].values[maxid[0][j]][maxid[1][j]]
        ival = int(val)
        ax.text(fig_z[0], fig_z[1] - 0.01, str(ival), size=12, color="blue", transform=ax.transAxes, verticalalignment="top", horizontalalignment="center")

# 地上気圧の極小
minid = detect_peaks(dsp['mslp'].values, filter_size=50, dist_cut=25.0, flag=1)
for j in range(len(minid[0])):
    wlon = dsp['lon'][minid[1][j]]
    wlat = dsp['lat'][minid[0][j]]
    fig_z, _, _ = transform_lonlat_to_figure((wlon, wlat), ax, proj)
    if fig_z[0] > 0.05 and fig_z[0] < 0.95 and fig_z[1] > 0.05 and fig_z[1] < 0.95:
        ax.plot(wlon, wlat, marker='x', markersize=4, color="red", transform=proj)
        ax.text(wlon, wlat + 0.5, 'L', size=24, color="red", transform=proj)
        val = dsp['mslp'].values[minid[0][j]][minid[1][j]]
        ival = int(val)
        ax.text(fig_z[0], fig_z[1] - 0.01, str(ival), size=12, color="red", transform=ax.transAxes, verticalalignment="top", horizontalalignment="center")

ax.coastlines(resolution='50m', linewidth=1.6)
xticks = np.arange(0, 360, dlon)
yticks = np.arange(-90, 90.1, dlat)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, alpha=0.8)
gl.xlocator = mticker.FixedLocator(xticks)
gl.ylocator = mticker.FixedLocator(yticks)
fig3.text(0.5, 0.01, dt_str + " Sea Level Pressure(hPa)", ha='center', va='bottom', size=20)
output_fig_nm = "{0}_surf.png".format(dt_str)
out_path = os.path.join(output_dir, output_fig_nm)
plt.savefig(out_path)
print("output:{}".format(output_fig_nm))
plt.show()


# --- 高層気象観測データ取得（最大4並列） ---
all_data = {}


def fetch_station_data(row):
    wmo_code = str(int(row['wmo'])).zfill(5)
    lat = row['lat']
    lon = row['lon']
    max_station_attempts = 3

    for attempt in range(1, max_station_attempts + 1):
        try:
            status, data = fetch_wyoming_data_all(wmo_code, vt)

            if status == "success":
                return wmo_code, {"lon": lon, "lat": lat, "data": data}, "success"

            if status == "no_data":
                return wmo_code, None, "no_data"

            if status == "request_error":
                print(f"Wyoming {wmo_code}: リクエストエラーのため取得を終了: {data}")
                return wmo_code, None, "request_error"

            if status == "http_error":
                print(f"Wyoming {wmo_code}: HTTPエラーのため取得を終了: {data}")
                return wmo_code, None, "failed"

            if attempt >= max_station_attempts:
                print(f"Wyoming {wmo_code}: {status}: {data}; 最大試行回数に到達 ({attempt}回目)")
                return wmo_code, None, "failed"

            wait = min(30 * (2 ** (attempt - 1)), 120)
            print(f"Wyoming {wmo_code}: {status}: {data}; {wait}秒後に再取得 ({attempt}回目)")
            time.sleep(wait)

        except requests.exceptions.RequestException as e:
            if attempt >= max_station_attempts:
                print(f"Wyoming {wmo_code}: 通信失敗: {e}")
                return wmo_code, None, "failed"
            wait = min(30 * (2 ** (attempt - 1)), 120)
            print(f"Wyoming {wmo_code}: 通信エラー: {e}; {wait}秒後に再取得 ({attempt}回目)")
            time.sleep(wait)

        except Exception as e:
            print(f"Error: {wmo_code} unexpected error: {e}")
            return wmo_code, None, "failed"

    return wmo_code, None, "failed"


fetch_status = {"success": 0, "no_data": 0, "request_error": 0, "failed": 0}

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(fetch_station_data, row) for _, row in df_sel.iterrows()]
    for future in as_completed(futures):
        wmo_code, result, status = future.result()
        fetch_status[status] += 1
        if result is not None:
            all_data[wmo_code] = result

print("高層観測データ取得結果:")
for status, count in fetch_status.items():
    print(f"  {status}: {count}")


# --- 各気圧面の高層天気図 ---
for tagHp in [300, 400, 500, 700, 850, 925]:
    lons, lats, temps, ttds, u_winds, v_winds = [], [], [], [], [], []

    for wmo, info in all_data.items():
        layer = info["data"].get(tagHp)
        if layer:
            temp, ttd, u, v = layer
            lons.append(info["lon"])
            lats.append(info["lat"])
            temps.append(temp)
            ttds.append(ttd)
            u_winds.append(u)
            v_winds.append(v)

    fig = plt.figure(figsize=(10, 8))
    plt.subplots_adjust(left=0, right=1, bottom=0.06, top=0.98)
    ax = fig.add_subplot(1, 1, 1, projection=proj)

    if tagHp in [850, 925]:
        ax.set_extent([125, 145, 30, 45], proj)
    else:
        ax.set_extent(i_area, proj)

    if tagHp in [300, 400]:
        step1, step2 = 120, 600
    else:
        step1, step2 = 60, 300

    cn_hgt1 = ax.contour(dsp['lon'], dsp['lat'], dsp['Geopotential_height'][np.where(levels == tagHp)[0][0], :, :], colors='black', linewidths=1.5, levels=np.arange(0, 12000, step1), transform=proj)
    ax.clabel(cn_hgt1, fontsize=18, inline=True, colors='black', inline_spacing=5, fmt='%i', rightside_up=True)

    cn_hgt2 = ax.contour(dsp['lon'], dsp['lat'], dsp['Geopotential_height'][np.where(levels == tagHp)[0][0], :, :], colors='black', linewidths=1.5, levels=np.arange(0, 12000, step2), transform=proj)
    ax.clabel(cn_hgt2, fontsize=15, inline=True, colors='black', inline_spacing=0, fmt='%i', rightside_up=True)

    ax.contourf(dsp['lon'], dsp['lat'], dsp['ttd'][np.where(levels == tagHp)[0][0], :, :], [3, 18], colors=['green', '1.0', 'yellow'], extend='both', transform=proj, alpha=0.5)

    cn_tmp0 = ax.contour(dsp['lon'], dsp['lat'], dsp['temperature'][np.where(levels == tagHp)[0][0], :, :], colors='red', linewidths=1.0, linestyles='solid', levels=np.arange(-60, 42, 3), transform=proj)
    ax.clabel(cn_tmp0, fontsize=12, inline=True, inline_spacing=5, fmt='%i', rightside_up=True, colors='red')

    cn_tmp1 = ax.contour(dsp['lon'], dsp['lat'], dsp['temperature'][np.where(levels == tagHp)[0][0], :, :], colors='red', linewidths=2.0, linestyles='solid', levels=np.arange(-60, 42, 6), transform=proj)

    if tagHp == 300:
        ax.contourf(dsp['lon'], dsp['lat'], dsp['wind_speed'][np.where(levels == 300)[0][0], :, :].values, [60, 80, 100], colors=['cyan', 'pink', 'magenta'], extend='max', transform=proj, alpha=0.5)

        # 300hPa強風軸：50kt以上の局所風速極大を連続追跡
        wind300 = dsp['wind_speed'][np.where(levels == 300)[0][0], :, :].values
        jet_lons, jet_lats, jet_speeds, jet_valid = detect_jet_axis(wind300, dsp['lat'].values, dsp['lon'].values, threshold=50.0)

        ax.plot(jet_lons, np.where(jet_valid, jet_lats, np.nan), color='blue', linewidth=2.2, transform=proj, zorder=8)

        jet_core = jet_valid & (jet_speeds >= 70.0)
        ax.scatter(jet_lons[jet_core], jet_lats[jet_core], color='blue', marker='o', s=12, edgecolors='white', linewidths=0.4, transform=proj, zorder=9)

    elif tagHp == 500:
        ax.contourf(dsp['lon'], dsp['lat'], ndimage.gaussian_filter(dsp['vorticity'][np.where(levels == 500)[0][0], :, :] * 1000000, sigma=1), np.arange(0, 1000, 6), cmap='Oranges', extend='max', transform=proj, alpha=0.5)

        # 500hPaトラフ・リッジ：南北方向の高度極値を連続追跡
        height500 = dsp['Geopotential_height'][np.where(levels == 500)[0][0], :, :].values
        trough_lons, trough_lats, ridge_lons, ridge_lats = detect_trough_ridge(height500, dsp['lat'].values, dsp['lon'].values)

        ax.plot(trough_lons, trough_lats, color='red', linewidth=2.5, linestyle='-', transform=proj, zorder=8, label='Trough axis')
        ax.plot(ridge_lons, ridge_lats, color='blue', linewidth=2.5, linestyle='-', transform=proj, zorder=8, label='Ridge axis')

    elif tagHp == 700:
        ax.contourf(dsp['lon'], dsp['lat'], ndimage.gaussian_filter(dsp['omega'][np.where(levels == 700)[0][0], :, :], sigma=1), np.arange(0, 100, 6), cmap='Reds', extend='max', transform=proj, alpha=0.5)

    # 気温の極大
    maxid = detect_peaks(dsp['temperature'][np.where(levels == tagHp)[0][0], :, :].values, filter_size=25, dist_cut=10.0)
    for j in range(len(maxid[0])):
        wlon = dsp['lon'][maxid[1][j]]
        wlat = dsp['lat'][maxid[0][j]]
        fig_z, _, _ = transform_lonlat_to_figure((wlon, wlat), ax, proj)
        if fig_z[0] > 0.0 and fig_z[0] < 1.0 and fig_z[1] > 0.0 and fig_z[1] < 1.0:
            ax.text(wlon, wlat, 'W', size=12, color="red", ha='center', va='center', transform=proj)

    # 気温の極小
    minid = detect_peaks(dsp['temperature'][np.where(levels == tagHp)[0][0], :, :].values, filter_size=25, dist_cut=10.0, flag=1)
    for j in range(len(minid[0])):
        wlon = dsp['lon'][minid[1][j]]
        wlat = dsp['lat'][minid[0][j]]
        fig_z, _, _ = transform_lonlat_to_figure((wlon, wlat), ax, proj)
        if fig_z[0] > 0.0 and fig_z[0] < 1.0 and fig_z[1] > 0.0 and fig_z[1] < 1.0:
            ax.text(wlon, wlat, 'C', size=12, color="blue", ha='center', va='center', transform=proj)

    # 高度の極大
    maxid = detect_peaks(dsp['Geopotential_height'][np.where(levels == tagHp)[0][0], :, :].values, filter_size=50, dist_cut=20.0)
    for j in range(len(maxid[0])):
        wlon = dsp['lon'][maxid[1][j]]
        wlat = dsp['lat'][maxid[0][j]]
        fig_z, _, _ = transform_lonlat_to_figure((wlon, wlat), ax, proj)
        if fig_z[0] > 0.0 and fig_z[0] < 1.0 and fig_z[1] > 0.0 and fig_z[1] < 1.0:
            ax.text(wlon, wlat, 'H', size=24, color="blue", ha='center', va='center', transform=proj)

    # 高度の極小
    minid = detect_peaks(dsp['Geopotential_height'][np.where(levels == tagHp)[0][0], :, :].values, filter_size=50, dist_cut=20.0, flag=1)
    for j in range(len(minid[0])):
        wlon = dsp['lon'][minid[1][j]]
        wlat = dsp['lat'][minid[0][j]]
        fig_z, _, _ = transform_lonlat_to_figure((wlon, wlat), ax, proj)
        if fig_z[0] > 0.0 and fig_z[0] < 1.0 and fig_z[1] > 0.0 and fig_z[1] < 1.0:
            ax.text(wlon, wlat, 'L', size=24, color="red", ha='center', va='center', transform=proj)

    ax.coastlines(resolution='50m')
    xticks = np.arange(0, 360.1, dlon)
    yticks = np.arange(-90, 90.1, dlat)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, alpha=0.8)
    gl.xlocator = mticker.FixedLocator(xticks)
    gl.ylocator = mticker.FixedLocator(yticks)

    temps_arr = np.asarray(temps, dtype=float)
    ttds_arr = np.asarray(ttds, dtype=float)
    u_winds_arr = np.asarray(u_winds, dtype=float)
    v_winds_arr = np.asarray(v_winds, dtype=float)
    mask = np.isfinite(temps_arr) & np.isfinite(ttds_arr) & np.isfinite(u_winds_arr) & np.isfinite(v_winds_arr)

    lons_p = np.asarray(lons)[mask]
    lats_p = np.asarray(lats)[mask]
    temps_p = temps_arr[mask]
    ttds_p = ttds_arr[mask]
    u_winds_p = u_winds_arr[mask]
    v_winds_p = v_winds_arr[mask]

    stationplot = StationPlot(ax, lons_p, lats_p, transform=ccrs.PlateCarree(), fontsize=12)
    stationplot.plot_parameter('NW', temps_p, color='red')
    stationplot.plot_parameter('SW', ttds_p, color='blue')
    stationplot.plot_barb(u_winds_p, v_winds_p, length=5)

    fig.text(0.5, 0.01, dt_str + " {0}hPa Tmp, T-Td".format(int(tagHp)), ha='center', va='bottom', size=20)
    out_fn = "{0}_{1:03d}ttd.png".format(dt_str, tagHp)
    out_path = os.path.join(output_dir, out_fn)
    plt.savefig(out_path)
    print("output:{}".format(out_fn))
    plt.show()
