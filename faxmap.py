import argparse
import datetime
import math
from io import StringIO
import os
import re
import sys
import time
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

def transform_lonlat_to_figure(lonlat, ax, proj):
    point_proj = proj.transform_point(*lonlat, ccrs.PlateCarree())
    point_pix = ax.transData.transform(point_proj)
    point_fig = ax.transAxes.inverted().transform(point_pix)
    return point_fig, point_pix, point_proj

## 極大/極小ピーク検出関数
def detect_peaks(image, filter_size, dist_cut, flag=0):
    if flag == 0:
        local_max = maximum_filter(image, footprint=np.ones((filter_size, filter_size)), mode='constant')
        detected_peaks = np.ma.array(image, mask=~(image == local_max))
    else:
        local_min = minimum_filter(image, footprint=np.ones((filter_size, filter_size)), mode='constant')
        detected_peaks = np.ma.array(image, mask=~(image == local_min))
    peaks_index = np.where((detected_peaks.mask != True))
    (x, y) = peaks_index
    size = y.size
    dist = np.full((y.size, y.size), -1.0)
    for i in range(size):
        for j in range(size):
            if i == j:
                dist[i][j] = 0.0
            elif i > j:
                d = math.sqrt(((y[i] - y[j]) * (y[i] - y[j])) + ((x[i] - x[j]) * (x[i] - x[j])))
                dist[i][j] = d
                dist[j][i] = d
    Kinrin = []
    dSum = []
    for i in range(size):
        tmpA = []
        distSum = 0.0
        for j in range(size):
            if dist[i][j] < dist_cut and dist[i][j] > 0.0:
                tmpA.append(j)
                distSum = distSum + dist[i][j]
        dSum.append(distSum)
        Kinrin.append(tmpA)
    cutPoint = []
    for i in range(size):
        val = dSum[i]
        val_i = image[x[i]][y[i]]
        for k in Kinrin[i]:
            val_k = image[x[k]][y[k]]
            if flag == 0 and val_i < val_k:
                cutPoint.append(i)
                break
            if flag != 0 and val_i > val_k:
                cutPoint.append(i)
                break
            if val > dSum[k]:
                cutPoint.append(i)
                break
            if val == dSum[k] and i > k:
                cutPoint.append(i)
                break
    newx = []
    newy = []
    for i in range(size):
        if i in cutPoint:
            continue
        newx.append(x[i])
        newy.append(y[i])
    peaks_index = (np.array(newx), np.array(newy))
    return peaks_index

# 高層気象観測データ取得
def fetch_wyoming_data_all(wmo, dt):
    url = f"https://weather.uwyo.edu/wsgi/sounding?datetime={dt.year}-{dt.month:02d}-{dt.day:02d}%20{dt.hour:d}:00:00&id={wmo}&type=TEXT:LIST"
    res = requests.get(url, timeout=60)

    if res.status_code != 200 or "Can't" in res.text or "Can't get" in res.text:
        return None

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
            except ValueError:
                continue

    return data

# 現在のUTC時刻を取得
now_utc = datetime.datetime.now(datetime.UTC)

# 12時間引く
adjusted_time = now_utc - datetime.timedelta(hours=12) - datetime.timedelta(minutes=80)

# 12時間単位で切り捨て
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
        response = requests.get(dat_fld + fname)
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
    coords={
        "level": levels,
        "lat": lats,
        "lon": lons,
    },
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

dsp['u_wind'] = (dsp['u_wind']).metpy.convert_units('knots')
dsp['v_wind'] = (dsp['v_wind']).metpy.convert_units('knots')
dsp['wind_speed'] = (dsp['wind_speed']).metpy.convert_units('knots')

dsp['vorticity'] = mpcalc.vorticity(dsp['u_wind'], dsp['v_wind'])
dsp['temperature'] = (dsp['temperature']).metpy.convert_units(units.degC)
dsp['dewpoint_temperature'] = mpcalc.dewpoint_from_relative_humidity(dsp['temperature'], dsp['relative_humidity'])
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
dsp['temperature'].data = ndimage.gaussian_filter(dsp['temperature'].data, sigma=(0, 3, 3))
dsp['Equivalent_Potential_temperature'].data = ndimage.gaussian_filter(dsp['Equivalent_Potential_temperature'].data, sigma=(0, 1, 1))

dsp['mslp'].data = ndimage.gaussian_filter(dsp['mslp'].data.magnitude, sigma=3) * dsp['mslp'].data.units

url = "https://www.ncei.noaa.gov/pub/data/igra/igra2-station-list.txt"
res = requests.get(url)
colspecs = [
    (0, 11),
    (12, 20),
    (21, 30),
    (31, 36),
    (36, 72),
    (77, 81)
]
names = ['station_id', 'lat', 'lon', 'elev', 'name', 'year']
df = pd.read_fwf(StringIO(res.text), colspecs=colspecs, names=names)

df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
df['elev'] = pd.to_numeric(df['elev'], errors='coerce')
df['wmo'] = pd.to_numeric(df['station_id'].astype(str).str[-5:], errors='coerce')
df['year'] = pd.to_numeric(df['year'], errors='coerce')

df_sel = df[(df['lon'] >= i_area[0]) & (df['lon'] <= i_area[1]) & (df['lat'] >= i_area[2]) & (df['lat'] <= i_area[3]) & (df['wmo'].notnull()) & (df['year'] > 2024)]

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

maxid = detect_peaks(dsp['mslp'].values, filter_size=70, dist_cut=25.0)
for j in range(len(maxid[0])):
    wlon = dsp['lon'][maxid[1][j]]
    wlat = dsp['lat'][maxid[0][j]]
    fig_z, _, _ = transform_lonlat_to_figure((wlon, wlat), ax, proj)
    if (fig_z[0] > 0.05 and fig_z[0] < 0.95 and fig_z[1] > 0.05 and fig_z[1] < 0.95):
        ax.plot(wlon, wlat, marker='x', markersize=4, color="blue", transform=proj)
        ax.text(wlon, wlat + 0.5, 'H', size=24, color="blue", transform=proj)
        val = dsp['mslp'].values[maxid[0][j]][maxid[1][j]]
        ival = int(val)
        ax.text(fig_z[0], fig_z[1] - 0.01, str(ival), size=12, color="blue", transform=ax.transAxes, verticalalignment="top", horizontalalignment="center")

minid = detect_peaks(dsp['mslp'].values, filter_size=70, dist_cut=25.0, flag=1)
for j in range(len(minid[0])):
    wlon = dsp['lon'][minid[1][j]]
    wlat = dsp['lat'][minid[0][j]]
    fig_z, _, _ = transform_lonlat_to_figure((wlon, wlat), ax, proj)
    if (fig_z[0] > 0.05 and fig_z[0] < 0.95 and fig_z[1] > 0.05 and fig_z[1] < 0.95):
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

all_data = {}
for _, row in df_sel.iterrows():
    wmo_code = str(int(row['wmo'])).zfill(5)
    lat = row['lat']
    lon = row['lon']

    try:
        data = fetch_wyoming_data_all(wmo_code, vt)
        if data:
            all_data[wmo_code] = {
                "lon": lon,
                "lat": lat,
                "data": data
            }
        time.sleep(3)
    except Exception as e:
        print(f"Error: {wmo_code} {e}")
        continue

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
        ax.contourf(dsp['lon'], dsp['lat'], dsp['wind_speed'][np.where(levels == tagHp)[0][0], :, :].values, [60, 80, 100], colors=['cyan', 'pink', 'magenta'], extend='max', transform=proj, alpha=0.5)
    elif tagHp == 500:
        vort_val = dsp['vorticity'][np.where(levels == tagHp)[0][0], :, :] * 1000000
        vort_levels = np.sqrt(np.arange(0, 1000000, 1))
        ax.contourf(dsp['lon'], dsp['lat'], vort_val, vort_levels, cmap='Oranges', extend='max', transform=proj, alpha=0.5)
    elif tagHp == 700:
        ax.contourf(dsp['lon'], dsp['lat'], dsp['omega'][np.where(levels == tagHp)[0][0], :, :], np.arange(0, 100, 10), cmap='Reds', extend='max', transform=proj, alpha=0.5)

    maxid = detect_peaks(dsp['temperature'][np.where(levels == tagHp)[0][0], :, :].values, filter_size=25, dist_cut=10.0)
    for j in range(len(maxid[0])):
        wlon = dsp['lon'][maxid[1][j]]
        wlat = dsp['lat'][maxid[0][j]]
        fig_z, _, _ = transform_lonlat_to_figure((wlon, wlat), ax, proj)
        if (fig_z[0] > 0.0 and fig_z[0] < 1.0 and fig_z[1] > 0.0 and fig_z[1] < 1.0):
            ax.text(wlon, wlat, 'W', size=12, color="red", ha='center', va='center', transform=proj)

    minid = detect_peaks(dsp['temperature'][np.where(levels == tagHp)[0][0], :, :].values, filter_size=25, dist_cut=10.0, flag=1)
    for j in range(len(minid[0])):
        wlon = dsp['lon'][minid[1][j]]
        wlat = dsp['lat'][minid[0][j]]
        fig_z, _, _ = transform_lonlat_to_figure((wlon, wlat), ax, proj)
        if (fig_z[0] > 0.0 and fig_z[0] < 1.0 and fig_z[1] > 0.0 and fig_z[1] < 1.0):
            ax.text(wlon, wlat, 'C', size=12, color="blue", ha='center', va='center', transform=proj)

    maxid = detect_peaks(dsp['Geopotential_height'][np.where(levels == tagHp)[0][0], :, :].values, filter_size=50, dist_cut=20.0)
    for j in range(len(maxid[0])):
        wlon = dsp['lon'][maxid[1][j]]
        wlat = dsp['lat'][maxid[0][j]]
        fig_z, _, _ = transform_lonlat_to_figure((wlon, wlat), ax, proj)
        if (fig_z[0] > 0.0 and fig_z[0] < 1.0 and fig_z[1] > 0.0 and fig_z[1] < 1.0):
            ax.text(wlon, wlat, 'H', size=24, color="blue", ha='center', va='center', transform=proj)

    minid = detect_peaks(dsp['Geopotential_height'][np.where(levels == tagHp)[0][0], :, :].values, filter_size=50, dist_cut=20.0, flag=1)
    for j in range(len(minid[0])):
        wlon = dsp['lon'][minid[1][j]]
        wlat = dsp['lat'][minid[0][j]]
        fig_z, _, _ = transform_lonlat_to_figure((wlon, wlat), ax, proj)
        if (fig_z[0] > 0.0 and fig_z[0] < 1.0 and fig_z[1] > 0.0 and fig_z[1] < 1.0):
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

    mask = (np.isfinite(temps_arr) & np.isfinite(ttds_arr) & np.isfinite(u_winds_arr) & np.isfinite(v_winds_arr))
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
