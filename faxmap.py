import argparse
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import datetime
import math
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import metpy.calc as mpcalc
import numpy as np
import os
import pandas as pd
import pygrib
import re
import requests
import scipy.ndimage as ndimage
import sys
import time
import xarray as xr
from io import StringIO
from metpy.calc import find_peaks
from metpy.plots.wx_symbols import sky_cover
from metpy.plots import StationPlot, scattertext
from metpy.units import units
from scipy.ndimage import gaussian_filter,maximum_filter, minimum_filter
from scipy.signal import savgol_filter

def transform_lonlat_to_figure(lonlat, ax, proj):
    # lonlat:経度と緯度  (lon, lat)
    # ax: Axes図の座標系    ex. fig.add_subplot()の戻り値
    # proj: axで指定した図法
    #
    # 例 緯度経度をpointで与え、ステレオ図法る場合
    #    point = (140.0,35.0)
    #    proj= ccrs.Stereographic(central_latitude=60, central_longitude=140)
    #    fig = plt.figure(figsize=(20,16))
    #    ax = fig.add_subplot(1, 1, 1, projection=proj)
    #    ax.set_extent([108, 156, 17, 55], ccrs.PlateCarree())
    #
    ## 図法の変換
    # 参照  https://scitools.org.uk/cartopy/docs/v0.14/crs/index.html
    point_proj = proj.transform_point(*lonlat, ccrs.PlateCarree())
    #
    # pixel座標へ変換
    # 参照　https://matplotlib.org/stable/tutorials/advanced/transforms_tutorial.html
    point_pix = ax.transData.transform(point_proj)
    #
    # 図の座標へ変換
    point_fig = ax.transAxes.inverted().transform(point_pix)
    return point_fig, point_pix, point_proj
 
## 極大/極小ピーク検出関数
def detect_peaks(image, filter_size, dist_cut, flag=0):
    # filter_size: この値xこの値 の範囲内の最大値のピークを検出
    # dist_cut: この距離内のピークは1つにまとめる
    # flag:  0:maximum検出  0以外:minimum検出
    if flag==0:
      local_max = maximum_filter(image, footprint=np.ones((filter_size, filter_size)), mode='constant')
      detected_peaks = np.ma.array(image, mask=~(image == local_max))
    else:
      local_min = minimum_filter(image, footprint=np.ones((filter_size, filter_size)), mode='constant')
      detected_peaks = np.ma.array(image, mask=~(image == local_min))
    peaks_index = np.where((detected_peaks.mask != True))
    # peak間の距離行例を求める
    (x,y) = peaks_index
    size=y.size
    dist=np.full((y.size, y.size), -1.0)
    for i in range(size):
      for j in range(size):
        if i == j:
          dist[i][j]=0.0
        elif i>j:
          d = math.sqrt(((y[i] - y[j])*(y[i] - y[j]))
                        +((x[i] - x[j])*(x[i] - x[j])))
          dist[i][j]= d
          dist[j][i]= d
    # 距離がdist_cut内のpeaksの距離の和と、そのピーク番号を取得する
    Kinrin=[]
    dSum=[]
    for i in range(size):
      tmpA=[]
      distSum=0.0
      for j in range(size):
        if dist[i][j] < dist_cut and dist[i][j] > 0.0:
          tmpA.append(j)
          distSum=distSum+dist[i][j]
      dSum.append(distSum)
      Kinrin.append(tmpA)
    # Peakから外すPeak番号を求める.  peak間の距離和が最も小さいものを残す
    cutPoint=[]
    for i in range(size):
      val = dSum[i]
      val_i=image[x[i]][y[i]]
      for k in Kinrin[i]:
        val_k=image[x[k]][y[k]]
        if flag==0 and val_i < val_k:
            cutPoint.append(i)
            break
        if flag!=0 and val_i > val_k:
            cutPoint.append(i)
            break
        if val > dSum[k]:
            cutPoint.append(i)
            break
        if val == dSum[k] and i > k:
            cutPoint.append(i)
            break
    # 戻り値用に外すpeak番号を配列から削除
    newx=[]
    newy=[]
    for i in range(size):
      if (i in cutPoint):
        continue
      newx.append(x[i])
      newy.append(y[i])
    peaks_index=(np.array(newx),np.array(newy))
    return peaks_index
        
# 高層気象観測データ取得

def fetch_wyoming_data_all(wmo, dt):
    url = (
        f"https://weather.uwyo.edu/wsgi/sounding?datetime={dt.year}-{dt.month:02d}-{dt.day:02d}%20{dt.hour:d}:00:00&id={wmo}&type=TEXT:LIST"
    )
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

    return data  # {pressure: (temp, ttd, u, v)} 形式で返す
 
# 現在のUTC時刻を取得
now_utc = datetime.datetime.now(datetime.UTC)
 
# 8時間引く
adjusted_time = now_utc - datetime.timedelta(hours=12) - datetime.timedelta(minutes=80)
 
# 6時間単位で切り捨て
truncated_hour = (adjusted_time.hour // 12) * 12
dt = adjusted_time.replace(hour=truncated_hour, minute=0, second=0, microsecond=0)
vt = dt + datetime.timedelta(hours=12)
 
# 個別に取り出したい場合
i_year = dt.year
i_month = dt.month
i_day = dt.day
i_hourZ = dt.hour
 
print("IT(UTC):", dt)
 
# 画像保存ディレクトリ
fig_fld="{0:4d}{1:02d}{2:02d}{3:02d}".format(i_year,i_month,i_day,i_hourZ)
if not os.path.exists(fig_fld):
    os.makedirs(fig_fld)
 
# 描画する範囲の大まかな指定
i_area = [105, 165, 15, 60]  # 日本付近
str_area = "jp"   # ファイル名に利用
 
## GPVの切り出し領域の指定：(lonW,latS)-(lonE,latN)の矩形                                                                                                      
latS=10
latN=60
lonW=100
lonE=170
 
# データの格納先フォルダー名
##!!! GRIB2データの保存先をFolderを指定すること !!!
data_fld="https://storage.googleapis.com/ecmwf-open-data/{0:04d}{1:02d}{2:02d}/{3:02d}z/ifs/0p25/oper/"
dat_fld=data_fld.format(i_year,i_month,i_day,i_hourZ)

# 読み込むGRIB2形式GSMのファイル名
gsm_fn_t="{0:4d}{1:02d}{2:02d}{3:02d}0000-12h-oper-fc.grib2"
fname_gfs = gsm_fn_t.format(i_year,i_month,i_day,i_hourZ)
 
# HTTPでファイルダウンロード
response = requests.get(dat_fld + fname_gfs)
response.raise_for_status()  # ダウンロードに失敗した場合、エラーを発生させる
 
# ダウンロードしたコンテンツをローカルに保存
with open(fname_gfs, 'wb') as f:
    f.write(response.content)
   
# 要素別に読み込み
grbs = pygrib.open(fname_gfs)
 
# 要素別に読み込み（tagHpの等圧面から下部のデータを全て）
grbTm = grbs(shortName="t",typeOfLevel='isobaricInhPa',level=lambda l:l >= 300)
valHt, latHt, lonHt = grbTm[0].data(lat1=latS,lat2=latN,lon1=lonW,lon2=lonE)
 
## データの大きさを調べる
# レベルの数
levels = np.sort(np.array([g['level'] for g in grbTm]))
l_size = levels.size
# 平面のsize
(lat_size, lon_size) = valHt.shape
lats = latHt[:,0]
lons = lonHt[0,:]
 
## 配列確保(0で初期化)
aryHt = np.zeros([l_size, lat_size, lon_size])
aryTm = np.zeros([l_size, lat_size, lon_size])
aryRh = np.zeros([l_size, lat_size, lon_size])
aryWu = np.zeros([l_size, lat_size, lon_size])
aryWv = np.zeros([l_size, lat_size, lon_size])
aryOmg = np.zeros([l_size, lat_size, lon_size])

# 要素別に読み込み
grbs = pygrib.open(fname_gfs)
grbHt = sorted(grbs.select(shortName="gh", typeOfLevel='isobaricInhPa', level=lambda l: l >= 300), key=lambda g: g.level, reverse=False)
grbTm = sorted(grbs.select(shortName="t", typeOfLevel='isobaricInhPa', level=lambda l: l >= 300), key=lambda g: g.level, reverse=False)
grbWu = sorted(grbs.select(shortName="u", typeOfLevel='isobaricInhPa', level=lambda l: l >= 300), key=lambda g: g.level, reverse=False)
grbWv = sorted(grbs.select(shortName="v", typeOfLevel='isobaricInhPa', level=lambda l: l >= 300), key=lambda g: g.level, reverse=False)
grbRh = sorted(grbs.select(shortName="r", typeOfLevel='isobaricInhPa', level=lambda l: l >= 300), key=lambda g: g.level, reverse=False)
grbOmg = sorted(grbs.select(shortName="w", typeOfLevel='isobaricInhPa', level=lambda l: l >= 300), key=lambda g: g.level, reverse=False)

# 気圧
arySlp = np.zeros([lat_size, lon_size]) 
valSlp, _, _ = grbs.select(name='Mean sea level pressure')[0].data(lat1=latS,lat2=latN,lon1=lonW,lon2=lonE)
aryHgt = np.zeros([lat_size, lon_size])
valHgt, _, _ = grbs.select(name='Surface pressure')[0].data(lat1=latS,lat2=latN,lon1=lonW,lon2=lonE)

# 要素毎に3次元配列作成
for l in range(l_size):
    valHt, _, _ = grbHt[l].data(lat1=latS,lat2=latN,lon1=lonW,lon2=lonE)
    valTm, _, _ = grbTm[l].data(lat1=latS,lat2=latN,lon1=lonW,lon2=lonE)
    valWu, _, _ = grbWu[l].data(lat1=latS,lat2=latN,lon1=lonW,lon2=lonE)
    valWv, _, _ = grbWv[l].data(lat1=latS,lat2=latN,lon1=lonW,lon2=lonE)
    valRh, _, _ = grbRh[l].data(lat1=latS,lat2=latN,lon1=lonW,lon2=lonE)
    valOmg, _, _ = grbOmg[l].data(lat1=latS,lat2=latN,lon1=lonW,lon2=lonE)

    ## 4次元配列に代入
    aryHt[l] = valHt
    aryTm[l] = valTm
    aryWu[l] = valWu
    aryWv[l] = valWv
    aryRh[l] = valRh
    aryOmg[l] = valOmg
    arySlp = valSlp
    aryHgt = valHgt
    
# ### Xarry データセットに変換
# GPVをmetpyを利用して物理量を計算するには、numpyの配列データからXarrayのデータセットを作成する
# その後で、必要な物理量を計算する
# Xarray data set作成
ds = xr.Dataset(
    {
        "Geopotential_height": (["level", "lat", "lon"], aryHt * units.meter),
        "temperature": (["level", "lat", "lon"], aryTm * units('K')),
        "relative_humidity": (["level", "lat", "lon"], aryRh * units('%')),
        "u_wind": (["level", "lat", "lon"], aryWu * units('m/s')),
        "v_wind": (["level", "lat", "lon"], aryWv * units('m/s')),
        "omega": (["level", "lat", "lon"], aryOmg / 100 * 3600 * units('Pa/s')), # Pa/s => hPa/h
        "mslp": (["lat", "lon"], arySlp / 100 * units('hPa')),
        "height": (["lat", "lon"], aryHgt / 100 * units('hPa')) # 地上気圧
    },
    coords={
        "level": levels,
        "lat": lats,
        "lon": lons,
    },
)
 
# 単位も入力する
ds['Geopotential_height'].attrs['units'] = 'm'
ds['u_wind'].attrs['units']='m/s'
ds['v_wind'].attrs['units']='m/s'
ds['omega'].attrs['units']='hPa/h'
ds['level'].attrs['units'] = 'hPa'
ds['lat'].attrs['units'] = 'degrees_north'
ds['lon'].attrs['units'] = 'degrees_east'
 
# metpy仕様に変換
dsp= ds.metpy.parse_cf()
 
# 風速
dsp['wind_speed'] = mpcalc.wind_speed(dsp['u_wind'],dsp['v_wind'])
 
# 収束発散                                                                              
dsp['conv'] = mpcalc.divergence(dsp['u_wind'],dsp['v_wind'])
 
# knotsへ変換
dsp['u_wind'] = (dsp['u_wind']).metpy.convert_units('knots')
dsp['v_wind'] = (dsp['v_wind']).metpy.convert_units('knots')
dsp['wind_speed'] = (dsp['wind_speed']).metpy.convert_units('knots')
 
# 相対渦度                                                                                                                          
dsp['vorticity'] = mpcalc.vorticity(dsp['u_wind'],dsp['v_wind'])
 
# 等温度線 実線
dsp['temperature'] = (dsp['temperature']).metpy.convert_units(units.degC)  # Kelvin => Celsius
 
# 露点温度
dsp['dewpoint_temperature'] = mpcalc.dewpoint_from_relative_humidity(dsp['temperature'],dsp['relative_humidity'])
dsp['ttd'] = dsp['temperature'] - dsp['dewpoint_temperature']
 
# 相当温位
dsp['Equivalent_Potential_temperature'] = mpcalc.equivalent_potential_temperature(dsp['level'],dsp['temperature'],dsp['dewpoint_temperature'])

dsp['NewIndex'] = 2 * (dsp['temperature'][np.where(levels == 850)[0][0],:,:] - dsp['temperature'][np.where(levels == 500)[0][0],:,:]) - dsp['ttd'][np.where(levels == 850)[0][0],:,:] - dsp['ttd'][np.where(levels == 700)[0][0],:,:] 

dsp['Geopotential_height'].data = ndimage.gaussian_filter(dsp['Geopotential_height'].data, sigma=(0, 2, 2))
dsp['temperature'].data = ndimage.gaussian_filter(dsp['temperature'].data, sigma=(0, 4, 4))
dsp['temperature'].data = ndimage.gaussian_filter(dsp['temperature'].data, sigma=(0, 6, 6))	
#dsp['ttd'].data = ndimage.gaussian_filter(dsp['ttd'].data, sigma=(0, 2, 2))
dsp['omega'].data = ndimage.gaussian_filter(dsp['omega'].data, sigma=(0, 2, 2))
dsp['Equivalent_Potential_temperature'].data = ndimage.gaussian_filter(dsp['Equivalent_Potential_temperature'].data, sigma=(0, 1, 1))

# 気圧の重み
w = (dsp['mslp'].data.magnitude - dsp['height'].data.magnitude) / 100.0
w = np.clip(w, 0.0, 1.0)

# 平滑化
dsp['mslp'].data = dsp['mslp'].data * (1.0 - w) + gaussian_filter(dsp['mslp'].data, sigma=8) * units('hPa') * w
dsp['mslp'].data = savgol_filter(savgol_filter(dsp['mslp'].data, window_length=21, polyorder=2, axis=0), window_length=21, polyorder=2, axis=1)

# 高層観測地点
url = "https://www.ncei.noaa.gov/pub/data/igra/igra2-station-list.txt"
res = requests.get(url)
colspecs = [
    (0, 11),    # station_id
    (12, 20),   # lat
    (21, 30),   # lon
    (31, 36),   # elev
    (36, 72),   # name
    (77, 81)    # year
]
names = ['station_id', 'lat', 'lon', 'elev', 'name', 'year']
df = pd.read_fwf(StringIO(res.text), colspecs=colspecs, names=names)
 
df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
df['elev'] = pd.to_numeric(df['elev'], errors='coerce')
df['wmo'] = pd.to_numeric(df['station_id'].astype(str).str[-5:], errors='coerce')
df['year'] = pd.to_numeric(df['year'], errors='coerce')

df_sel = df[(df['lon'] >= i_area[0]) & (df['lon'] <= i_area[1]) & (df['lat'] >= i_area[2]) & (df['lat'] <= i_area[3]) & (df['wmo'].notnull()) & (df['year'] > 2024)]
                             
## 年月日                                                                                                    
dt_str = (vt.strftime("%Y%m%d%HUTC")).upper()
dt_str2 = vt.strftime("%Y%m%d%H")
 
# 緯線・経線の指定
dlon,dlat=10,10   # 10度ごとに
 
## 図法指定                                                                            
proj = ccrs.PlateCarree()
## 図のSIZE指定inch                                                                        
fig3 = plt.figure(figsize=(10,8))
## 余白設定                                                                                
plt.subplots_adjust(left=0, right=1, bottom=0.06, top=0.98)                  
## 作図                                                                                    
ax = fig3.add_subplot(1, 1, 1, projection=proj)
ax.set_extent(i_area, proj)
 
# EPT塗りつぶし
cnf_ept = ax.contourf(dsp['lon'], dsp['lat'], dsp['Equivalent_Potential_temperature'][np.where(levels == 850)[0][0],:,:], np.arange(255, 375, 3), cmap="jet", extend='both', transform=proj)

# EPT等値線 実線
cn_ept0 = ax.contour(dsp['lon'], dsp['lat'], dsp['Equivalent_Potential_temperature'][np.where(levels == 850)[0][0],:,:], colors='black', linewidths=0.3, levels=np.arange(255, 375, 3), transform=proj)
#ax.clabel(cn_ept0, fontsize=8, inline=True, inline_spacing=5, fmt='%i', rightside_up=True, colors='black')
cn_ept1 = ax.contour(dsp['lon'], dsp['lat'], dsp['Equivalent_Potential_temperature'][np.where(levels == 850)[0][0],:,:], colors='black', linewidths=1.0, levels=np.arange(255, 375, 15), transform=proj)
ax.clabel(cn_ept1, fontsize=12, inline=True, inline_spacing=5, fmt='%i', rightside_up=True, colors='black')
 
## 海岸線
ax.coastlines(resolution='50m', linewidth=1.6) # 海岸線の解像度を上げる  
 
# グリッド線を引く                                                              
xticks=np.arange(0,360,dlon)
yticks=np.arange(-90,90.1,dlat)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, alpha=0.8)
gl.xlocator = mticker.FixedLocator(xticks)
gl.ylocator = mticker.FixedLocator(yticks)
 
## 風                                                  
wind_slice = (slice(None, None, 8), slice(None, None, 8))
ax.barbs(dsp['lon'][wind_slice[0]], dsp['lat'][wind_slice[1]], dsp['u_wind'][np.where(levels == 850)[0][0],wind_slice[0],wind_slice[1]].values, dsp['v_wind'][np.where(levels == 850)[0][0],wind_slice[0],wind_slice[1]].values, length=5.5, pivot='middle', color='black', transform=proj)
   
## 図の説明
fig3.text(0.5, 0.01, dt_str + " 850hPa EPT(K), Wind" ,ha='center',va='bottom', size=20)

# 出力先ディレクトリを作成
output_dir = os.path.join("Data/", dt_str)
os.makedirs(output_dir, exist_ok=True)  # 再帰的に作成、すでにあってもOK

## file出力
output_fig_nm="{0}_850ept.png".format(dt_str)
out_path = os.path.join(output_dir, output_fig_nm)
plt.savefig(out_path)
print("output:{}".format(output_fig_nm))
plt.show()

## 図法指定                                                                            
proj = ccrs.PlateCarree()
## 図のSIZE指定inch                                                                        
fig3 = plt.figure(figsize=(10,8))
## 余白設定                                                                                
plt.subplots_adjust(left=0, right=1, bottom=0.06, top=0.98)                  
## 作図                                                                                    
ax = fig3.add_subplot(1, 1, 1, projection=proj)
ax.set_extent(i_area, proj)
 
# 地上気圧
clevs_mslp = np.arange(800, 1200, 4)
ax.contour(dsp['lon'], dsp['lat'], dsp['mslp'], clevs_mslp, colors='black', linestyles='solid', linewidths=[1.25, 0.75, 0.75, 0.75, 0.75], transform=proj)

# NewIndex
ax.contourf(dsp['lon'], dsp['lat'], dsp['NewIndex'].values, [30, 50], colors=['yellow','red'], extend='max', transform=proj, alpha=0.5)
# 気圧 H
maxid = detect_peaks(dsp['mslp'].values, filter_size=30, dist_cut=8.0)
for j in range(len(maxid[0])):                                                      
    wlon = dsp['lon'][maxid[1][j]]
    wlat = dsp['lat'][maxid[0][j]]
    # 図の範囲内に座標があるか確認                                                        
    fig_z, _, _ = transform_lonlat_to_figure((wlon,wlat),ax,proj)
    if ( fig_z[0] > 0.05 and fig_z[0] < 0.95  and fig_z[1] > 0.05 and fig_z[1] < 0.95):
        ax.plot(wlon, wlat, marker='x' , markersize=4, color="blue", transform=proj)
        ax.text(wlon, wlat + 0.5, 'H', size=16, color="blue", transform=proj)
        val = dsp['mslp'].values[maxid[0][j]][maxid[1][j]]
        ival = int(val)
        ax.text(fig_z[0], fig_z[1] - 0.01, str(ival), size=12, color="blue", transform=ax.transAxes, verticalalignment="top", horizontalalignment="center")

# 気圧 L
minid = detect_peaks(dsp['mslp'].values, filter_size=30, dist_cut=8.0,flag=1)
for j in range(len(minid[0])):
    wlon = dsp['lon'][minid[1][j]]
    wlat = dsp['lat'][minid[0][j]]
    # 図の範囲内に座標があるか確認                                                        
    fig_z, _, _ = transform_lonlat_to_figure((wlon,wlat),ax,proj)
    if ( fig_z[0] > 0.05 and fig_z[0] < 0.95  and fig_z[1] > 0.05 and fig_z[1] < 0.95):
        ax.plot(wlon, wlat, marker='x' , markersize=4, color="red", transform=proj)
        ax.text(wlon, wlat + 0.5, 'L', size=16, color="red", transform=proj)
        val = dsp['mslp'].values[minid[0][j]][minid[1][j]]
        ival = int(val)
        ax.text(fig_z[0], fig_z[1] - 0.01, str(ival), size=12, color="red", transform=ax.transAxes, verticalalignment="top", horizontalalignment="center")
 
## 海岸線
ax.coastlines(resolution='50m', linewidth=1.6) # 海岸線の解像度を上げる  
 
# グリッド線を引く                                                              
xticks=np.arange(0,360,dlon)
yticks=np.arange(-90,90.1,dlat)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, alpha=0.8)
gl.xlocator = mticker.FixedLocator(xticks)
gl.ylocator = mticker.FixedLocator(yticks)
   
## 図の説明
fig3.text(0.5, 0.01, dt_str + " Sea Level Pressure(hPa)" ,ha='center',va='bottom', size=20)

# 出力先ディレクトリを作成
output_dir = os.path.join("Data/", dt_str)
os.makedirs(output_dir, exist_ok=True)  # 再帰的に作成、すでにあってもOK

## file出力
output_fig_nm="{0}_surf.png".format(dt_str)
out_path = os.path.join(output_dir, output_fig_nm)
plt.savefig(out_path)
print("output:{}".format(output_fig_nm))
plt.show()

# 一度にfetchして保存
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

# 各等圧面ごとに抽出
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
 
    ## 図のSIZE指定inch
    fig = plt.figure(figsize=(10,8))
    ## 余白設定
    plt.subplots_adjust(left=0, right=1, bottom=0.06, top=0.98)
       
    ## 余白  FAX図に合わせる
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent(i_area, proj)
 
    # 等高度線
    cn_hgt1 = ax.contour(dsp['lon'], dsp['lat'], dsp['Geopotential_height'][np.where(levels == tagHp)[0][0],:,:], colors='black', linewidths=1.5, levels=np.arange(0, 12000, 60), transform=proj)
    ax.clabel(cn_hgt1, fontsize=18, inline=True, colors='black', inline_spacing=5, fmt='%i', rightside_up=True)
    cn_hgt2= ax.contour(dsp['lon'], dsp['lat'], dsp['Geopotential_height'][np.where(levels == tagHp)[0][0],:,:], colors='black', linewidths=1.5, levels=np.arange(0, 12000, 300), transform=proj)
    ax.clabel(cn_hgt2, fontsize=15, inline=True, colors='black', inline_spacing=0, fmt='%i', rightside_up=True)
 
    # ハッチ preTTd hPa面 T - Td
    ax.contourf(dsp['lon'], dsp['lat'], dsp['ttd'][np.where(levels == tagHp)[0][0],:,:], [3, 18], colors=['green','1.0','yellow'], extend='both', transform=proj ,alpha=0.5)
 
    # preT hPa面 等温度線
    cn_tmp0 = ax.contour(dsp['lon'], dsp['lat'], dsp['temperature'][np.where(levels == tagHp)[0][0],:,:], colors='red', linewidths=1.0, linestyles='solid', levels=np.arange(-60, 42, 3), transform=proj)
    ax.clabel(cn_tmp0, fontsize=12, inline=True, inline_spacing=5, fmt='%i', rightside_up=True, colors='red')
    cn_tmp1 = ax.contour(dsp['lon'], dsp['lat'], dsp['temperature'][np.where(levels == tagHp)[0][0],:,:], colors='red', linewidths=2.0, linestyles='solid', levels=np.arange(-60, 42, 6), transform=proj)
 
    if tagHp == 300:
        ax.contourf(dsp['lon'], dsp['lat'], dsp['wind_speed'][np.where(levels == tagHp)[0][0],:,:].values, [60,80,100], colors=['cyan','pink','magenta'], extend='max', transform=proj, alpha=0.5)
    elif tagHp == 500:
        ax.contourf(dsp['lon'], dsp['lat'], dsp['vorticity'][np.where(levels == tagHp)[0][0],:,:] * 1000000, np.arange(0, 400, 10), cmap="Oranges", extend='max', transform=proj, alpha=0.5)
    elif tagHp == 700:
        ax.contourf(dsp['lon'], dsp['lat'], dsp['omega'][np.where(levels == tagHp)[0][0],:,:], np.arange(0, 40, 1), cmap="Reds", extend='max', transform=proj, alpha=0.5)
   
    # + stamp
    maxid = detect_peaks(dsp['omega'][np.where(levels == tagHp)[0][0],:,:].values, filter_size=3, dist_cut=4.0)
    for j in range(len(maxid[0])):
        wlon = dsp['lon'][maxid[1][j]]
        wlat = dsp['lat'][maxid[0][j]]
    # 図の範囲内に座標があるか確認
    fig_z, _, _ = transform_lonlat_to_figure((wlon,wlat),ax,proj)
    if ( fig_z[0] > 0.05 and fig_z[0] < 0.95  and fig_z[1] > 0.05 and fig_z[1] < 0.95):
        val = dsp['omega'][i,np.where(levels == tagHp)[0][0],:,:].values[maxid[0][j]][maxid[1][j]]
        ival = int(val)
        if ival > 30:
          ax.plot(wlon, wlat, marker='+' , markersize=8, color="purple", transform=proj)
        if ival > 30:
          ax.text(fig_z[0], fig_z[1] - 0.008, str(ival), size=12, color="purple", transform=ax.transAxes, verticalalignment="top", horizontalalignment="center")
 
    # - stamp
    minid = detect_peaks(dsp['omega'][np.where(levels == tagHp)[0][0],:,:].values, filter_size=3, dist_cut=4.0, flag=1)
    for j in range(len(minid[0])):
        wlon = dsp['lon'][minid[1][j]]
        wlat = dsp['lat'][minid[0][j]]
    # 図の範囲内に座標があるか確認
    fig_z, _, _ = transform_lonlat_to_figure((wlon,wlat),ax,proj)
    if ( fig_z[0] > 0.05 and fig_z[0] < 0.95  and fig_z[1] > 0.05 and fig_z[1] < 0.95):
        val = dsp['omega'][i,np.where(levels == tagHp)[0][0],:,:].values[minid[0][j]][minid[1][j]]
        ival = int(val * -1.0)
        if ival > 30:
            ax.plot(wlon, wlat, marker='_' , markersize=8, color="red",transform=proj)
        if ival > 30:
            ax.text(fig_z[0], fig_z[1] - 0.008, str(ival), size=12, color="red", transform=ax.transAxes, verticalalignment="top", horizontalalignment="center")
 
    ## W スタンプ
    maxid = detect_peaks(dsp['temperature'][np.where(levels == tagHp)[0][0],:,:].values, filter_size=12, dist_cut=2.0)
    for j in range(len(maxid[0])):
        wlon = dsp['lon'][maxid[1][j]]
        wlat = dsp['lat'][maxid[0][j]]
        # 図の範囲内に座標があるか確認
        fig_z, _, _ = transform_lonlat_to_figure((wlon,wlat),ax,proj)
        if ( fig_z[0] > 0.0 and fig_z[0] < 1.0  and fig_z[1] > 0.0 and fig_z[1] < 1.0):
            ax.text(wlon, wlat, 'W', size=12, color="red", ha='center', va='center', transform=proj)
 
    ## C スタンプ
    minid = detect_peaks(dsp['temperature'][np.where(levels == tagHp)[0][0],:,:].values, filter_size=12, dist_cut=2.0, flag=1)
    for j in range(len(minid[0])):
        wlon = dsp['lon'][minid[1][j]]
        wlat = dsp['lat'][minid[0][j]]
        # 図の範囲内に座標があるか確認
        fig_z, _, _ = transform_lonlat_to_figure((wlon,wlat),ax,proj)
        if ( fig_z[0] > 0.0 and fig_z[0] < 1.0  and fig_z[1] > 0.0 and fig_z[1] < 1.0):
            ax.text(wlon, wlat, 'C', size=12, color="blue", ha='center', va='center', transform=proj)
 
    # H stamp                                                                                                
    maxid = detect_peaks(dsp['Geopotential_height'][np.where(levels == tagHp)[0][0],:,:].values, filter_size=10, dist_cut=8.0)
    for j in range(len(maxid[0])):
        wlon = dsp['lon'][maxid[1][j]]
        wlat = dsp['lat'][maxid[0][j]]
        # 図の範囲内に座標があるか確認                                                                          
        fig_z, _, _ = transform_lonlat_to_figure((wlon,wlat),ax,proj)
        if ( fig_z[0] > 0.0 and fig_z[0] < 1.0  and fig_z[1] > 0.0 and fig_z[1] < 1.0):
            ax.text(wlon, wlat, 'H', size=24, color="blue", ha='center', va='center', transform=proj)
 
    # L stamp                                                                                                
    minid = detect_peaks(dsp['Geopotential_height'][np.where(levels == tagHp)[0][0],:,:].values, filter_size=10, dist_cut=8.0, flag=1)
    for j in range(len(minid[0])):
        wlon = dsp['lon'][minid[1][j]]
        wlat = dsp['lat'][minid[0][j]]
        # 図の範囲内に座標があるか確認                                                                          
        fig_z, _, _ = transform_lonlat_to_figure((wlon,wlat),ax,proj)
        if ( fig_z[0] > 0.0 and fig_z[0] < 1.0  and fig_z[1] > 0.0 and fig_z[1] < 1.0):
            ax.text(wlon, wlat, 'L', size=24, color="red", ha='center', va='center', transform=proj)
 
    ## 海岸線など
    ax.coastlines(resolution='50m',) # 海岸線の解像度を上げる
    ## グリッド線を引く
    xticks=np.arange(0,360.1,dlon)
    yticks=np.arange(-90,90.1,dlat)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, alpha=0.8)
    gl.xlocator = mticker.FixedLocator(xticks)
    gl.ylocator = mticker.FixedLocator(yticks)

    # 高層観測プロット
    stationplot = StationPlot(ax, lons, lats, transform=ccrs.PlateCarree(), fontsize=12)
    stationplot.plot_parameter('NW', temps, color='red')
    stationplot.plot_parameter('SW', ttds, color='blue')
    stationplot.plot_barb(u_winds, v_winds, length=5)
 
    ## 図の説明
    fig.text(0.5, 0.01, dt_str + " {0}hPa Tmp, T-Td".format(int(tagHp)), ha='center',va='bottom', size=20)
 
    ## 出力
    out_fn="{0}_{1:03d}ttd.png".format(dt_str,tagHp)
    out_path = os.path.join(output_dir, out_fn)
    plt.savefig(out_path)
    print("output:{}".format(out_fn))
    plt.show()
