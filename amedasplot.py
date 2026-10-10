import argparse, json, math, matplotlib, os, pytz, sys, csv, re, requests
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import pandas.tseries.offsets as offsets
from datetime import datetime, timedelta
from metpy.units import units
from scipy.interpolate import Rbf
from scipy.ndimage import maximum_filter, minimum_filter
from urllib.request import urlopen
from matplotlib.colors import ListedColormap, BoundaryNorm
from bs4 import BeautifulSoup

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
          d = math.sqrt(((y[i] - y[j])*(y[i] - y[j])) + ((x[i] - x[j])*(x[i] - x[j])))
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

def parse_datetime(arg):
    try:
        # 引数が12桁の数字の場合、YYYYMMDDHHMM形式の文字列を解析
        if len(arg) == 12 and arg.isdigit():
            year = arg[:4]
            month = arg[4:6]
            day = arg[6:8]
            hour = arg[8:10]
            min = arg[10:12]
            dt = datetime(int(year), int(month), int(day), int(hour), int(min))
        elif len(arg) == 10 and arg.isdigit():
            year = arg[:4]
            month = arg[4:6]
            day = arg[6:8]
            hour = arg[8:10]
            min = 0
            dt = datetime(int(year), int(month), int(day), int(hour), int(min))        
        else: 
            raise ValueError()
        return dt        
    except ValueError:
        return None
   
# 緯度経度で指定したポイントの図上の座標などを取得する関数 
# 図法の座標 => pixel座標 => 図の座標　と3回の変換を行う
# pixel座標: plt.figureで指定した大きさxDPIに合わせ、左下を原点とするpixelで測った座標   
# 図の座標: axesで指定した範囲を(0,1)x(0,1)とする座標
# 3つの座標（図の座標, Pixel座標, 図法の座標）を出力する 

def transform_lonlat_to_figure(lonlat, ax, proj):
    # lonlat:経度と緯度(lon, lat)
    # ax: Axes図の座標系 例：fig.add_subplot()の戻り値
    # proj: axで指定した図法     
    # 図法の変換：参照  https://scitools.org.uk/cartopy/docs/v0.14/crs/index.html                    
    point_proj = proj.transform_point(*lonlat, ccrs.PlateCarree())
    # pixel座標へ変換：参照　https://matplotlib.org/stable/tutorials/advanced/transforms_tutorial.html
    point_pix = ax.transData.transform(point_proj)
    # 図の座標へ変換                                                           
    point_fig = ax.transAxes.inverted().transform(point_pix)
    return point_fig, point_pix, point_proj

# 1地点のアメダスjsonデータから、elem要素で指定した値を返す(ただしFlagが 0以外は Noneとする)
# 要素:elem = 'temp','humidity','snow1h','snow6h','snow12h','snow24h','sun10m','sun1h','precipitation10m','precipitation1h','precipitation3h','precipitation24h','wind','windDirection'
def get_obs_value(amd_obs,elem):
    try:
        et = amd_obs[elem]
        if int(et[1]) != 0:
            return None
        return float(et[0])
    except Exception:
        return None
    
# 描画指定：順に気温(左上),湿球温度(右下),露点温度(左下))
temp_dispflag = False
wbt_dispflag = False
dp_dispflag = False

char_size=8 # 文字サイズ
barb_length=6 # 矢羽の長さ

# 描画地域と描画時刻の設定
if len(sys.argv) == 2:
    dt = parse_datetime(arg)
else:
    jst = pytz.timezone('Asia/Tokyo')
    dt = datetime.now(jst) - timedelta(minutes=30)

# 描画開始 
if dt:
    year=dt.year
    month=dt.month
    day=dt.day
    hour=dt.hour
    min=0
else:
    print('Usage: python script.py [YYYYMMDDHH(MM)]')
    exit()

# 観測データJSONの url作成
url_data_json = 'https://www.jma.go.jp/bosai/amedas/data/map/{:4d}{:02d}{:02d}{:02d}{:02d}00.json'
url_data_json = url_data_json.format(year,month,day,hour,min)

# 気象庁HPからアメダスデータを読み込む
response = urlopen(url_data_json)
content = response.read()
response.close()
data_json=content.decode()
dat_json = json.loads(data_json)

# アメダス地点Tableを読み込む
response = urlopen("https://www.jma.go.jp/bosai/amedas/const/amedastable.json")
content = response.read()
response.close()
station_json=content.decode()
amd_json = json.loads(station_json)

# アメダスデータと同じ時刻のUTCを計算
time = pd.Timestamp(year,month,day,hour,min)
utc = time - offsets.Hour(9)

year = utc.strftime("%Y")
month = utc.strftime("%m")
day = utc.strftime("%d")
hour = utc.strftime("%H")

# URLからHTMLを取得
url_buoy = "https://www.ndbc.noaa.gov/ship_obs.php?uom=M&time=1"
response = requests.get(url_buoy)
html = response.text

# HTMLを解析
soup = BeautifulSoup(html, "html.parser")
ship_data = soup.find_all("span")

# ヘッダー行を特定する
header_row = None
for data in ship_data:
    data_str = data.text.split()
    if len(data_str) >= 22 and all(char in data_str[0] for char in "SHIP"):
        header_row = data
        break

# データを格納する配列
lat_list_p = []
lon_list_p = []
npre_list = []
lat_list_w = []
lon_list_w = []
wind_x_components = []
wind_y_components = []

# ヘッダー行以降のデータを処理
if header_row:
    for data in ship_data[ship_data.index(header_row) + 1:]:
        data_str = data.text.split()
        if len(data_str) >= 22 and "SHIP" in data_str[0] and "{}".format(utc.strftime("%H")) in data_str[1]:
            lat = float(data_str[2])
            lon = float(data_str[3])
            #if lat > 0 and lon >= 90 and lon <= 180:
            if lat > 0 or lon > 90 :
                wdir = data_str[4]
                wspd = data_str[5]
                pres = data_str[9]

                if pres != "-":
                ##if pres == "114514.810":
                    lat_list_p.append(lat)
                    lon_list_p.append(lon)                    
                    npre_list.append(pres)
                
                # 風のx成分とy成分の計算
                if wdir != "-" and wspd != "-" and wdir != "VRB":
                    wdir_rad = math.radians(float(wdir))
                    wx = float(wspd) * math.cos(wdir_rad)
                    wy = float(wspd) * math.sin(wdir_rad)
                    # データを各配列に追加
                    lat_list_w.append(lat)
                    lon_list_w.append(lon)
                    wind_x_components.append(wx)
                    wind_y_components.append(wy)

# 前1時間の雷実況
for i in range(1,12):
    time_liden = utc - offsets.Minute(5*i)

    # LIDENデータのURL
    data_url = "https://www.jma.go.jp/bosai/jmatile/data/nowc/{}00/none/{}00/surf/liden/data.geojson?id=liden"
    data_url=data_url.format(time_liden.strftime("%Y%m%d%H%M"),time_liden.strftime("%Y%m%d%H%M"))

    # データの取得
    response = requests.get(data_url)
    data = response.json()

    # データの解析
    lons_liden = []
    lats_liden = []

    for feature in data['features']:
        coordinates = feature['geometry']['coordinates']
        lon, lat = coordinates
        lons_liden.append(lon)
        lats_liden.append(lat)

### 解析雨量
# データのURL
data_url = "https://www.jma.go.jp/bosai/jmatile/data/rasrf/{}00/immed/{}00/surf/rasrf_point/data.geojson?id=rasrf_point"
data_url=data_url.format(utc.strftime("%Y%m%d%H%M"),utc.strftime("%Y%m%d%H%M"))

# データの取得
response = requests.get(data_url)
data = json.loads(response.text)
 
# 座標と値のリストを作成
coordinates = []
values = []
for feature in data["features"]:
    coordinate = feature["geometry"]["coordinates"]
    value = float(feature["properties"]["value"])
    coordinates.append(coordinate)
    values.append(value)
 
# 座標データをNumPy配列に変換
coordinates = np.array(coordinates)
x = coordinates[:, 0]
y = coordinates[:, 1]

# 値データをNumPy配列に変換
values = np.array(values)

# 解析雨量の最小値は0.4??
values[values == 0.4] = 0.0

# グリッドの作成
xi = np.linspace(np.min(x), np.max(x), 352)
yi = np.linspace(np.min(y), np.max(y), 429)
xi, yi = np.meshgrid(xi, yi)
 
# 値データを補間
zi = np.zeros_like(xi)
for i in range(len(values)):
    xi_index = np.abs(xi[0] - x[i]).argmin()
    yi_index = np.abs(yi[:, 0] - y[i]).argmin()
    zi[yi_index, xi_index] = values[i]

# メッシュグリッドの作成
grid_lon_s, grid_lat_s = np.meshgrid(np.arange(120, 150 + 0.0625, 0.0625), np.arange(22.4, 47.6, 0.05))

# 図法指定                                                                             
proj = ccrs.PlateCarree()

#カラーバーの設定(気象庁RGBカラー)
jmacolors=np.array([[0.95,0.95,0.95,1],[0.63,0.82,0.99,1],[0.13,0.55,0.99,1],[0.00,0.25,0.99,1],[0.98,0.96,0.00,1],[0.99,0.60,0.00,1],[0.99,0.16,0.00,1],[0.71,0.00,0.41,1]])

#等高線値
clevs = np.array([1,5,10,20,30,50,80]) 

for area in [0, 1, 2, 3]:
    # 地図の描画範囲指定
    if (area == 0):
        i_area = [139, 147, 40, 46]
        areaname="Hokkaido"
    elif (area == 1):
        i_area = [134, 142, 33, 39]
        areaname="East"
    elif (area == 2):
        i_area = [128, 136, 31, 37]
        areaname="West"
    elif (area == 3):
        i_area = [135, 143, 36, 42]
        areaname="Tohoku"
        
    # 図のSIZE指定inch                                                                        
    fig = plt.figure(figsize=(8,6))
    # 余白設定                                                                                
    plt.subplots_adjust(left=0.04, right=1.1, bottom=0.0, top=1.0)                  
    # 作図                                                                                    
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    ax.set_extent(i_area, proj)

    # カラーマップの作成
    norm = BoundaryNorm(clevs, len(clevs) - 1)

    # 配列の宣言
    lat_list_t = []
    lon_list_t = []
    temp_list = []
    
    # レーダーGPV描画
    ## 経度方向： 118°〜150° (1/80°間隔)
    ## 緯度方向： 20°〜48° (1/120°間隔)
    lon = np.arange(118, 150, 1/80)
    lat = np.arange(20, 48, 1/120)
    LON, LAT = np.meshgrid(lon, lat)
    LON, LAT = LON.T, LAT.T
    cs = ax.contourf(xi, yi, zi, levels=clevs, cmap=ListedColormap(jmacolors), norm=norm)
    cb = plt.colorbar(cs, orientation="vertical", ticks=clevs, shrink=0.6)    

    # 地点プロット                                                                                                 
    for stno,val in dat_json.items():
        # 緯度・経度のtuple(度分形式)をtuple(度単位)に変換
        wlat = amd_json[stno]['lat'][0] + amd_json[stno]['lat'][1]/60.0
        wlon = amd_json[stno]['lon'][0] + amd_json[stno]['lon'][1]/60.0
        walt = amd_json[stno]['alt']
        # 天気
        weather = get_obs_value(val,'weather')
        # 風
        wind_ok = True
        ws = get_obs_value(val,'wind')
        wd = get_obs_value(val,'windDirection')
        if ws is not None and wd is not None:
            # 16方位風向と風速から、u,v作成   
            if ws == None or wd == None:
                u = None
                v = None
            else:
                wd = wd / 8.0 * math.pi  # 1/8 = (360/16) / 360 * 2
                au = -1.0 * ws * math.sin(wd)
                av = -1.0 * ws * math.cos(wd)
        else:
            wind_ok = False
        # 気温
        temp = get_obs_value(val,'temp')
        if temp is None:
            temp = np.nan
        elif walt < 800: #標高の高い観測点は無視する
            # 配列に格納
            tempsl = temp
            lat_list_t.append(wlat)
            lon_list_t.append(wlon)
            temp_list.append(tempsl)
        # 湿度
        hu = get_obs_value(val,'humidity')
        if hu is None:
            hu = -1.0
        # 露点温度
        dp_temp = -200.0
        if hu >= 0.0 and temp > -200.0:
            dp_temp = mpcalc.dewpoint_from_relative_humidity(temp * units.degC,hu/100.0).m
        # 更正気圧
        npre = get_obs_value(val,'normalPressure')
        if npre is None:
            npre = np.nan
        else:
            # 配列に格納
            lat_list_p.append(wlat)
            lon_list_p.append(wlon)
            npre_list.append(npre)   
        # 気圧
        pre = get_obs_value(val,'pressure')
        if pre is None:
            pre = -1.0

        # 湿球温度
        wb_temp = -200.0
        if dp_temp > -200.0 and temp > -200.0 and pre > 0.0:
            wb_temp = mpcalc.wet_bulb_temperature(pre * units.hPa, temp * units.degC, dp_temp * units.degC).m
        
        ## プロット
        fig_z, _, _ = transform_lonlat_to_figure((wlon,wlat),ax,proj) 
        if ( fig_z[0] > 0.01 and fig_z[0] < 0.99  and fig_z[1] > 0.01 and fig_z[1] < 0.99):
            if weather == 0:
                color="orange"
            elif weather == 1:
                color="gray"
            elif weather == 7:
                color="green"
            elif weather == 10:
                color="blue"
            else:
                color="none"
            ax.plot(wlon, wlat, marker='o', markersize=8, color=color, transform=proj)
            if wind_ok and au*au+av*av>4.0: # 矢羽プロット
                ax.barbs(wlon, wlat, (au * units('m/s')).to('kt').m, (av * units('m/s')).to('kt').m, length=barb_length, transform=proj)
            if temp_dispflag and temp > -200.0: # 気温プロット
                ax.text(fig_z[0]-0.025, fig_z[1]+0.015,'{:5.1f}'.format(temp),size=char_size, color="red", transform=ax.transAxes, verticalalignment="top", horizontalalignment="center")
            if wbt_dispflag and wb_temp > -200.0: # 湿球温度プロット
                ax.text(fig_z[0]+0.025, fig_z[1]-0.003,'{:5.1f}'.format(wb_temp),size=char_size, color=purple, transform=ax.transAxes, verticalalignment="top", horizontalalignment="center")
            if dp_dispflag and dp_temp > -200.0: # 露点温度プロット
                ax.text(fig_z[0]-0.025, fig_z[1]-0.003,'{:5.1f}'.format(dp_temp),size=char_size, color=green, transform=ax.transAxes, verticalalignment="top", horizontalalignment="center")  

    # RBF補間器を作成（multiquadricが一般的、smoothでなめらかさ調整）
    rbf_temp = Rbf(lon_list_t, lat_list_t, temp_list, function='thin_plate', smooth=0.0001)
    rbf_npre = Rbf(lon_list_p, lat_list_p, npre_list, function='thin_plate', smooth=0.0001)

    # グリッド上に評価
    grid_temp = rbf_temp(grid_lon_s, grid_lat_s)
    grid_npre = rbf_npre(grid_lon_s, grid_lat_s)

    # 描画領域のデータを切り出す（等圧線のラベルを表示するためのおまじない）
    lon_range = np.where((grid_lon_s[0, :] >= i_area[0] - 0.25) & (grid_lon_s[0, :] <= i_area[1] + 0.25))
    lat_range = np.where((grid_lat_s[:, 0] >= i_area[2] - 0.25) & (grid_lat_s[:, 0] <= i_area[3] + 0.25))

    # 切り出したい範囲のインデックスを取得
    lon_indices = lon_range[0]
    lat_indices = lat_range[0]

    # 切り出し
    grid_lon_sliced = grid_lon_s[lat_indices][:, lon_indices]
    grid_lat_sliced = grid_lat_s[lat_indices][:, lon_indices]
    psea_grid = grid_npre[lat_indices][:, lon_indices]
    temp_grid = grid_temp[lat_indices][:, lon_indices]

    # BUOY
    ax.barbs(lat_list_w, lon_list_w, (wind_x_components * units('m/s')).to('kt').m, (wind_y_components * units('m/s')).to('kt').m, length=barb_length, transform=proj,color='magenta')

    # 等温線をプロット
    levels = np.arange(-30, 45, 3)
    #cont = plt.contour(grid_lon_sliced, grid_lat_sliced, temp_grid, levels=levels, linewidths=2, linestyles='solid', colors='red')
    #plt.clabel(cont, fontsize=15)

    # 等圧線をプロット
    levels = np.arange(900, 1050, 1)
    cont = plt.contour(grid_lon_sliced, grid_lat_sliced, psea_grid, levels=levels, linewidths=2, colors='black')
    plt.clabel(cont, fontsize=15)

    # LIDENプロット
    plt.scatter(lons_liden, lats_liden, marker='x', color='deeppink', s=200)

    ## H stamp
    maxid = detect_peaks(psea_grid, filter_size=40, dist_cut=10)
    for i in range(len(maxid[0])):
        wlon = grid_lon_sliced[0][maxid[1][i]]
        wlat = grid_lat_sliced[maxid[0][i]][0]
        # 図の範囲内に座標があるか確認                                                                           
        fig_z, _, _ = transform_lonlat_to_figure((wlon,wlat),ax,proj)
        if ( fig_z[0] > 0.05 and fig_z[0] < 0.95  and fig_z[1] > 0.05 and fig_z[1] < 0.95):
            ax.plot(wlon, wlat, marker='x' , markersize=16, color="blue", transform=proj)
            ax.text(wlon - 0.12, wlat + 0.12, 'H', size=30, color="blue", transform=proj)
            val = psea_grid[maxid[0][i]][maxid[1][i]]
            ival = int(val)
            ax.text(fig_z[0], fig_z[1] - 0.025, str(ival), size=24, color="blue", transform=ax.transAxes, verticalalignment="top", horizontalalignment="center")

    ## L stamp
    minid = detect_peaks(psea_grid, filter_size=40, dist_cut=10, flag=1)
    for i in range(len(minid[0])):
        wlon = grid_lon_sliced[0][minid[1][i]]
        wlat = grid_lat_sliced[minid[0][i]][0]
        # 図の範囲内に座標があるか確認                                                                           
        fig_z, _, _ = transform_lonlat_to_figure((wlon,wlat),ax,proj)
        if ( fig_z[0] > 0.05 and fig_z[0] < 0.95  and fig_z[1] > 0.05 and fig_z[1] < 0.95):
            ax.plot(wlon, wlat, marker='x' , markersize=16, color="red", transform=proj)
            ax.text(wlon - 0.12, wlat + 0.12, 'L', size=30, color="red", transform=proj)
            val = psea_grid[minid[0][i]][minid[1][i]]
            ival = int(val)
            ax.text(fig_z[0], fig_z[1] - 0.025, str(ival), size=24, color="red", transform=ax.transAxes, verticalalignment="top", horizontalalignment="center")

    # 海岸線
    ax.coastlines(resolution='10m', linewidth=1.6, color='black', alpha=0.8)  
            
    # 図の説明
    plt.title('{}'.format("AMeDAS, RA1h, LIDEN1h"), loc='left',size=15)
    plt.title('{}'.format(time.strftime("%Y-%m-%d %HJST")), loc='right',size=15);
    plt.savefig("Data/latest{}.png".format(areaname), format="png")
    plt.clf()
