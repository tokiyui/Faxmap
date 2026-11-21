import datetime
import email.utils
from io import StringIO
import os
import re
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import metpy.calc as mpcalc
from metpy.plots import StationPlot
from metpy.units import units
import numpy as np
import pandas as pd
import pygrib
import requests
import scipy.ndimage as ndimage
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
import xarray as xr

# ============================================================
# 調整パラメータ（見た目・解析の感度はここで変更）
# ============================================================
STAMP_FONT_SIZE = 24          # H/Lスタンプの大きさ（地上・等圧面共通。中心気圧の数字はその半分）
SLP_SMOOTH_SIGMA = 5          # 地上の等圧線の平滑化（格子数, 0.1度格子）。中心気圧は元の値を保つ
OMEGA_SMOOTH_SIGMA = 5        # 700hPa鉛直速度の平滑化（格子数）
JET_SMOOTH_SIGMA = 5          # 300hPa強風軸の細かさ（格子数）。小さいほど細かい
JET_MIN_LENGTH_KM = 1000.0    # 強風軸の最小長さ(km)
TROUGH_SMOOTH_SIGMA = 20      # 500hPaトラフ・リッジ解析の細かさ（格子数）。小さいほど細かい
TROUGH_MIN_ZETA_C = 2.0       # 軸として採用する曲率渦度の大きさ（1e-5 s^-1）。大きいほど軸が減る
TROUGH_MAX_LENGTH_KM = 1200.0 # 1本の軸の最大長さ(km)
LOW_LEVEL_EXTENT = [128, 146, 31, 44]  # 850/925hPa図の表示範囲 [西, 東, 南, 北]
KM_PER_DEG = 111.2

# ============================================================
# HTTP取得共通設定
# ============================================================
HTTP_SESSION = requests.Session()
HTTP_SESSION.headers.update({"User-Agent": "Mozilla/5.0 (compatible; MeteorologicalAnalysis/2.0; +https://www.noaa.gov/)"})
HTTP_DEFAULT_ATTEMPTS = 4
HTTP_RETRY_BASE_SECONDS = 8.0

def get_with_retry(url, timeout=60, max_wait=180, description="HTTP request", max_attempts=HTTP_DEFAULT_ATTEMPTS, params=None):
    """HTTP取得を有限回だけ再試行し、Retry-Afterがあれば尊重する。"""
    for attempt in range(1, max_attempts + 1):
        try:
            response = HTTP_SESSION.get(url, params=params, timeout=timeout)
            if response.status_code == 429 or 500 <= response.status_code <= 599:
                if attempt >= max_attempts:
                    return response
                retry_after = response.headers.get("Retry-After")
                server_wait = 0.0
                if retry_after:
                    try:
                        server_wait = max(0.0, float(retry_after))
                    except (TypeError, ValueError):
                        try:
                            retry_at = email.utils.parsedate_to_datetime(retry_after)
                            if retry_at.tzinfo is None:
                                retry_at = retry_at.replace(tzinfo=datetime.timezone.utc)
                            server_wait = max(0.0, (retry_at - datetime.datetime.now(datetime.timezone.utc)).total_seconds())
                        except (TypeError, ValueError, OverflowError):
                            server_wait = 0.0
                wait = min(max(HTTP_RETRY_BASE_SECONDS * (2 ** (attempt - 1)), server_wait), max_wait)
                print(f"{description}: HTTP {response.status_code}; {wait:.0f}秒後に再試行 ({attempt}/{max_attempts})")
                response.close()
                time.sleep(wait)
                continue
            return response
        except requests.exceptions.RequestException as exc:
            if attempt >= max_attempts:
                raise
            wait = min(HTTP_RETRY_BASE_SECONDS * (2 ** (attempt - 1)), max_wait)
            print(f"{description}: 通信エラー ({exc}); {wait:.0f}秒後に再試行 ({attempt}/{max_attempts})")
            time.sleep(wait)
    raise RuntimeError(f"{description}: retry loop ended unexpectedly")

def download_file(url, path, description):
    """ファイルを一時ファイルにダウンロードしてから置換する。"""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return
    temp_path = path + ".download"
    response = get_with_retry(url, timeout=180, description=description)
    try:
        response.raise_for_status()
        with open(temp_path, "wb") as f:
            f.write(response.content)
        if os.path.getsize(temp_path) == 0:
            raise RuntimeError(f"ダウンロードファイルが空です: {path}")
        os.replace(temp_path, path)
    finally:
        response.close()
        if os.path.exists(temp_path):
            os.remove(temp_path)

def transform_lonlat_to_figure(lonlat, ax, proj):
    point_proj = proj.transform_point(*lonlat, ccrs.PlateCarree())
    point_pix = ax.transData.transform(point_proj)
    point_fig = ax.transAxes.inverted().transform(point_pix)
    return point_fig, point_pix, point_proj

def to_magnitude(data):
    """Pint Quantity・DataArray・通常配列をfloat ndarrayへ変換する。"""
    if hasattr(data, "metpy") and hasattr(data.metpy, "unit_array"):
        data = data.metpy.unit_array
    if hasattr(data, "magnitude"):
        data = data.magnitude
    elif hasattr(data, "values"):
        data = data.values
        if hasattr(data, "magnitude"):
            data = data.magnitude
    return np.asarray(data, dtype=float)

def smooth_quantity(data, sigma):
    """単位を保持したままSciPy Gaussian filterを適用する。"""
    raw = data.data
    if hasattr(raw, "magnitude"):
        filtered = ndimage.gaussian_filter(np.asarray(raw.magnitude, dtype=float), sigma=sigma)
        return filtered * raw.units
    return ndimage.gaussian_filter(np.asarray(raw, dtype=float), sigma=sigma)

def get_level_index(levels, target):
    """指定気圧面のインデックスを取得し、存在しなければ明示的に停止する。"""
    indices = np.where(np.asarray(levels) == target)[0]
    if len(indices) == 0:
        raise RuntimeError(f"必要な気圧面 {target} hPa がGRIBデータにありません。")
    return int(indices[0])

# ============================================================
# 極大・極小ピーク検出
# ============================================================
def detect_peaks(image, filter_size, dist_cut, flag=0):
    """flag=0で極大、flag=1で極小。戻り値は (行インデックス配列, 列インデックス配列)。"""
    image = to_magnitude(image)
    if image.ndim != 2:
        raise ValueError("detect_peaksには2次元配列を指定してください。")
    finite = np.isfinite(image)
    if not np.any(finite):
        return np.array([], dtype=int), np.array([], dtype=int)
    if flag == 0:
        work = np.where(finite, image, -np.inf)
        local_extreme = maximum_filter(work, footprint=np.ones((filter_size, filter_size), dtype=bool), mode="constant", cval=-np.inf)
    else:
        work = np.where(finite, image, np.inf)
        local_extreme = minimum_filter(work, footprint=np.ones((filter_size, filter_size), dtype=bool), mode="constant", cval=np.inf)
    peaks_mask = finite & (work == local_extreme)
    x, y = np.where(peaks_mask)
    if len(x) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    coords = np.column_stack((x, y))
    tree = cKDTree(coords)
    neighbors = tree.query_ball_point(coords, r=dist_cut)
    d_sum = np.zeros(len(x), dtype=float)
    neighbor_distances = []
    for i, indices in enumerate(neighbors):
        tmp = []
        for k in indices:
            if i == k:
                continue
            distance = np.linalg.norm(coords[i] - coords[k])
            if distance < dist_cut:
                tmp.append((k, distance))
                d_sum[i] += distance
        neighbor_distances.append(tmp)
    cut_point = np.zeros(len(x), dtype=bool)
    for i in range(len(x)):
        val_i = image[x[i], y[i]]
        for k, distance in neighbor_distances[i]:
            val_k = image[x[k], y[k]]
            if flag == 0 and val_i < val_k:
                cut_point[i] = True
                break
            if flag != 0 and val_i > val_k:
                cut_point[i] = True
                break
            if d_sum[i] > d_sum[k] or (d_sum[i] == d_sum[k] and i > k):
                cut_point[i] = True
                break
    return x[~cut_point], y[~cut_point]

# ============================================================
# 解析用ヘルパー（平滑化・微分・ゼロ等値線抽出）
# ============================================================
def smooth_nan(data, sigma):
    """NaNを含む配列を、NaNの影響を広げずにGaussian平滑化する（正規化畳み込み）。"""
    arr = to_magnitude(data)
    finite = np.isfinite(arr)
    weighted = ndimage.gaussian_filter(np.where(finite, arr, 0.0), sigma=sigma)
    weight = ndimage.gaussian_filter(finite.astype(float), sigma=sigma)
    return np.divide(weighted, weight, out=np.full_like(weighted, np.nan), where=weight > 0.5)

def horizontal_derivatives(field, lats, lons):
    """緯度経度格子上の水平微分（単位: 1/km）。緯度が降順でも動作する。"""
    lat_arr = np.asarray(lats, dtype=float)
    lon_arr = np.asarray(lons, dtype=float)
    coslat = np.cos(np.deg2rad(lat_arr))[:, None]
    fy = np.gradient(field, lat_arr, axis=0) / KM_PER_DEG
    fx = np.gradient(field, lon_arr, axis=1) / (KM_PER_DEG * coslat)
    return fx, fy

def coordinate_to_index(values, coords):
    """座標値を（小数）配列インデックスへ変換する。座標が降順でも動作する。"""
    coords = np.asarray(coords, dtype=float)
    idx = np.arange(len(coords), dtype=float)
    if coords[0] > coords[-1]:
        return np.interp(values, coords[::-1], idx[::-1])
    return np.interp(values, coords, idx)

def sample_field(field, lats, lons, line):
    rows = coordinate_to_index(line[:, 1], lats)
    cols = coordinate_to_index(line[:, 0], lons)
    return ndimage.map_coordinates(field, [rows, cols], order=1, mode="nearest")

def interior_mask(shape, margin):
    """領域端（平滑化の反射で偽の極値が出る帯）を除外するマスク。"""
    mask = np.zeros(shape, dtype=bool)
    margin = int(margin)
    if margin <= 0:
        mask[:, :] = True
    else:
        mask[margin:-margin, margin:-margin] = True
    return mask

def extract_zero_lines(lons, lats, field, valid):
    """validな格子だけを対象に field=0 の等値線を抽出し、(経度, 緯度)の折れ線リストで返す。"""
    field = np.asarray(field, dtype=float)
    valid = np.asarray(valid, dtype=bool) & np.isfinite(field)
    if int(valid.sum()) < 4:
        return []
    masked = np.ma.masked_where(~valid, np.where(valid, field, 0.0))
    fig_tmp = plt.figure()
    try:
        ax_tmp = fig_tmp.add_subplot(1, 1, 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cs = ax_tmp.contour(np.asarray(lons), np.asarray(lats), masked, levels=[0.0])
        lines = [np.array(seg, dtype=float) for seg in cs.allsegs[0] if len(seg) >= 2]
    finally:
        plt.close(fig_tmp)
    return lines

def polyline_length_km(line):
    if len(line) < 2:
        return 0.0
    dlon = np.diff(line[:, 0])
    dlat = np.diff(line[:, 1])
    mid_lat = np.deg2rad(0.5 * (line[1:, 1] + line[:-1, 1]))
    return float(np.sum(np.hypot(dlon * KM_PER_DEG * np.cos(mid_lat), dlat * KM_PER_DEG)))

def smooth_polyline(line, points=10):
    if len(line) < 5:
        return line
    smoothed = line.copy()
    smoothed[:, 0] = ndimage.gaussian_filter1d(line[:, 0], sigma=points, mode="nearest")
    smoothed[:, 1] = ndimage.gaussian_filter1d(line[:, 1], sigma=points, mode="nearest")
    return smoothed

# ============================================================
# 300 hPa強風軸の自動解析
#   風速場の「流れに直交する方向の尾根線」を抽出する。
#   流れに直交する単位ベクトル n=(-v,u)/|V| として
#     g = dS/dn = 0  かつ  d2S/dn2 < 0  かつ  S >= 閾値
#   を満たす等値線が強風軸。複数本の軸を検出できる。
# ============================================================
def detect_jet_axes(wind_speed, u_wind, v_wind, lats, lons):
    speed = smooth_nan(wind_speed, JET_SMOOTH_SIGMA)
    u = smooth_nan(u_wind, JET_SMOOTH_SIGMA)
    v = smooth_nan(v_wind, JET_SMOOTH_SIGMA)
    sx, sy = horizontal_derivatives(speed, lats, lons)
    sxx, sxy = horizontal_derivatives(sx, lats, lons)
    _, syy = horizontal_derivatives(sy, lats, lons)
    vmag = np.hypot(u, v)
    with np.errstate(all="ignore"):
        g = (-v * sx + u * sy) / vmag
        curv = (v * v * sxx - 2.0 * u * v * sxy + u * u * syy) / (vmag * vmag)
    valid = np.isfinite(g) & np.isfinite(curv) & (vmag > 1.0) & (curv < 0.0)
    valid &= interior_mask(speed.shape, 2 * JET_SMOOTH_SIGMA)
    axes = []
    for line in extract_zero_lines(lons, lats, g, valid):
        if len(line) < 10 or polyline_length_km(line) < JET_MIN_LENGTH_KM:
            continue
        line = smooth_polyline(line, points=6)
        speeds = np.nan_to_num(sample_field(speed, lats, lons, line), nan=0.0)
        u_line = np.nan_to_num(sample_field(u, lats, lons, line), nan=0.0)
        v_line = np.nan_to_num(sample_field(v, lats, lons, line), nan=0.0)
        axes.append((line, speeds, u_line, v_line))
    return axes

def jet_arrow_segments(line, u_line, v_line, interval_km=400.0, half_width=4):
    """強風軸に沿って一定間隔で矢印の(始点, 終点)を返す。向きは軸上の風向に合わせる。"""
    if len(line) < 2 * half_width + 2:
        return []
    mid_lat = np.deg2rad(0.5 * (line[1:, 1] + line[:-1, 1]))
    seg_km = np.hypot(np.diff(line[:, 0]) * KM_PER_DEG * np.cos(mid_lat), np.diff(line[:, 1]) * KM_PER_DEG)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_km)])
    total = cumulative[-1]
    positions = np.arange(interval_km / 2.0, total, interval_km)
    if len(positions) == 0:
        positions = np.array([total / 2.0])
    arrows = []
    n = len(line)
    for pos in positions:
        i = int(np.searchsorted(cumulative, pos))
        i1, i2 = max(0, i - half_width), min(n - 1, i + half_width)
        if i2 <= i1:
            continue
        east = (line[i2, 0] - line[i1, 0]) * np.cos(np.deg2rad(line[i, 1]))
        north = line[i2, 1] - line[i1, 1]
        if east * u_line[i] + north * v_line[i] < 0.0:
            i1, i2 = i2, i1
        arrows.append((line[i1].copy(), line[i2].copy()))
    return arrows

# ============================================================
# 500 hPaトラフ・リッジ軸の自動解析
#   MetPyの曲率渦度を利用し、曲率渦度が「流れに沿って」極大になる点を
#   結んだ線をトラフ軸、極小になる点を結んだ線をリッジ軸とする。
#   （流れに直交する方向で微分すると等高度線と平行な線になり、軸にならない）
# ============================================================
def detect_trough_ridge_axes(u_wind, v_wind, lats, lons):
    """曲率渦度を用いて500hPaトラフ・リッジ軸を抽出する。
    トラフ軸: 曲率渦度が正で、流れに沿って極大になる点を結んだ線
    リッジ軸: 曲率渦度が負で、流れに沿って極小になる点を結んだ線"""
    lon_coord = np.asarray(lons, dtype=float) * units.degrees
    lat_coord = np.asarray(lats, dtype=float) * units.degrees
    dx, dy = mpcalc.lat_lon_grid_deltas(lon_coord, lat_coord)
    curvature_vort = mpcalc.curvature_vorticity(
        np.asarray(u_wind, dtype=float) * units.knots,
        np.asarray(v_wind, dtype=float) * units.knots,
        dx=dx, dy=dy)
    # 調整パラメータは3つだけ（TROUGH_SMOOTH_SIGMA, TROUGH_MIN_ZETA_C, TROUGH_MAX_LENGTH_KM）。
    # 他の値はそこから決めるか、固定値にしている。
    direction_sigma = TROUGH_SMOOTH_SIGMA / 2.0                      # 流れの向きを決める風の平滑化
    edge_margin = TROUGH_SMOOTH_SIGMA                                # 領域端の除外
    cross_smooth_points = max(3, int(0.75 * TROUGH_SMOOTH_SIGMA))    # 交差角の線に沿った平滑化
    max_half_km = TROUGH_MAX_LENGTH_KM / 2.0                         # ピークから片側の最大距離
    min_length_km = TROUGH_MAX_LENGTH_KM / 4.0                       # 軸の最小長さ
    max_along_cos = 0.5       # 軸と流れが60度以上で交差する部分だけ残す
    trim_fraction = 0.75      # 曲率渦度がピークの75%以上の区間だけ残す
    curv_fraction = 0.25      # 流れ沿いの極値の鋭さの下限（領域内90%点に対する比）
    min_wind_kt = 10.0        # 流れの向きを定義できる最小風速

    zeta_c = smooth_nan(to_magnitude(curvature_vort) * 1.0e5, TROUGH_SMOOTH_SIGMA)

    # 流れの向きは平滑化した風から決める（生の風だと向きが乱れて線が千切れる）
    u = smooth_nan(u_wind, direction_sigma)
    v = smooth_nan(v_wind, direction_sigma)
    vmag = np.hypot(u, v)

    zx, zy = horizontal_derivatives(zeta_c, lats, lons)
    zxx, zxy = horizontal_derivatives(zx, lats, lons)
    _, zyy = horizontal_derivatives(zy, lats, lons)
    with np.errstate(all="ignore"):
        along_gradient = (u * zx + v * zy) / vmag                          # d(zeta_c)/ds
        along_curvature = (u * u * zxx + 2.0 * u * v * zxy + v * v * zyy) / (vmag * vmag)  # d2/ds2

    # 流れに沿った曲率渦度の変化が緩い所（ほぼ一様な帯）では軸が定まらないので除く。
    # 閾値は領域内の分布に対する相対値にして、風の強さによらず効くようにする。
    finite_curv = np.abs(along_curvature[np.isfinite(along_curvature) & interior_mask(zeta_c.shape, edge_margin)])
    curv_floor = curv_fraction * np.percentile(finite_curv, 90) if finite_curv.size else np.inf
    base = (np.isfinite(along_gradient) & np.isfinite(along_curvature) & np.isfinite(zeta_c)
            & (vmag >= min_wind_kt) & interior_mask(zeta_c.shape, edge_margin))
    trough_valid = base & (along_curvature <= -curv_floor) & (zeta_c >= TROUGH_MIN_ZETA_C)
    ridge_valid = base & (along_curvature >= curv_floor) & (zeta_c <= -TROUGH_MIN_ZETA_C)

    results = []
    for valid, sign in ((trough_valid, 1.0), (ridge_valid, -1.0)):
        axes = []
        for line in extract_zero_lines(lons, lats, along_gradient, valid):
            if len(line) < 10 or polyline_length_km(line) < min_length_km:
                continue
            line = smooth_polyline(line, points=10)
            # 軸は流れと交差するはず。流れと平行に近い「部分」を点ごとに除外する
            # （線全体で判定すると、C字に曲がった線の流れに沿った半分が残ってしまうため）。
            lu = sample_field(u, lats, lons, line)
            lv = sample_field(v, lats, lons, line)
            tx = np.gradient(line[:, 0]) * np.cos(np.deg2rad(line[:, 1]))
            ty = np.gradient(line[:, 1])
            with np.errstate(all="ignore"):
                cos_angle = np.abs(tx * lu + ty * lv) / (np.hypot(tx, ty) * np.hypot(lu, lv))
            cos_angle = ndimage.uniform_filter1d(np.nan_to_num(cos_angle, nan=1.0), size=cross_smooth_points, mode="nearest")
            crossing = cos_angle <= max_along_cos
            if not crossing.any():
                continue
            # 軸全体ではなく、曲率渦度が強い区間（トラフ・リッジの本体）だけを残す。
            strength = sign * np.nan_to_num(sample_field(zeta_c, lats, lons, line), nan=0.0)
            strength = np.where(crossing, strength, 0.0)
            peak = strength.max()
            if peak < TROUGH_MIN_ZETA_C:
                continue
            keep = np.where(strength >= max(TROUGH_MIN_ZETA_C, trim_fraction * peak))[0]
            if len(keep) == 0:
                continue
            for run in np.split(keep, np.where(np.diff(keep) > 1)[0] + 1):
                if len(run) < 10:
                    continue
                seg = line[run]
                # ピークから片側 max_half_km までに制限する
                seg_mid_lat = np.deg2rad(0.5 * (seg[1:, 1] + seg[:-1, 1]))
                seg_km = np.hypot(np.diff(seg[:, 0]) * KM_PER_DEG * np.cos(seg_mid_lat), np.diff(seg[:, 1]) * KM_PER_DEG)
                along_km = np.concatenate([[0.0], np.cumsum(seg_km)])
                peak_pos = along_km[int(np.argmax(strength[run]))]
                seg = seg[np.abs(along_km - peak_pos) <= max_half_km]
                if len(seg) >= 10 and polyline_length_km(seg) >= min_length_km:
                    axes.append(seg)
        results.append(axes)
    return results[0], results[1]

# トラフ: 茶色の二重線、リッジ: 青の波線。等温線（赤）と区別できる。
import matplotlib.patheffects as pe
def wavy_line(line, amplitude_km=35.0, wavelength_km=140.0, step_km=8.0):
    """折れ線を、進行方向に直交する向きへ正弦波状にずらした波線にして返す。"""
    mid_lat = np.deg2rad(0.5 * (line[1:, 1] + line[:-1, 1]))
    seg_km = np.hypot(np.diff(line[:, 0]) * KM_PER_DEG * np.cos(mid_lat), np.diff(line[:, 1]) * KM_PER_DEG)
    along = np.concatenate([[0.0], np.cumsum(seg_km)])
    if along[-1] < step_km * 3:
        return line
    s_grid = np.arange(0.0, along[-1], step_km)
    lon = np.interp(s_grid, along, line[:, 0])
    lat = np.interp(s_grid, along, line[:, 1])
    cos_lat = np.cos(np.deg2rad(lat))
    east_km = np.gradient(lon) * KM_PER_DEG * cos_lat
    north_km = np.gradient(lat) * KM_PER_DEG
    norm = np.hypot(east_km, north_km)
    norm[norm == 0.0] = 1.0
    normal_e, normal_n = -north_km / norm, east_km / norm
    # 端ではゼロから始まるように振幅を滑らかに立ち上げる
    taper = np.clip(np.minimum(s_grid, along[-1] - s_grid) / wavelength_km, 0.0, 1.0)
    offset = amplitude_km * taper * np.sin(2.0 * np.pi * s_grid / wavelength_km)
    return np.column_stack([lon + normal_e * offset / (KM_PER_DEG * cos_lat), lat + normal_n * offset / KM_PER_DEG])

def draw_trough_ridge(ax, trough_axes, ridge_axes, transform):
    trough_color, trough_width = "#7a3b00", 7.0
    for line in trough_axes:
        # 太い茶色の線の上に細い白線を重ねて二重線にする（外側に白い縁取り）
        ax.plot(line[:, 0], line[:, 1], color="white", linewidth=trough_width + 3.0, solid_capstyle="butt", transform=transform, zorder=8)
        ax.plot(line[:, 0], line[:, 1], color=trough_color, linewidth=trough_width, solid_capstyle="butt", transform=transform, zorder=8.1)
        ax.plot(line[:, 0], line[:, 1], color="white", linewidth=trough_width * 0.35, solid_capstyle="butt", transform=transform, zorder=8.2)
    for line in ridge_axes:
        wave = wavy_line(line)
        halo = [pe.Stroke(linewidth=5.5, foreground="white"), pe.Normal()]
        ax.plot(wave[:, 0], wave[:, 1], color="blue", linewidth=2.2, solid_capstyle="round", transform=transform, zorder=8, path_effects=halo)

# ============================================================
# 地上気圧: 中心気圧を保ったまま等圧線を滑らかにする
# ============================================================
def collect_extrema(raw, detect_field, flag, filter_size, dist_cut, refine_radius=5):
    """極値の位置と、生データ上の中心値を返す。 [(行, 列, 中心値), ...]"""
    rows, cols = detect_peaks(detect_field, filter_size, dist_cut, flag)
    ny, nx = raw.shape
    result = []
    for r, c in zip(rows, cols):
        r1, r2 = max(0, r - refine_radius), min(ny, r + refine_radius + 1)
        c1, c2 = max(0, c - refine_radius), min(nx, c + refine_radius + 1)
        window = raw[r1:r2, c1:c2]
        target = np.nanmin(window) if flag else np.nanmax(window)
        result.append((int(r), int(c), float(target)))
    return result

def smooth_keep_extrema(field, sigma, extrema, bump_sigma):
    """Gaussian平滑化で失われた中心値の差分を、極値の周囲にだけ滑らかに戻す。"""
    smooth = ndimage.gaussian_filter(field, sigma=sigma)
    out = smooth.copy()
    ny, nx = field.shape
    radius = int(3 * bump_sigma)
    for r0, c0, target in extrema:
        r1, r2 = max(0, r0 - radius), min(ny, r0 + radius + 1)
        c1, c2 = max(0, c0 - radius), min(nx, c0 + radius + 1)
        rr, cc = np.mgrid[r1:r2, c1:c2]
        weight = np.exp(-((rr - r0) ** 2 + (cc - c0) ** 2) / (2.0 * bump_sigma ** 2))
        out[r1:r2, c1:c2] += (target - out[r0, c0]) * weight
    return out

# ============================================================
# H/Lスタンプ（地上・等圧面で共通）
# ============================================================
def draw_extremum_stamp(ax, lon, lat, kind):
    color = "blue" if kind == "H" else "red"
    ax.text(lon, lat, kind, size=STAMP_FONT_SIZE, color=color, ha="center", va="center", transform=ccrs.PlateCarree(), zorder=10)

def draw_surface_center(ax, lon, lat, kind, value_text):
    """地上の高低気圧: 中心に×印、すぐ上にH/L（等圧面と同じ大きさ）、下に中心気圧。"""
    color = "blue" if kind == "H" else "red"
    transform = ccrs.PlateCarree()._as_mpl_transform(ax)
    ax.plot(lon, lat, marker="x", markersize=9, markeredgewidth=2.0, color=color, transform=ccrs.PlateCarree(), zorder=10)
    ax.annotate(kind, xy=(lon, lat), xycoords=transform, xytext=(0, 6), textcoords="offset points", color=color, size=STAMP_FONT_SIZE, ha="center", va="bottom", zorder=10)
    ax.annotate(value_text, xy=(lon, lat), xycoords=transform, xytext=(0, -8), textcoords="offset points", color=color, size=STAMP_FONT_SIZE // 2, ha="center", va="top", zorder=10)

# ============================================================
# 相当温位（バグ回避版）
#   MetPyのequivalent_potential_temperatureは内部でLCLを反復計算するため、
#   寒冷・乾燥域でNaNになり、850hPa EPT図に白抜けが出ていた。Bolton式で置き換え。
#   旧実装は dsp["level"]（1次元）を3次元の気温・露点と直接組み合わせていたため、
#   気圧面の軸と経度の軸が取り違えられる危険があった。
#   ここでは気圧を気温と同じ形状へ明示的にブロードキャストし、
#   有効な格子だけを1次元にして計算してから元の形へ戻す。
# ============================================================
def saturation_vapor_pressure_hpa(temperature_c):
    """飽和水蒸気圧(hPa)。Bolton(1980)。"""
    t = np.asarray(temperature_c, dtype=float)
    return 6.112 * np.exp(17.67 * t / (t + 243.5))

def equivalent_potential_temperature_k(pressure_hpa, temperature_c, dewpoint_c):
    """相当温位(K)。Bolton(1980)式(43)の閉形式。反復計算を使わないため、
    寒冷・乾燥な空気（上空や冬季の大陸上）でもNaNにならない。
    気圧は気温と同じ形状へブロードキャストする。"""
    t_c = np.asarray(temperature_c, dtype=float)
    td_c = np.minimum(np.asarray(dewpoint_c, dtype=float), t_c)
    p = np.broadcast_to(np.asarray(pressure_hpa, dtype=float), t_c.shape)
    out = np.full(t_c.shape, np.nan, dtype=float)
    valid = np.isfinite(t_c) & np.isfinite(td_c) & np.isfinite(p) & (p > 1.0)
    if not np.any(valid):
        return out
    t_k = t_c[valid] + 273.15
    td_k = td_c[valid] + 273.15
    pv = p[valid]
    e = np.minimum(saturation_vapor_pressure_hpa(td_c[valid]), 0.99 * pv)
    r = 622.0 * e / (pv - e)                                  # 混合比 g/kg
    t_lcl = 1.0 / (1.0 / (td_k - 56.0) + np.log(t_k / td_k) / 800.0) + 56.0
    theta_e = t_k * (1000.0 / (pv - e)) ** (0.2854 * (1.0 - 0.28e-3 * r)) * np.exp((3.376 / t_lcl - 0.00254) * r * (1.0 + 0.81e-3 * r))
    theta_e[~np.isfinite(theta_e)] = np.nan
    out[valid] = theta_e
    return out

def dewpoint_from_rh_c(temperature_c, rh_percent):
    """相対湿度(%)から露点温度(degC)を計算する。0%や範囲外の湿度でも停止しない。"""
    rh = np.array(rh_percent, dtype=float, copy=True)
    finite = rh[np.isfinite(rh)]
    if finite.size and np.nanmax(finite) <= 1.5 and np.nanmin(finite) >= 0.0:
        rh *= 100.0
    # 100%をわずかに超える値（雲域の過飽和）は欠測ではないので、NaNにせず100%に丸める。
    # NaNにすると、その格子の露点・相当温位が白抜けになる。明らかな異常値だけを欠測扱いにする。
    rh[(rh < -1.0) | (rh > 200.0)] = np.nan
    rh = np.where(np.isfinite(rh), np.clip(rh, 1.0, 100.0), np.nan)
    t = np.asarray(temperature_c, dtype=float)
    with np.errstate(all="ignore"):
        ln_ratio = np.log(rh / 100.0 * saturation_vapor_pressure_hpa(t) / 6.112)
        dew = 243.5 * ln_ratio / (17.67 - ln_ratio)
    dew[~np.isfinite(dew)] = np.nan
    return np.minimum(dew, t)

# ============================================================
# Wyoming Upper Air 新API
# ============================================================
WYOMING_TARGET_PRESSURES = (300, 400, 500, 700, 850, 925)
WYOMING_REQUEST_INTERVAL = 2.0
WYOMING_MAX_ATTEMPTS = 3
WYOMING_PRESSURE_TOLERANCE = 0.015
WYOMING_WIND_MS_TO_KNOTS = 1.9438444924406
WYOMING_API_URL = "https://weather.uwyo.edu/wsgi/sounding"

def _wyoming_value(row, name):
    try:
        value = float(row[name])
        return value if np.isfinite(value) else np.nan
    except (KeyError, TypeError, ValueError, OverflowError):
        return np.nan

def _wyoming_normalize_columns(frame):
    """Wyoming TEXT:CSVの列名を標準名へ変換する。"""
    rename = {}
    for original in frame.columns:
        name = str(original).replace("\ufeff", "").strip().lower()
        compact = re.sub(r"[^a-z0-9]+", "_", name).strip("_")
        if compact.startswith("pressure"):
            target = "pressure"
        elif compact.startswith("geopotential_height") or compact.startswith("height"):
            target = "height"
        elif compact.startswith("dew_point_temperature") or compact.startswith("dewpoint"):
            target = "dewpoint"
        elif compact.startswith("temperature"):
            target = "temperature"
        elif compact.startswith("wind_direction") or compact in ("direction", "wind_dir"):
            target = "direction"
        elif compact.startswith("wind_speed") or compact in ("speed", "wind_spd"):
            target = "speed"
        elif compact.startswith("relative_humidity") or compact in ("humidity", "rh"):
            target = "relative_humidity"
        elif compact.startswith("mixing_ratio"):
            target = "mixing_ratio"
        elif compact in ("latitude", "longitude", "time"):
            target = compact
        else:
            target = compact
        rename[original] = target
    return frame.rename(columns=rename)

def _parse_wyoming_csv_response(csv_text, requested_wmo, requested_time):
    """Wyoming新APIのTEXT:CSVを解析する。"""
    if not csv_text or not csv_text.strip():
        raise ValueError("Wyoming応答が空です")
    body = csv_text.lstrip("\ufeff \r\n")
    if body.startswith("<"):
        message = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body))[:240]
        raise ValueError(f"CSVではないHTML応答: {message}")
    try:
        frame = pd.read_csv(StringIO(csv_text), skipinitialspace=True, na_values=["", "-9999", "-999.0", "-99.99", "-999.00"])
    except Exception as exc:
        raise ValueError(f"CSV解析失敗: {type(exc).__name__}: {exc}") from exc
    frame.columns = [str(c).strip() for c in frame.columns]
    frame = _wyoming_normalize_columns(frame)
    required = ("pressure", "temperature", "dewpoint", "direction", "speed", "time")
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"CSV必須列がありません: {missing}; 応答列={list(frame.columns)}")
    numeric_columns = ("pressure", "height", "temperature", "dewpoint", "direction", "speed", "relative_humidity", "mixing_ratio", "latitude", "longitude")
    for name in numeric_columns:
        if name in frame.columns:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["pressure", "time"]).copy()
    frame = frame[np.isfinite(frame["pressure"]) & (frame["pressure"] > 0) & (frame["pressure"] <= 1100)]
    frame = frame.sort_values("pressure", ascending=False).drop_duplicates(subset=["pressure"], keep="first").reset_index(drop=True)
    if frame.empty:
        raise ValueError("CSVに有効な気圧データがありません")
    direction = np.deg2rad(frame["direction"].to_numpy(dtype=float))
    speed = frame["speed"].to_numpy(dtype=float)
    frame["u_wind"] = -np.sin(direction) * speed
    frame["v_wind"] = -np.cos(direction) * speed
    frame["station_number"] = int(requested_wmo)
    frame.attrs["units"] = {"pressure": "hPa", "height": "meter", "temperature": "degC", "dewpoint": "degC", "direction": "degrees", "speed": "m/s", "u_wind": "m/s", "v_wind": "m/s", "time": None, "station_number": None, "latitude": "degrees", "longitude": "degrees"}
    return frame

def _wyoming_observation_time(frame, requested_time):
    if frame is None or frame.empty or "time" not in frame.columns:
        return None
    value = pd.to_datetime(frame["time"].iloc[0], utc=True, errors="coerce")
    if pd.isna(value):
        return None
    return value.to_pydatetime().astimezone(datetime.timezone.utc)

def request_wyoming_wmo(wmo, request_dt):
    """Wyoming新APIへWMO番号とUTC日時を指定して1地点分を取得する。"""
    site_id = str(int(wmo)).zfill(5)
    if request_dt.tzinfo is None:
        request_dt = request_dt.replace(tzinfo=datetime.timezone.utc)
    request_dt = request_dt.astimezone(datetime.timezone.utc)
    params = {"datetime": request_dt.strftime("%Y-%m-%d %H:%M:%S"), "id": site_id, "type": "TEXT:CSV"}
    response = get_with_retry(WYOMING_API_URL, timeout=60, max_wait=120, description=f"Wyoming WMO {site_id} {request_dt:%Y%m%d%H%M}", max_attempts=WYOMING_MAX_ATTEMPTS, params=params)
    try:
        if response.status_code in (400, 404, 422):
            return None
        response.raise_for_status()
        body = response.text or ""
        if not body.strip() or "no data" in body.lower() or "not found" in body.lower():
            return None
        return _parse_wyoming_csv_response(body, site_id, request_dt)
    finally:
        response.close()

def fetch_wyoming_station(wmo, target_dt, station_row):
    """WMO番号で新APIを照会し、実観測時刻と6気圧面の値を検証する。"""
    site_id = str(int(wmo)).zfill(5)
    if target_dt.tzinfo is None:
        target_dt = target_dt.replace(tzinfo=datetime.timezone.utc)
    target_dt = target_dt.astimezone(datetime.timezone.utc)
    try:
        frame = request_wyoming_wmo(site_id, target_dt)
    except Exception as exc:
        return site_id, None, f"request_error:{type(exc).__name__}:{exc}"
    if frame is None or frame.empty:
        return site_id, None, "no_sounding"
    obs_time = _wyoming_observation_time(frame, target_dt)
    if obs_time is None:
        return site_id, None, "observation_time_missing"
    delta_minutes = abs((obs_time - target_dt).total_seconds()) / 60.0
    if delta_minutes > 90.0:
        return site_id, None, f"time_mismatch:{obs_time:%Y%m%d%H%M}"
    lat = _wyoming_value(frame.iloc[0], "latitude")
    lon = _wyoming_value(frame.iloc[0], "longitude")
    if not np.isfinite(lat):
        lat = float(station_row["lat"])
    if not np.isfinite(lon):
        lon = float(station_row["lon"])
    if not np.isfinite(lat) or not np.isfinite(lon):
        return site_id, None, "coordinate_missing"
    lon = ((lon + 180.0) % 360.0) - 180.0
    p = pd.to_numeric(frame["pressure"], errors="coerce").to_numpy(dtype=float)
    t = pd.to_numeric(frame["temperature"], errors="coerce").to_numpy(dtype=float)
    td = pd.to_numeric(frame["dewpoint"], errors="coerce").to_numpy(dtype=float)
    u = pd.to_numeric(frame["u_wind"], errors="coerce").to_numpy(dtype=float) * WYOMING_WIND_MS_TO_KNOTS
    v = pd.to_numeric(frame["v_wind"], errors="coerce").to_numpy(dtype=float) * WYOMING_WIND_MS_TO_KNOTS
    n = min(len(p), len(t), len(td), len(u), len(v))
    p, t, td, u, v = (x[:n] for x in (p, t, td, u, v))
    good = np.isfinite(p) & np.isfinite(t) & np.isfinite(td) & np.isfinite(u) & np.isfinite(v) & (p > 0) & (p <= 1100)
    if not np.any(good):
        return site_id, None, "no_valid_levels"
    p, t, td, u, v = (x[good] for x in (p, t, td, u, v))
    station_data = {}
    for target in WYOMING_TARGET_PRESSURES:
        idx = int(np.argmin(np.abs(p - target)))
        tolerance = max(4.0, target * WYOMING_PRESSURE_TOLERANCE)
        if abs(p[idx] - target) > tolerance:
            continue
        ttd = float(t[idx] - td[idx])
        if not np.isfinite(ttd) or ttd < -1.0 or ttd > 50.0:
            continue
        station_data[target] = (float(t[idx]), max(0.0, ttd), float(u[idx]), float(v[idx]))
    if not station_data:
        return site_id, None, "target_levels_missing"
    result = {"lon": lon, "lat": lat, "data": station_data, "obs_time": obs_time, "time_offset_minutes": (obs_time - target_dt).total_seconds() / 60.0, "wmo": site_id}
    return site_id, result, "success"

def fetch_wyoming_data_all(stations, target_dt):
    """対象領域のIGRA WMO高層局をWyoming新APIで地点別取得する。"""
    station_rows = stations.dropna(subset=["wmo", "lat", "lon"]).copy()
    station_rows["wmo"] = station_rows["wmo"].astype(int)
    station_rows = station_rows.drop_duplicates(subset=["wmo"]).sort_values("wmo")
    found = {}
    counts = {"success": 0, "no_sounding": 0, "time_mismatch": 0, "observation_time_missing": 0, "coordinate_missing": 0, "no_valid_levels": 0, "target_levels_missing": 0, "request_error": 0, "other": 0}
    total = len(station_rows)
    if target_dt.tzinfo is None:
        target_dt = target_dt.replace(tzinfo=datetime.timezone.utc)
    target_dt = target_dt.astimezone(datetime.timezone.utc)
    print(f"Wyoming Upper Air 新API: WMO高層観測局 {total}地点を個別取得（{target_dt:%Y-%m-%d %H:%M} UTC）")
    for index, (_, row) in enumerate(station_rows.iterrows(), start=1):
        wmo, item, status = fetch_wyoming_station(row["wmo"], target_dt, row)
        if item is not None:
            found[wmo] = item
            counts["success"] += 1
            print(f"[{index}/{total}] {wmo}: 取得成功（{len(item['data'])}/6気圧面、観測時刻 {item['obs_time']:%m-%d %H:%M} UTC）")
        else:
            key = status.split(":", 1)[0]
            counts[key if key in counts else "other"] += 1
            print(f"[{index}/{total}] {wmo}: {status}")
        if index < total:
            time.sleep(WYOMING_REQUEST_INTERVAL)
    print("Wyoming取得集計: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    print(f"Wyoming最終取得: {len(found)}/{total}地点（要求時刻 {target_dt:%Y-%m-%d %H:%M} UTC）")
    return found

# ============================================================
# 基準時刻
# ============================================================
# 定刻から約2時間後に実行する運用を前提とし、サイクル経過時間による停止・除外は行わない。
now_utc = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=90)
vt = now_utc.replace(hour=(now_utc.hour // 12) * 12, minute=0, second=0, microsecond=0)
dt = vt - datetime.timedelta(hours=12)
i_year, i_month, i_day, i_hourZ = dt.year, dt.month, dt.day, dt.hour
print("IT(UTC):", dt)
fig_fld = "{0:4d}{1:02d}{2:02d}{3:02d}".format(i_year, i_month, i_day, i_hourZ)
os.makedirs(fig_fld, exist_ok=True)
i_area = [115, 155, 20, 50]
str_area = "jp"
latS, latN, lonW, lonE = 10, 60, 105, 165

# ============================================================
# ECMWF 0.1度格子データのダウンロード
# ============================================================
data_fld = "https://storage.googleapis.com/ecmwf-open-data/{0:04d}{1:02d}{2:02d}/{3:02d}z/ifs/0p1/oper/"
dat_fld = data_fld.format(i_year, i_month, i_day, i_hourZ)
fname_pl = "{0:4d}{1:02d}{2:02d}{3:02d}0000-12h-oper-fc-pl.grib2".format(i_year, i_month, i_day, i_hourZ)
fname_sfc = "{0:4d}{1:02d}{2:02d}{3:02d}0000-12h-oper-fc-sfc.grib2".format(i_year, i_month, i_day, i_hourZ)
download_file(dat_fld + fname_pl, fname_pl, f"ECMWF {fname_pl}")
download_file(dat_fld + fname_sfc, fname_sfc, f"ECMWF {fname_sfc}")

# ============================================================
# GRIB2読み込み
# ============================================================
grbs_pl = None
grbs_sfc = None
try:
    grbs_pl = pygrib.open(fname_pl)
    grbs_sfc = pygrib.open(fname_sfc)

    def select_level_messages(grbs, short_name):
        selected = grbs.select(shortName=short_name, typeOfLevel="isobaricInhPa")
        result = {}
        for message in selected:
            level = int(message.level)
            if level >= 300:
                if level in result:
                    raise RuntimeError(f"GRIBに同一気圧面の重複があります: {short_name} {level} hPa")
                result[level] = message
        return result

    grb_ht = select_level_messages(grbs_pl, "gh")
    grb_tm = select_level_messages(grbs_pl, "t")
    grb_wu = select_level_messages(grbs_pl, "u")
    grb_wv = select_level_messages(grbs_pl, "v")
    grb_rh = select_level_messages(grbs_pl, "r")
    grb_omg = select_level_messages(grbs_pl, "w")
    levels = np.array(sorted(grb_tm.keys()), dtype=int)
    required_levels = set(WYOMING_TARGET_PRESSURES)
    missing_levels = sorted(required_levels - set(levels.tolist()))
    if missing_levels:
        raise RuntimeError(f"ECMWF GRIBに必要な気圧面がありません: {missing_levels}")
    for name, messages in (("gh", grb_ht), ("u", grb_wu), ("v", grb_wv), ("r", grb_rh), ("w", grb_omg)):
        missing = sorted(set(levels.tolist()) - set(messages.keys()))
        if missing:
            raise RuntimeError(f"GRIBの{name}データに気圧面がありません: {missing}")
    first_message = grb_tm[int(levels[0])]
    val_first, lat_ht, lon_ht = first_message.data(lat1=latS, lat2=latN, lon1=lonW, lon2=lonE)
    lat_size, lon_size = val_first.shape
    lats, lons = lat_ht[:, 0], lon_ht[0, :]
    l_size = len(levels)
    ary_ht = np.empty((l_size, lat_size, lon_size), dtype=np.float32)
    ary_tm = np.empty_like(ary_ht)
    ary_rh = np.empty_like(ary_ht)
    ary_wu = np.empty_like(ary_ht)
    ary_wv = np.empty_like(ary_ht)
    ary_omg = np.empty_like(ary_ht)
    for l, level in enumerate(levels):
        for messages, array, label in ((grb_ht, ary_ht, "height"), (grb_tm, ary_tm, "temperature"), (grb_wu, ary_wu, "u_wind"), (grb_wv, ary_wv, "v_wind"), (grb_rh, ary_rh, "relative_humidity"), (grb_omg, ary_omg, "omega")):
            values, _, _ = messages[int(level)].data(lat1=latS, lat2=latN, lon1=lonW, lon2=lonE)
            if values.shape != (lat_size, lon_size):
                raise RuntimeError(f"{label} {level} hPaの格子形状が一致しません: {values.shape}")
            array[l] = values
    slp_messages = grbs_sfc.select(name="Mean sea level pressure")
    surface_messages = grbs_sfc.select(name="Surface pressure")
    if not slp_messages:
        raise RuntimeError("GRIBにMean sea level pressureがありません。")
    if not surface_messages:
        raise RuntimeError("GRIBにSurface pressureがありません。")
    val_slp, _, _ = slp_messages[0].data(lat1=latS, lat2=latN, lon1=lonW, lon2=lonE)
    val_hgt, _, _ = surface_messages[0].data(lat1=latS, lat2=latN, lon1=lonW, lon2=lonE)
    ary_slp, ary_hgt = val_slp, val_hgt
finally:
    if grbs_pl is not None:
        grbs_pl.close()
    if grbs_sfc is not None:
        grbs_sfc.close()

# ============================================================
# xarray / MetPy Dataset
# ============================================================
ds = xr.Dataset({
    "Geopotential_height": (["level", "lat", "lon"], ary_ht * units.meter),
    "temperature": (["level", "lat", "lon"], ary_tm * units("K")),
    "relative_humidity": (["level", "lat", "lon"], ary_rh * units("%")),
    "u_wind": (["level", "lat", "lon"], ary_wu * units("m/s")),
    "v_wind": (["level", "lat", "lon"], ary_wv * units("m/s")),
    "omega": (["level", "lat", "lon"], ary_omg * units("Pa/s")),
    "mslp": (["lat", "lon"], ary_slp / 100.0 * units("hPa")),
    "height": (["lat", "lon"], ary_hgt / 100.0 * units("hPa"))
}, coords={"level": levels, "lat": lats, "lon": lons})
ds["Geopotential_height"].attrs["units"] = "m"
ds["u_wind"].attrs["units"] = "m/s"
ds["v_wind"].attrs["units"] = "m/s"
ds["omega"].attrs["units"] = "Pa/s"
ds["level"].attrs["units"] = "hPa"
ds["lat"].attrs["units"] = "degrees_north"
ds["lon"].attrs["units"] = "degrees_east"
dsp = ds.metpy.parse_cf()
dsp["wind_speed"] = mpcalc.wind_speed(dsp["u_wind"], dsp["v_wind"])
dsp["conv"] = mpcalc.divergence(dsp["u_wind"], dsp["v_wind"])
dsp["vorticity"] = mpcalc.vorticity(dsp["u_wind"], dsp["v_wind"])
dsp["u_wind"] = dsp["u_wind"].metpy.convert_units("knots")
dsp["v_wind"] = dsp["v_wind"].metpy.convert_units("knots")
dsp["wind_speed"] = dsp["wind_speed"].metpy.convert_units("knots")
dsp["temperature"] = dsp["temperature"].metpy.convert_units(units.degC)

# ============================================================
# 相対湿度・露点温度・相当温位（NumPy配列で計算し、形状の取り違えを防ぐ）
# ============================================================
temperature_c_all = np.asarray(ary_tm, dtype=float) - 273.15
dewpoint_c_all = dewpoint_from_rh_c(temperature_c_all, ary_rh)
ttd_all = temperature_c_all - dewpoint_c_all
pressure_3d = np.asarray(levels, dtype=float)[:, None, None]
ept_all = equivalent_potential_temperature_k(pressure_3d, temperature_c_all, dewpoint_c_all)

# ============================================================
# 各気圧面のインデックス・相当温位指標
# ============================================================
idx_300 = get_level_index(levels, 300)
idx_400 = get_level_index(levels, 400)
idx_500 = get_level_index(levels, 500)
idx_700 = get_level_index(levels, 700)
idx_850 = get_level_index(levels, 850)
idx_925 = get_level_index(levels, 925)
theta_e_925 = equivalent_potential_temperature_k(925.0, temperature_c_all[idx_925], dewpoint_c_all[idx_925])
theta_e_700 = equivalent_potential_temperature_k(700.0, temperature_c_all[idx_700], dewpoint_c_all[idx_700])
theta_es_700 = equivalent_potential_temperature_k(700.0, temperature_c_all[idx_700], temperature_c_all[idx_700])
theta_es_500 = equivalent_potential_temperature_k(500.0, temperature_c_all[idx_500], temperature_c_all[idx_500])
new_index_values = theta_e_925 - theta_es_500 - (theta_es_700 - theta_e_700)

# ============================================================
# 平滑化
#   相当温位はNaN対応の平滑化を使用。地上気圧は描画側で
#   「中心気圧を保つ平滑化」を行うためここでは平滑化しない。
# ============================================================
ept_all = smooth_nan(ept_all, sigma=(0, 1, 1))
dsp["Geopotential_height"].data = smooth_quantity(dsp["Geopotential_height"], sigma=(0, 1, 1))
dsp["temperature"].data = smooth_quantity(dsp["temperature"], sigma=(0, 4, 4))
dsp["omega"] = dsp["omega"].metpy.convert_units("hPa/hour")

# ============================================================
# IGRA2地点リスト
# ============================================================
url = "https://www.ncei.noaa.gov/pub/data/igra/igra2-station-list.txt"
res = get_with_retry(url, timeout=120, description="NOAA IGRA station list")
try:
    res.raise_for_status()
    colspecs = [(0, 11), (12, 20), (21, 30), (31, 37), (38, 40), (41, 71), (72, 76), (77, 81), (82, 88)]
    names = ["station_id", "lat", "lon", "elev", "state", "name", "first_year", "last_year", "nobs"]
    df = pd.read_fwf(StringIO(res.text), colspecs=colspecs, names=names)
finally:
    res.close()

for col in ("lat", "lon", "elev", "first_year", "last_year", "nobs"):
    df[col] = pd.to_numeric(df[col], errors="coerce")
df["station_id"] = df["station_id"].astype(str).str.strip()
df["network"] = df["station_id"].str[2:3]
df["wmo"] = pd.to_numeric(df["station_id"].str[6:11], errors="coerce")
IGRA_RECENT_YEARS = 2
min_last_year = vt.year - IGRA_RECENT_YEARS
df_sel = df[(df["station_id"].str[2:6] == "M000") & (df["wmo"].notna()) & (df["nobs"].fillna(0) > 0) & (df["last_year"] >= min_last_year) & (df["lon"].between(i_area[0], i_area[1])) & (df["lat"].between(i_area[2], i_area[3]))].copy()
df_sel["wmo"] = df_sel["wmo"].astype(int)
df_sel = df_sel.sort_values(["last_year", "nobs"], ascending=False).drop_duplicates(subset=["wmo"]).sort_values("wmo").reset_index(drop=True)
print(f"IGRA地点選別: 全{len(df)}地点 → WMO高層局・直近{IGRA_RECENT_YEARS}年・対象領域 {len(df_sel)}地点")
if df_sel.empty:
    raise RuntimeError("対象領域に該当する直近のWMO高層観測局がありません。IGRA地点リストの列位置/対象期間を確認してください。")

# ============================================================
# 描画共通設定
# ============================================================
dt_str = (vt.strftime("%Y%m%d%HUTC")).upper()
dt_str2 = vt.strftime("%Y%m%d%H")
dlon, dlat = 10, 10
proj = ccrs.PlateCarree()
output_dir = os.path.join("Data", dt_str)
os.makedirs(output_dir, exist_ok=True)
lon_values, lat_values = dsp["lon"].values, dsp["lat"].values

def save_figure(fig, filename):
    """画像保存後にFigureを閉じ、メモリを解放する。"""
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"output:{filename}")
    plt.close(fig)

def add_gridlines(ax):
    xticks, yticks = np.arange(0, 360.1, dlon), np.arange(-90, 90.1, dlat)
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, alpha=0.8)
    gl.xlocator = mticker.FixedLocator(xticks)
    gl.ylocator = mticker.FixedLocator(yticks)

def add_coastlines(ax, linewidth=1.0):
    ax.coastlines(resolution="50m", linewidth=linewidth)

# ============================================================
# 850 hPa EPT図
# ============================================================
fig = plt.figure(figsize=(10, 8))
fig.subplots_adjust(left=0, right=1, bottom=0.06, top=0.98)
ax = fig.add_subplot(1, 1, 1, projection=proj)
ax.set_extent(i_area, crs=proj)
ept850 = ept_all[idx_850, :, :]
cnf_ept = ax.contourf(lon_values, lat_values, ept850, np.arange(255, 375, 3), cmap="jet", extend="both", transform=proj)
cn_ept0 = ax.contour(lon_values, lat_values, ept850, colors="black", linewidths=0.3, levels=np.arange(255, 375, 3), transform=proj)
cn_ept1 = ax.contour(lon_values, lat_values, ept850, colors="black", linewidths=1.0, levels=np.arange(255, 375, 15), transform=proj)
ax.clabel(cn_ept1, fontsize=12, inline=True, inline_spacing=5, fmt="%i", rightside_up=True, colors="black")
add_coastlines(ax, linewidth=1.6)
add_gridlines(ax)
wind_slice = (slice(None, None, 10), slice(None, None, 10))
u850 = to_magnitude(dsp["u_wind"][idx_850, wind_slice[0], wind_slice[1]])
v850 = to_magnitude(dsp["v_wind"][idx_850, wind_slice[0], wind_slice[1]])
ax.barbs(lon_values[wind_slice[1]], lat_values[wind_slice[0]], u850, v850, length=5.5, pivot="middle", color="black", transform=proj)
fig.text(0.5, 0.01, dt_str + " 850hPa EPT(K), Wind", ha="center", va="bottom", size=20)
save_figure(fig, "{0}_850ept.png".format(dt_str))

# ============================================================
# 地上天気図
#   等圧線は滑らかにしつつ、高低気圧の中心気圧は元データの値を保持する。
# ============================================================
fig = plt.figure(figsize=(10, 8))
fig.subplots_adjust(left=0, right=1, bottom=0.06, top=0.98)
ax = fig.add_subplot(1, 1, 1, projection=proj)
ax.set_extent(i_area, crs=proj)
mslp_raw = to_magnitude(dsp["mslp"])
mslp_detect = ndimage.gaussian_filter(mslp_raw, sigma=2.0)  # 極値位置の検出用（弱い平滑化）
slp_highs = collect_extrema(mslp_raw, mslp_detect, 0, filter_size=50, dist_cut=25.0)
slp_lows = collect_extrema(mslp_raw, mslp_detect, 1, filter_size=50, dist_cut=25.0)
mslp_values = smooth_keep_extrema(mslp_raw, SLP_SMOOTH_SIGMA, slp_highs + slp_lows, 2 * SLP_SMOOTH_SIGMA)
clevs_mslp = np.arange(800, 1200, 4)
ax.contour(lon_values, lat_values, mslp_values, clevs_mslp, colors="black", linestyles="solid", linewidths=[1.25, 0.75, 0.75, 0.75, 0.75], transform=proj)
ax.contourf(lon_values, lat_values, new_index_values, [-3, 0], colors=["yellow", "red"], extend="max", transform=proj, alpha=0.5)

# 地上気圧の極大・極小（スタンプの大きさは等圧面と共通）
for extrema, kind in ((slp_highs, "H"), (slp_lows, "L")):
    for r, c, value in extrema:
        wlon, wlat = lon_values[c], lat_values[r]
        fig_z, _, _ = transform_lonlat_to_figure((wlon, wlat), ax, proj)
        if 0.05 < fig_z[0] < 0.95 and 0.05 < fig_z[1] < 0.95:
            draw_surface_center(ax, wlon, wlat, kind, str(int(round(value))))
add_coastlines(ax, linewidth=1.6)
add_gridlines(ax)
fig.text(0.5, 0.01, dt_str + " Sea Level Pressure(hPa)", ha="center", va="bottom", size=20)
save_figure(fig, "{0}_surf.png".format(dt_str))

# ============================================================
# 高層気象観測データ取得
# ============================================================
all_data = fetch_wyoming_data_all(df_sel, vt)
fetch_status = {
    "wyoming_success": len(all_data),
    "target_stations": int(df_sel["wmo"].dropna().astype(int).nunique()),
    "missing_station": max(0, int(df_sel["wmo"].dropna().astype(int).nunique()) - len(all_data)),
    "complete_six_levels": sum(1 for item in all_data.values() if len(item["data"]) == len(WYOMING_TARGET_PRESSURES)),
    "partial_levels": sum(1 for item in all_data.values() if len(item["data"]) < len(WYOMING_TARGET_PRESSURES))
}
print(f"高層観測データ取得結果（Wyoming個別取得・基準時刻 {vt:%Y-%m-%d %H:%M} UTC）:")
for status, count in fetch_status.items():
    print(f"  {status}: {count}")
if not all_data:
    raise RuntimeError(f"Wyomingから基準時刻 {vt:%Y-%m-%d %H:%M} UTC の高層観測を取得できませんでした。")

# ============================================================
# 各気圧面の高層天気図
# ============================================================
level_index_map = {300: idx_300, 400: idx_400, 500: idx_500, 700: idx_700, 850: idx_850, 925: idx_925}
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
    fig.subplots_adjust(left=0, right=1, bottom=0.06, top=0.98)
    ax = fig.add_subplot(1, 1, 1, projection=proj)
    if tagHp in [850, 925]:
        plot_extent = LOW_LEVEL_EXTENT
    else:
        plot_extent = i_area
    ax.set_extent(plot_extent, crs=proj)
    if tagHp in [300, 400]:
        step1, step2 = 120, 600
    else:
        step1, step2 = 60, 300
    idx = level_index_map[tagHp]
    height_values = to_magnitude(dsp["Geopotential_height"][idx, :, :])
    temperature_values = to_magnitude(dsp["temperature"][idx, :, :])
    ttd_values = ttd_all[idx, :, :]
    wind_speed_values = to_magnitude(dsp["wind_speed"][idx, :, :])
    vorticity_values = to_magnitude(dsp["vorticity"][idx, :, :])
    omega_values = to_magnitude(dsp["omega"][idx, :, :])

    # 高度
    cn_hgt1 = ax.contour(lon_values, lat_values, height_values, colors="black", linewidths=1.5, levels=np.arange(0, 12000, step1), transform=proj)
    ax.clabel(cn_hgt1, fontsize=18, inline=True, colors="black", inline_spacing=5, fmt="%i", rightside_up=True)
    cn_hgt2 = ax.contour(lon_values, lat_values, height_values, colors="black", linewidths=1.5, levels=np.arange(0, 12000, step2), transform=proj)
    ax.clabel(cn_hgt2, fontsize=15, inline=True, colors="black", inline_spacing=0, fmt="%i", rightside_up=True)

    # 気温・露点差
    ax.contourf(lon_values, lat_values, ttd_values, [3, 18], colors=["green", "1.0", "yellow"], extend="both", transform=proj, alpha=0.5)
    cn_tmp0 = ax.contour(lon_values, lat_values, temperature_values, colors="red", linewidths=1.0, linestyles="solid", levels=np.arange(-60, 42, 3), transform=proj)
    ax.clabel(cn_tmp0, fontsize=12, inline=True, inline_spacing=5, fmt="%i", rightside_up=True, colors="red")
    cn_tmp1 = ax.contour(lon_values, lat_values, temperature_values, colors="red", linewidths=2.0, linestyles="solid", levels=np.arange(-60, 42, 6), transform=proj)

    # 300 hPa強風軸（自動解析）
    if tagHp == 300:
        ax.contourf(lon_values, lat_values, wind_speed_values, [60, 80, 100], colors=["cyan", "pink", "magenta"], extend="max", transform=proj, alpha=0.5)
        u300 = to_magnitude(dsp["u_wind"][idx, :, :])
        v300 = to_magnitude(dsp["v_wind"][idx, :, :])

        # 300hPa天気図の表示範囲外を強風軸解析から除外
        jet_mask = ((lon_values[None, :] >= plot_extent[0]) & (lon_values[None, :] <= plot_extent[1]) & (lat_values[:, None] >= plot_extent[2]) & (lat_values[:, None] <= plot_extent[3]))
        jet_speed_in = np.where(jet_mask, wind_speed_values, np.nan)
        jet_u_in = np.where(jet_mask, u300, np.nan)
        jet_v_in = np.where(jet_mask, v300, np.nan)

        jet_axes = detect_jet_axes(jet_speed_in, jet_u_in, jet_v_in, lat_values, lon_values)
        arrow_transform = ccrs.PlateCarree()._as_mpl_transform(ax)
        for jet_line, jet_speeds, jet_u_line, jet_v_line in jet_axes:
            ax.plot(jet_line[:, 0], jet_line[:, 1], color="blue", linewidth=2.2, transform=proj, zorder=8)
            for start, end in jet_arrow_segments(jet_line, jet_u_line, jet_v_line):
                ax.annotate("", xy=(end[0], end[1]), xytext=(start[0], start[1]), xycoords=arrow_transform, textcoords=arrow_transform, arrowprops=dict(arrowstyle="-|>", color="blue", lw=2.2, mutation_scale=22, shrinkA=0, shrinkB=0), zorder=9)
            core_idx, _ = find_peaks(jet_speeds, prominence=5.0, distance=80)
            if len(core_idx):
                ax.scatter(jet_line[core_idx, 0], jet_line[core_idx, 1], color="blue", marker="o", s=30, edgecolors="white", linewidths=0.6, transform=proj, zorder=9)
        print(f"300hPa強風軸: {len(jet_axes)}本")

    # 500 hPaトラフ・リッジ（自動解析）
    elif tagHp == 500:
        smooth_vorticity = ndimage.gaussian_filter(vorticity_values * 1000000.0, sigma=1)
        ax.contourf(lon_values, lat_values, smooth_vorticity, np.arange(0, 1000, 6), cmap="Oranges", extend="max", transform=proj, alpha=0.5)
        u500 = to_magnitude(dsp["u_wind"][idx, :, :])
        v500 = to_magnitude(dsp["v_wind"][idx, :, :])
        trough_axes, ridge_axes = detect_trough_ridge_axes(u500, v500, lat_values, lon_values)
        draw_trough_ridge(ax, trough_axes, ridge_axes, proj)
        print(f"500hPaトラフ軸: {len(trough_axes)}本, リッジ軸: {len(ridge_axes)}本")

    # 700 hPa鉛直流（広い範囲で平滑化）
    elif tagHp == 700:
        smooth_omega = smooth_nan(omega_values, OMEGA_SMOOTH_SIGMA)
        ax.contourf(lon_values, lat_values, smooth_omega, np.arange(0, 100, 6), cmap="Reds", extend="max", transform=proj, alpha=0.5)

    # 気温の極大
    maxid = detect_peaks(temperature_values, filter_size=25, dist_cut=10.0)
    for j in range(len(maxid[0])):
        wlon, wlat = lon_values[maxid[1][j]], lat_values[maxid[0][j]]
        fig_z, _, _ = transform_lonlat_to_figure((wlon, wlat), ax, proj)
        if 0.0 < fig_z[0] < 1.0 and 0.0 < fig_z[1] < 1.0:
            ax.text(wlon, wlat, "W", size=12, color="red", ha="center", va="center", transform=proj)

    # 気温の極小
    minid = detect_peaks(temperature_values, filter_size=25, dist_cut=10.0, flag=1)
    for j in range(len(minid[0])):
        wlon, wlat = lon_values[minid[1][j]], lat_values[minid[0][j]]
        fig_z, _, _ = transform_lonlat_to_figure((wlon, wlat), ax, proj)
        if 0.0 < fig_z[0] < 1.0 and 0.0 < fig_z[1] < 1.0:
            ax.text(wlon, wlat, "C", size=12, color="blue", ha="center", va="center", transform=proj)

    # 高度の極大
    maxid = detect_peaks(height_values, filter_size=50, dist_cut=20.0)
    for j in range(len(maxid[0])):
        wlon, wlat = lon_values[maxid[1][j]], lat_values[maxid[0][j]]
        fig_z, _, _ = transform_lonlat_to_figure((wlon, wlat), ax, proj)
        if 0.0 < fig_z[0] < 1.0 and 0.0 < fig_z[1] < 1.0:
            draw_extremum_stamp(ax, wlon, wlat, "H")

    # 高度の極小
    minid = detect_peaks(height_values, filter_size=50, dist_cut=20.0, flag=1)
    for j in range(len(minid[0])):
        wlon, wlat = lon_values[minid[1][j]], lat_values[minid[0][j]]
        fig_z, _, _ = transform_lonlat_to_figure((wlon, wlat), ax, proj)
        if 0.0 < fig_z[0] < 1.0 and 0.0 < fig_z[1] < 1.0:
            draw_extremum_stamp(ax, wlon, wlat, "L")

    # 海岸線・緯度経度線
    add_coastlines(ax)
    add_gridlines(ax)

    # Wyoming観測点データ
    temps_arr, ttds_arr = np.asarray(temps, dtype=float), np.asarray(ttds, dtype=float)
    u_winds_arr, v_winds_arr = np.asarray(u_winds, dtype=float), np.asarray(v_winds, dtype=float)
    lons_arr, lats_arr = np.asarray(lons, dtype=float), np.asarray(lats, dtype=float)
    # 表示範囲外の観測点は描かない（範囲外の数字が図の外に出て、保存範囲が広がるのを防ぐ）
    in_extent = (lons_arr >= plot_extent[0]) & (lons_arr <= plot_extent[1]) & (lats_arr >= plot_extent[2]) & (lats_arr <= plot_extent[3])
    mask = in_extent & np.isfinite(lons_arr) & np.isfinite(lats_arr) & np.isfinite(temps_arr) & np.isfinite(ttds_arr) & np.isfinite(u_winds_arr) & np.isfinite(v_winds_arr)
    lons_p, lats_p = lons_arr[mask], lats_arr[mask]
    temps_p, ttds_p = temps_arr[mask], ttds_arr[mask]
    u_winds_p, v_winds_p = u_winds_arr[mask], v_winds_arr[mask]

    # StationPlotは維持し、風向・風速のバーブを描画する。
    # plot_parameter()は使用せず、気温・T-Tdはannotateで描画する。
    if len(lons_p) > 0:
        stationplot = StationPlot(ax, lons_p, lats_p, transform=ccrs.PlateCarree(), fontsize=12)
        stationplot.plot_barb(u_winds_p, v_winds_p, length=5)
        for lon, lat, temp, ttd in zip(lons_p, lats_p, temps_p, ttds_p):
            ax.annotate(f"{temp:.0f}", xy=(lon, lat), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), xytext=(-10, 9), textcoords="offset points", color="red", fontsize=12, ha="right", va="center", zorder=10)
            ax.annotate(f"{ttd:.0f}", xy=(lon, lat), xycoords=ccrs.PlateCarree()._as_mpl_transform(ax), xytext=(-10, -9), textcoords="offset points", color="blue", fontsize=12, ha="right", va="center", zorder=10)

    fig.text(0.5, 0.01, dt_str + " {0}hPa Tmp, T-Td".format(int(tagHp)), ha="center", va="bottom", size=20)
    out_fn = "{0}_{1:03d}ttd.png".format(dt_str, tagHp)
    save_figure(fig, out_fn)

# ============================================================
# 終了処理
# ============================================================
HTTP_SESSION.close()
print("全ての天気図の作成が完了しました。")
print(f"出力先: {output_dir}")
