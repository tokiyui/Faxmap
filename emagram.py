import datetime
import numpy as np
import matplotlib.pyplot as plt

from metpy.calc import (potential_temperature, equivalent_potential_temperature,
                        saturation_equivalent_potential_temperature, lfc, el,
                        k_index, showalter_index, total_totals_index)
from metpy.units import units
from siphon.simplewebservice.wyoming import WyomingUpperAir
from scipy.interpolate import interp1d
import os

# === 観測地点辞書（番号: 名称） ===
stations = {
    '47401': 'Wakkanai',
    '47412': 'Sapporo',
    '47418': 'Kushiro',
    '47582': 'Akita',
    '47600': 'Wajima',
    '47646': 'Tateno',
    '47678': 'Hachijojima',
    '47741': 'Matsue',
    '47778': 'Shionomisaki',
    '47807': 'Fukuoka',
    '47827': 'Kagoshima',
    '47909': 'Naze',
    '47918': 'Ishigakijima',
    '47945': 'Minamidaitojima',
    '47971': 'Chichijima',
    '47991': 'Minamitorishima',
}

def get_nearest_synoptic_time():
    now = datetime.datetime.now(datetime.UTC) - datetime.timedelta(minutes=80)
    hour = 0 if now.hour < 12 else 12
    return datetime.datetime(now.year, now.month, now.day, hour)

def plot_sounding(station_id, station_name, dt):
    print(f"Processing station {station_id} {station_name} at {dt}")

    try:
        df = WyomingUpperAir.request_data(dt, station_id)
        if df.empty:
            print(f"No data for station {station_id} {station_name} at {dt}, skipping.")
            return
    except Exception as e:
        print(f"Failed to get data for station {station_id} {station_name} at {dt}: {e}")
        return

    p = df['pressure'].values * units.hPa
    T = df['temperature'].values * units.degC
    Td = df['dewpoint'].values * units.degC
    z = df['height'].values * units.meter

    theta = potential_temperature(p, T)
    theta_e = equivalent_potential_temperature(p, T, Td)
    theta_es = saturation_equivalent_potential_temperature(p, T)

    k = k_index(p, T, Td)
    ssi = showalter_index(p, T, Td)
    tt = total_totals_index(p, T, Td)

    lfc_p, lfc_t = lfc(p, T, Td)
    el_p, el_t = el(p, T, Td)

    # 高度補間関数
    interp_func = interp1d(p.m, z.m, bounds_error=False, fill_value='extrapolate')
    lfc_height_m = interp_func(lfc_p.m) if lfc_p is not None else np.nan
    el_height_m = interp_func(el_p.m) if el_p is not None else np.nan

    # 500hPa 気温
    try:
        idx_500 = np.where(p.m >= 500)[0][-1]
        T500 = T[idx_500]
    except IndexError:
        T500 = np.nan * units.degC

    # 850hPa 気温
    try:
        idx_850 = np.where(p.m >= 850)[0][-1]
        T850 = T[idx_850]
    except IndexError:
        T850 = np.nan * units.degC
      
    fig, ax = plt.subplots(figsize=(8, 10))

    ax.plot(theta, p, color='red', label='θ')
    ax.plot(theta_e, p, color='blue', label='θe')
    ax.plot(theta_es, p, color='green', label='θes')

    if lfc_p is not None:
        ax.axhline(lfc_p.m, color='orange', linestyle='--', label=f'LFC: {lfc_height_m:.0f} m')
    if el_p is not None:
        ax.axhline(el_p.m, color='purple', linestyle='--', label=f'EL: {el_height_m:.0f} m')

    title_center = f"{dt.strftime('%Y-%m-%d %H:%M UTC')} - {station_name} ({station_id})"
    index_info = f"SSI: {ssi[0].magnitude:.1f}   K-Index: {k.magnitude:.1f}   Total Totals: {tt.magnitude:.1f}   500hPa T: {T500.magnitude:.1f} °C   850hPa T: {T850.magnitude:.1f} °C"
    fig.suptitle(title_center, x=0.5, y=0.98, ha='center', fontsize=16)
    fig.text(0.5, 0.955, index_info, ha='center', va='top', fontsize=12)

    ax.set_xlabel('Potential Temperature (K)')
    ax.set_ylabel('Pressure (hPa)')
    ax.invert_yaxis()
    ax.set_yscale('log')
    ax.set_xlim(250, 400)
    ax.set_yticks(np.arange(255, 405, 15))
    ax.set_ylim(1050, 200)
    ax.set_yticks(np.arange(1000, 150, -100))
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())

    ax.grid(True, which='both', linestyle=':')
    ax.legend(loc='lower right')
    plt.tight_layout(rect=[0, 0, 1, 0.97])
  
    # 出力先ディレクトリを作成
    dt_str = (dt.strftime("%Y%m%d%HUTC")).upper()
    output_dir = os.path.join("Data/", dt_str)
    os.makedirs(output_dir, exist_ok=True)  # 再帰的に作成、すでにあってもOK
 
    fname = f"{station_id}_{dt.strftime('%Y%m%d%H')}.png"
    out_path = os.path.join(output_dir, fname)
    plt.savefig(out_path)
    plt.close()
    print(f"Saved to {fname}")

if __name__ == "__main__":
    dt = get_nearest_synoptic_time()
    for sid, sname in stations.items():
        plot_sounding(sid, sname, dt)
