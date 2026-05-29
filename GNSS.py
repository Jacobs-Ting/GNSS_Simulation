import streamlit as st
import plotly.graph_objects as go
import numpy as np
import re
import math
import time

# --- 1. Streamlit 頁面設定 ---
st.set_page_config(page_title="GNSS Performance Simulator", layout="wide", initial_sidebar_state="expanded")

# --- 公用常數與資料 ---
SATELLITE_ALMANAC = [
    {"prn": "G01", "az": 45, "el": 60, "sys": "GPS"}, {"prn": "G04", "az": 135, "el": 30, "sys": "GPS"},
    {"prn": "G07", "az": 225, "el": 75, "sys": "GPS"}, {"prn": "G10", "az": 315, "el": 40, "sys": "GPS"},
    {"prn": "G13", "az": 10, "el": 85, "sys": "GPS"}, {"prn": "G16", "az": 100, "el": 20, "sys": "GPS"},
    {"prn": "G19", "az": 190, "el": 50, "sys": "GPS"}, {"prn": "G22", "az": 280, "el": 15, "sys": "GPS"},
    {"prn": "G25", "az": 80, "el": 70, "sys": "GPS"}, {"prn": "G28", "az": 170, "el": 25, "sys": "GPS"},
    {"prn": "G31", "az": 260, "el": 65, "sys": "GPS"}, {"prn": "G02", "az": 350, "el": 35, "sys": "GPS"},
    {"prn": "E03", "az": 60, "el": 45, "sys": "Galileo"}, {"prn": "E05", "az": 150, "el": 55, "sys": "Galileo"},
    {"prn": "E09", "az": 240, "el": 25, "sys": "Galileo"}, {"prn": "E12", "az": 330, "el": 80, "sys": "Galileo"},
    {"prn": "E24", "az": 25, "el": 35, "sys": "Galileo"}, {"prn": "E30", "az": 115, "el": 65, "sys": "Galileo"},
    {"prn": "E33", "az": 205, "el": 10, "sys": "Galileo"}, {"prn": "E36", "az": 295, "el": 50, "sys": "Galileo"},
    {"prn": "R02", "az": 30, "el": 70, "sys": "GLONASS"}, {"prn": "R05", "az": 120, "el": 40, "sys": "GLONASS"},
    {"prn": "R08", "az": 210, "el": 85, "sys": "GLONASS"}, {"prn": "R15", "az": 300, "el": 20, "sys": "GLONASS"},
    {"prn": "R18", "az": 75, "el": 15, "sys": "GLONASS"}, {"prn": "R21", "az": 165, "el": 60, "sys": "GLONASS"},
    {"prn": "R24", "az": 255, "el": 30, "sys": "GLONASS"}, {"prn": "R01", "az": 345, "el": 55, "sys": "GLONASS"},
    {"prn": "C01", "az": 90, "el": 80, "sys": "BeiDou"}, {"prn": "C06", "az": 180, "el": 35, "sys": "BeiDou"},
    {"prn": "C10", "az": 270, "el": 60, "sys": "BeiDou"}, {"prn": "C13", "az": 0, "el": 25, "sys": "BeiDou"},
    {"prn": "C16", "az": 50, "el": 50, "sys": "BeiDou"}, {"prn": "C21", "az": 140, "el": 10, "sys": "BeiDou"},
    {"prn": "C29", "az": 230, "el": 45, "sys": "BeiDou"}, {"prn": "C33", "az": 320, "el": 75, "sys": "BeiDou"},
]

SYS_COLORS = {"GPS": "#00e5ff", "Galileo": "#39ff14", "GLONASS": "#ff003c", "BeiDou": "#ffea00"}
DEVICE_CONFIGS = {
    'Navigator': {'name': 'Pro Navigator (RHCP Ant)', 'mismatch_loss': 0.0, 'body_loss': 0.0, 'mp_factor': 1.0, 'color': '#00e5ff'},
    'Wearable': {'name': 'Smartwatch (LP Ant)', 'mismatch_loss': 3.0, 'body_loss': 4.0, 'mp_factor': 2.5, 'color': '#ff003c'}
}
BORDER_COLOR = "#1f2937"

PRN_L1 = np.random.choice([1, -1], size=50)
TIME_L1 = np.arange(50)
PRN_L5 = np.random.choice([1, -1], size=500)
TIME_L5 = np.linspace(0, 50, 500, endpoint=False) 

# --- 輔助函數 ---
def parse_coordinate_string(coord_str):
    if not coord_str: return 25.033964, 121.564468
    dms_pattern = r"(\d+)[°\s]+(\d+)['\s]+([\d.]+)[^NSns]*([NSns])[,\s]*(\d+)[°\s]+(\d+)['\s]+([\d.]+)[^EWew]*([EWew])"
    match_dms = re.search(dms_pattern, coord_str)
    if match_dms:
        lat_d, lat_m, lat_s, lat_dir = float(match_dms.group(1)), float(match_dms.group(2)), float(match_dms.group(3)), match_dms.group(4).upper()
        lon_d, lon_m, lon_s, lon_dir = float(match_dms.group(5)), float(match_dms.group(6)), float(match_dms.group(7)), match_dms.group(8).upper()
        lat = lat_d + (lat_m / 60.0) + (lat_s / 3600.0)
        lon = lon_d + (lon_m / 60.0) + (lon_s / 3600.0)
        if lat_dir == 'S': lat = -lat
        if lon_dir == 'W': lon = -lon
        return lat, lon

    dec_pattern = r"(-?[\d.]+)[,\s]+(-?[\d.]+)"
    match_dec = re.search(dec_pattern, coord_str)
    if match_dec: return float(match_dec.group(1)), float(match_dec.group(2))
    return 25.033964, 121.564468

def generate_circle_polygon(lat, lon, radius_m, num_points=36):
    points_lat, points_lon = [], []
    for i in range(num_points + 1):
        angle = math.radians(float(i) / num_points * 360.0)
        dx, dy = radius_m * math.cos(angle), radius_m * math.sin(angle)
        points_lat.append(lat + (dy / 6371000.0) * (180.0 / math.pi))
        points_lon.append(lon + (dx / 6371000.0) * (180.0 / math.pi) / math.cos(lat * math.pi / 180.0))
    return points_lat, points_lon

def get_interpolated_path(points, steps_per_segment=15):
    path = []
    if not points: return path
    for i in range(len(points)-1):
        lat_start, lon_start = points[i]
        lat_end, lon_end = points[i+1]
        for j in range(steps_per_segment):
            path.append((
                lat_start + (lat_end - lat_start) * j / steps_per_segment,
                lon_start + (lon_end - lon_start) * j / steps_per_segment
            ))
    path.append(points[-1])
    return path

# --- 2. Sidebar (側邊欄控制面板) ---
with st.sidebar:
    st.markdown("<h2 style='color: #38bdf8;'>1. LINK BUDGET</h2>", unsafe_allow_html=True)
    device_type = st.radio("1a. Device Type (Polarization):", ["Navigator", "Wearable"], format_func=lambda x: "Pro Navigator (RHCP Ant)" if x=="Navigator" else "Smartwatch (LP Ant)")
    sky_cn0 = st.slider("1b. Sky Signal Strength (C/N0):", 20, 50, 42, 1)
    ant_eff_str = st.text_input("1c. Antenna Efficiency (%) - Gain Impact:", value="85.0")
    
    # 🚀 終極版 RF 前端 Friis 計算區域
    st.markdown("<h3 style='color: #a78bfa; font-size: 16px; margin-top: 15px;'>1d. RF Front-End (Friis Formula)</h3>", unsafe_allow_html=True)
    pre_filter_loss = st.number_input("Pre-Filter Loss (dB):", value=1.0, step=0.1)
    lna_gain = st.number_input("LNA Gain (dB):", value=18.0, step=0.5)
    lna_nf = st.number_input("LNA NF (dB):", value=1.2, step=0.1)
    post_lna_nf = st.number_input("Post-LNA/RFIC NF (dB):", value=4.0, step=0.1)
    
    st.markdown("<h2 style='color: #38bdf8; margin-top: 20px;'>2. ENVIRONMENT</h2>", unsafe_allow_html=True)
    coord_str = st.text_input("True Coordinates (Lat, Lon):", value="25.033964, 121.564468")
    tolerance_str = st.text_input("Tolerance Radius (m):", value="5.0")
    map_style = st.radio("Map Display Style:", ["satellite", "carto-darkmatter", "open-street-map"], format_func=lambda x: "🛰️ Satellite (Tech)" if x=="satellite" else "🌃 Dark (Cyber)" if x=="carto-darkmatter" else "🗺️ Light Street")
    selected_sys = st.multiselect("Enabled Constellations:", ["GPS", "Galileo", "GLONASS", "BeiDou"], default=["GPS"])
    freq_mode = st.radio("Frequency Band:", ["L1", "L1+L5"], format_func=lambda x: "Single-Band (L1)" if x=="L1" else "Dual-Band (L1+L5)")
    max_sats = st.slider("Max Satellite Channels:", 4, 36, 12, 4)

    st.markdown("<h2 style='color: #f43f5e; margin-top: 20px;'>3. DYNAMIC KINEMATIC ENGINE</h2>", unsafe_allow_html=True)
    scenario = st.selectbox(
        "Select Stress Test Scenario:",
        ["🔴 Urban Canyon (Taipei 101 Perimeter)", "🟢 Open Sky (Kaohsiung Park Loop)"]
    )
    run_sim = st.button("▶️ RUN DYNAMIC TEST", use_container_width=True, type="primary")

# --- 核心邏輯解析與 Friis 計算 ---
input_lat, input_lon = parse_coordinate_string(coord_str)
try: tolerance_m = float(tolerance_str)
except: tolerance_m = 5.0
try: ant_eff_percent = float(ant_eff_str)
except: ant_eff_percent = 85.0
ant_eff_percent = max(0.1, min(100.0, ant_eff_percent))

# 📐 Friis 公式計算 Final LNA NF
# 1. 將 dB 轉成線性值 (Linear scale)
f_pre = 10 ** (pre_filter_loss / 10)
g_pre = 10 ** (-pre_filter_loss / 10) # 濾波器的增益是負的 loss
f_lna = 10 ** (lna_nf / 10)
g_lna = 10 ** (lna_gain / 10)
f_post = 10 ** (post_lna_nf / 10)

# 2. 套用 Friis Cascaded Noise Factor formula
f_total = f_pre + (f_lna - 1) / g_pre + (f_post - 1) / (g_pre * g_lna)

# 3. 轉回 dB 成為 Final LNA NF (System NF)
nf_val = 10 * math.log10(f_total)

# 鏈路損耗計算
dev = DEVICE_CONFIGS[device_type]
ant_loss_db = -10 * math.log10(ant_eff_percent / 100.0)
total_loss = ant_loss_db + dev['mismatch_loss'] + dev['body_loss']

# --- Main Area 渲染 ---
st.markdown(f"<h1 style='color: #38bdf8; letter-spacing: 2px;'>GNSS PERFORMANCE SIMULATOR</h1>", unsafe_allow_html=True)
st.markdown("---")

col_cn0, col_metrics = st.columns([1, 2])
with col_cn0:
    ui_cn0_box = st.empty()
with col_metrics:
    m1, m2, m3 = st.columns(3)
    with m1: ui_error_box = st.empty()
    with m2: ui_evm_box = st.empty()
    with m3: ui_status_box = st.empty()

st.markdown("---")

chart_layout_base = dict(paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', font=dict(color='#94a3b8'), margin={"r":20,"t":40,"l":20,"b":30})
c_chart1, c_chart2 = st.columns(2)

with c_chart1:
    fig_sky = go.Figure()
    pool_sats = [sat for sat in SATELLITE_ALMANAC if sat['sys'] in selected_sys]
    visible_sats = pool_sats[:max_sats]
    actual_sat_count = len(visible_sats) 
    for sys_name in ['GPS', 'Galileo', 'GLONASS', 'BeiDou']:
        sys_sats = [s for s in visible_sats if s['sys'] == sys_name]
        if sys_sats: fig_sky.add_trace(go.Scatterpolar(r=[90 - sat["el"] for sat in sys_sats], theta=[sat["az"] for sat in sys_sats], mode='markers+text', marker=dict(size=12, color=SYS_COLORS[sys_name], line=dict(color='#fff', width=1)), text=[sat["prn"] for sat in sys_sats], textposition="top center", name=sys_name))
    fig_sky.update_layout(**chart_layout_base, title=dict(text="Constellation Skyplot", font=dict(color='#e2e8f0', size=16)), showlegend=True, legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5), polar=dict(bgcolor='#0e1117', angularaxis=dict(rotation=90, direction="clockwise", gridcolor='#334155', linecolor='#334155'), radialaxis=dict(range=[0, 90], gridcolor='#334155', showline=False)))
    st.plotly_chart(fig_sky, use_container_width=True)

with c_chart2:
    effective_cn0_static = max(10, sky_cn0 - total_loss - nf_val)
    waveform_noise_std = (50 - effective_cn0_static) * 0.08 
    received_l1 = PRN_L1 + np.random.normal(0, waveform_noise_std, len(PRN_L1))
    received_l5 = PRN_L5 + np.random.normal(0, waveform_noise_std, len(PRN_L5))
    fig_wave = go.Figure()
    l5_is_active = (freq_mode == 'L1+L5') and (effective_cn0_static >= 23.5)
    l5_opacity = 0.8 if l5_is_active else 0.2
    l5_color = '#ff003c' if l5_is_active else '#475569' 
    fig_wave.add_trace(go.Scatter(x=TIME_L1, y=received_l1 + 2, mode='lines+markers', line=dict(color='#475569', width=1), marker=dict(size=3, color='#475569'), name='L1 Rx'))
    fig_wave.add_trace(go.Scatter(x=TIME_L1, y=PRN_L1 + 2, mode='lines', line=dict(color='#00e5ff', width=2, shape='hv'), name='L1 Ideal'))
    fig_wave.add_trace(go.Scatter(x=TIME_L5, y=received_l5 - 2, mode='lines', line=dict(color='#475569', width=1), opacity=l5_opacity, name='L5 Rx'))
    fig_wave.add_trace(go.Scatter(x=TIME_L5, y=PRN_L5 - 2, mode='lines', line=dict(color=l5_color, width=2, shape='hv'), opacity=l5_opacity, name='L5 Ideal'))
    fig_wave.update_layout(**chart_layout_base, title=dict(text="Baseband Waveform", font=dict(color='#e2e8f0', size=16)), yaxis=dict(range=[-4.5, 4.5], tickvals=[-2, 2], ticktext=['L5 Band', 'L1 Band'], gridcolor='#334155', zeroline=True, zerolinecolor='#475569'), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_wave, use_container_width=True)

map_placeholder = st.empty()

# --- 🚀 模式分流：靜態顯示 vs 動態模擬 ---
if not run_sim:
    effective_cn0 = effective_cn0_static
    complete_loss = effective_cn0 < 22.0
    l5_lost_lock = False

    if freq_mode == 'L1+L5': 
        if effective_cn0 >= 23.5: base_error, noise_factor = 0.000015, (50 - effective_cn0 + 1) / 30 
        else:
            l5_lost_lock = True 
            base_error, noise_factor = (0.0001, 100) if complete_loss else (0.00003, (50 - effective_cn0 + 1) / 5)
    else: 
        base_error, noise_factor = (0.0001, 100) if complete_loss else (0.00003, (50 - effective_cn0 + 1) / 10)

    dop_multiplier = 10.0 if actual_sat_count < 4 else max(0.5, 1.0 + (12 - actual_sat_count) * 0.1) * dev['mp_factor']
    std_dev = base_error * noise_factor * dop_multiplier
    simulated_lats = np.random.normal(input_lat, std_dev, 100)
    simulated_lons = np.random.normal(input_lon, std_dev, 100)
    avg_error_m = std_dev * 111320
    error_percent = 999.9 if (complete_loss or actual_sat_count < 4) else (avg_error_m / tolerance_m) * 100

    status_str, status_color = ("LOSS OF LOCK", "#ff003c") if complete_loss else ("TRACKING", "#39ff14")

    # 寫入靜態數據與 Friis 結果
    ui_cn0_box.markdown(f"<div style='background-color: #131a26; padding: 15px; border-radius: 10px; border: 1px solid {BORDER_COLOR};'><div style='color: #00e5ff; font-weight: bold; font-size: 1.2em;'>Static Effective C/N0: {effective_cn0:.1f} dB-Hz</div><div style='color: #a78bfa; font-size: 0.9em; margin-top: 5px;'>⚙️ Final LNA NF (Friis): {nf_val:.2f} dB</div></div>", unsafe_allow_html=True)
    ui_error_box.metric("Avg. Error Distance", f"{avg_error_m:.1f} m")
    ui_evm_box.metric("Positioning EVM", "N/A" if error_percent == 999.9 else f"{error_percent:.1f}%", delta="Pass" if error_percent <= 100 else "Fail", delta_color="normal" if error_percent <= 100 else "inverse")
    ui_status_box.markdown(f"<div style='background-color: #131a26; padding: 15px; border-radius: 10px; border: 1px solid #1f2937;'><div style='color: #94a3b8; font-size: 14px;'>Receiver Status</div><div style='color: {status_color}; font-size: 20px; font-weight: bold;'>{status_str}</div></div>", unsafe_allow_html=True)

    circle_lats, circle_lons = generate_circle_polygon(input_lat, input_lon, tolerance_m)
    fig_map = go.Figure()
    fig_map.add_trace(go.Scattermapbox(lat=circle_lats, lon=circle_lons, mode='lines', fill='toself', fillcolor='rgba(192, 132, 252, 0.1)', line=dict(color='#c084fc', width=2)))
    fig_map.add_trace(go.Scattermapbox(lat=simulated_lats, lon=simulated_lons, mode='markers', marker=dict(size=8, color=dev['color'], opacity=0.6)))
    fig_map.add_trace(go.Scattermapbox(lat=[input_lat], lon=[input_lon], mode='markers', marker=dict(size=18, color='#ff4081', symbol='star')))
    mapbox_args = dict(style="white-bg", center=dict(lat=input_lat, lon=input_lon), zoom=17, layers=[dict(sourcetype="raster", source=["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"], below="traces")]) if map_style == 'satellite' else dict(style=map_style, center=dict(lat=input_lat, lon=input_lon), zoom=17)
    fig_map.update_layout(mapbox=mapbox_args, margin={"r":0,"t":0,"l":0,"b":0}, height=500, uirevision=f"{input_lat}_{input_lon}", paper_bgcolor='#0e1117', showlegend=False)
    map_placeholder.plotly_chart(fig_map, use_container_width=True)

else:
    if "101" in scenario:
        route_waypoints = [
            (25.0326 + 0.0002, 121.5616 + 0.0007),
            (25.0358 - 0.0004, 121.5616 + 0.0007),
            (25.0358 - 0.0004, 121.5646 + 0.0002),
            (25.0326 + 0.0002, 121.5646 + 0.0002),
            (25.0326 + 0.0002, 121.5616 + 0.0007)
        ]
        map_center_dyn = dict(lat=25.0342, lon=121.5638) 
        zoom_level_dyn = 16.5
        true_path = get_interpolated_path(route_waypoints, steps_per_segment=15) 
    else:
        park_loop_points = [
            (22.731306, 120.307000), 
            (22.731778, 120.307301), 
            (22.732156, 120.307223), 
            (22.732252, 120.307010), 
            (22.731853, 120.306894), 
            (22.731306, 120.307000)  
        ]
        map_center_dyn = dict(lat=22.7317, lon=120.3071)
        zoom_level_dyn = 17.2 
        true_path = get_interpolated_path(park_loop_points, steps_per_segment=13) 

    trail_lats, trail_lons = [], []
    true_trail_lats, true_trail_lons = [], []
    
    progress_bar = st.progress(0)
    
    for step, (cur_lat, cur_lon) in enumerate(true_path):
        progress_bar.progress((step + 1) / len(true_path))
        
        dyn_cn0 = sky_cn0
        current_env_status = "NORMAL LOS"
        status_color = "#39ff14"
        
        if "101" in scenario:
            if 15 <= step < 30: 
                dyn_cn0 -= 18
                current_env_status = "SEVERE NLOS (Canyon)"
                status_color = "#ff003c"
            elif 30 <= step < 45: 
                dyn_cn0 -= 8
                current_env_status = "PARTIAL SHADOWING"
                status_color = "#eab308"
            else: dyn_cn0 -= 2
        else: 
            scintillation = np.random.uniform(-3.0, 1.0) 
            dyn_cn0 += (3.0 + scintillation) 
            current_env_status = "OPEN SKY w/ TREE FADING"
            status_color = "#00e5ff"

        effective_cn0 = max(10, dyn_cn0 - total_loss - nf_val)
        complete_loss = effective_cn0 < 22.0

        if freq_mode == 'L1+L5': 
            if effective_cn0 >= 23.5: base_error, noise_factor = 0.000015, (50 - effective_cn0 + 1) / 30 
            else: base_error, noise_factor = (0.0001, 50) if complete_loss else (0.00003, (50 - effective_cn0 + 1) / 5)
        else: 
            base_error, noise_factor = (0.0001, 50) if complete_loss else (0.00003, (50 - effective_cn0 + 1) / 10)

        turn_angle = 0
        if step > 0 and step < len(true_path)-1:
            prev_lat, prev_lon = true_path[step-1]
            next_lat, next_lon = true_path[step+1]
            bearing1 = math.atan2(cur_lon - prev_lon, cur_lat - prev_lat)
            bearing2 = math.atan2(next_lon - cur_lon, next_lat - cur_lat)
            turn_angle = abs(bearing2 - bearing1)
            if turn_angle > 0.5: dev['mp_factor'] *= 1.3 

        dop_multiplier = 10.0 if actual_sat_count < 4 else max(0.5, 1.0 + (12 - actual_sat_count) * 0.1) * dev['mp_factor']
        std_dev = base_error * noise_factor * dop_multiplier
        if turn_angle > 0.5: dev['mp_factor'] /= 1.3
        
        sim_lat = np.random.normal(cur_lat, std_dev)
        sim_lon = np.random.normal(cur_lon, std_dev)
        trail_lats.append(sim_lat)
        trail_lons.append(sim_lon)
        true_trail_lats.append(cur_lat)
        true_trail_lons.append(cur_lon)
        
        avg_error_m = std_dev * 111320
        error_percent = 999.9 if (complete_loss or actual_sat_count < 4) else (avg_error_m / tolerance_m) * 100

        ui_cn0_box.markdown(f"<div style='background-color: #131a26; padding: 15px; border-radius: 10px; border: 1px solid {status_color}; box-shadow: 0 0 10px {status_color};'><div style='color: #94a3b8; font-size: 14px;'>Live Telemetry (Step {step+1}/{len(true_path)})</div><div style='color: {status_color}; font-weight: bold; font-size: 1.5em;'>{effective_cn0:.1f} dB-Hz</div><div style='color: #a78bfa; font-size: 0.9em; margin-top: 5px;'>⚙️ Final LNA NF: {nf_val:.2f} dB</div></div>", unsafe_allow_html=True)
        ui_error_box.metric("Live Avg. Error", f"{avg_error_m:.1f} m")
        ui_evm_box.metric("Live EVM", "N/A" if error_percent == 999.9 else f"{error_percent:.1f}%", delta="Tracking" if not complete_loss else "Loss of Lock", delta_color="normal" if not complete_loss else "inverse")
        ui_status_box.markdown(f"<div style='background-color: #131a26; padding: 15px; border-radius: 10px; border: 1px solid #1f2937;'><div style='color: #94a3b8; font-size: 14px;'>Environment Profile</div><div style='color: {status_color}; font-size: 18px; font-weight: bold;'>{current_env_status}</div></div>", unsafe_allow_html=True)

        time.sleep(0.05) 

    progress_bar.empty()
    
    fig_map_dyn = go.Figure()
    fig_map_dyn.add_trace(go.Scattermapbox(lat=true_trail_lats, lon=true_trail_lons, mode='lines', line=dict(color='#ff4081', width=3), name='True Path'))
    fig_map_dyn.add_trace(go.Scattermapbox(lat=trail_lats, lon=trail_lons, mode='lines+markers', marker=dict(size=6, color=dev['color'], opacity=0.6), line=dict(color=dev['color'], width=2), name='Sim Track'))
    
    circle_lats, circle_lons = generate_circle_polygon(true_trail_lats[-1], true_trail_lons[-1], tolerance_m)
    fig_map_dyn.add_trace(go.Scattermapbox(lat=circle_lats, lon=circle_lons, mode='lines', fill='toself', fillcolor='rgba(192, 132, 252, 0.1)', line=dict(color='#c084fc', width=2), name=f'Tolerance ({tolerance_m}m)'))
    
    mapbox_args_dyn = dict(style="white-bg", center=map_center_dyn, zoom=zoom_level_dyn, layers=[dict(sourcetype="raster", source=["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"], below="traces")]) if map_style == 'satellite' else dict(style=map_style, center=map_center_dyn, zoom=zoom_level_dyn)
    
    fig_map_dyn.update_layout(mapbox=mapbox_args_dyn, margin={"r":0,"t":0,"l":0,"b":0}, height=500, paper_bgcolor='#0e1117', showlegend=False)
    
    map_placeholder.plotly_chart(fig_map_dyn, use_container_width=True)