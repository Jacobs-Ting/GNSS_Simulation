import dash
from dash import dcc, html, no_update
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import numpy as np
import re
import math

# --- 1. App Initialization ---
app = dash.Dash(__name__)
server = app.server

# --- Common: Multi-Constellation Satellite Data ---
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
    'Navigator': {
        'name': 'Pro Navigator (RHCP Ant)',
        'mismatch_loss': 0.0, 
        'body_loss': 0.0,     
        'mp_factor': 1.0,     
        'color': '#00e5ff' 
    },
    'Wearable': {
        'name': 'Smartwatch (LP Ant)',
        'mismatch_loss': 3.0, 
        'body_loss': 4.0,     
        'mp_factor': 2.5,     
        'color': '#ff003c' 
    }
}

PRN_L1 = np.random.choice([1, -1], size=50)
TIME_L1 = np.arange(50)
PRN_L5 = np.random.choice([1, -1], size=500)
TIME_L5 = np.linspace(0, 50, 500, endpoint=False) 

def parse_coordinate_string(coord_str):
    if not coord_str or not isinstance(coord_str, str): return 25.033964, 121.564468
    
    # 解析 DMS 格式
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

    # 解析 Decimal 格式
    dec_pattern = r"(-?[\d.]+)[,\s]+(-?[\d.]+)"
    match_dec = re.search(dec_pattern, coord_str)
    if match_dec: 
        return float(match_dec.group(1)), float(match_dec.group(2))
    
    return 25.033964, 121.564468

def generate_circle_polygon(lat, lon, radius_m, num_points=36):
    points_lat, points_lon = [], []
    for i in range(num_points + 1):
        angle = math.radians(float(i) / num_points * 360.0)
        dx, dy = radius_m * math.cos(angle), radius_m * math.sin(angle)
        points_lat.append(lat + (dy / 6371000.0) * (180.0 / math.pi))
        points_lon.append(lon + (dx / 6371000.0) * (180.0 / math.pi) / math.cos(lat * math.pi / 180.0))
    return points_lat, points_lon

# --- Theme Variables ---
BG_COLOR = "#0a0e17"       
PANEL_BG = "#131a26"       
BORDER_COLOR = "#1f2937"   
TEXT_COLOR = "#e2e8f0"     
H_COLOR = "#38bdf8"        
ACCENT_GREEN = "#10b981"
ACCENT_RED = "#ef4444"
INPUT_STYLE = {
    'width': '100%', 'marginTop': '5px', 'padding': '8px', 
    'borderRadius': '4px', 'border': f'1px solid {BORDER_COLOR}', 
    'backgroundColor': '#0a0e17', 'color': '#00e5ff', 'boxSizing': 'border-box',
    'fontWeight': 'bold'
}

# --- 2. Layout Design ---
app.layout = html.Div([
    html.H1("GNSS PERFORMANCE SIMULATOR", style={'marginBottom': '20px', 'color': H_COLOR, 'letterSpacing': '2px', 'fontWeight': '900'}),
    dcc.Store(id='map-view-store', data={'zoom': 17, 'center': {'lat': 25.033964, 'lon': 121.564468}}),

    html.Div([
        # --- Left Panel ---
        html.Div([
            html.Div([
                html.H3("1. LINK BUDGET", style={'color': H_COLOR, 'marginTop': '0', 'letterSpacing': '1px'}),
                html.Label("1a. Device Type (Polarization & Body Loss):", style={'fontWeight': 'bold', 'color': TEXT_COLOR}),
                dcc.RadioItems(
                    id='device-type-radio',
                    options=[
                        {'label': ' Pro Navigator (RHCP Ant)', 'value': 'Navigator'},
                        {'label': ' Smartwatch (LP Ant)', 'value': 'Wearable'}
                    ],
                    value='Navigator',
                    inputStyle={"marginRight": "5px"}, labelStyle={"display": "block", "marginBottom": "5px", "color": "#94a3b8"}
                ),
                html.Br(),
                html.Label("1b. Sky Signal Strength (C/N0):", style={'color': TEXT_COLOR}),
                dcc.Slider(
                    id='sky-cn0-slider', min=20, max=50, step=1, value=42,
                    marks={i: {'label': f'{i}', 'style': {'color': '#64748b'}} for i in range(20, 51, 5)},
                ),
                
                html.Div([
                    html.Label("1c. Antenna Efficiency (%) - Gain Impact:", style={'fontWeight': 'bold', 'color': ACCENT_GREEN, 'marginTop': '15px'}),
                    dcc.Input(id='ant-eff-input', type='text', value='85.0', placeholder='e.g., 85', style=INPUT_STYLE)
                ]),

                html.Div([
                    html.Label("1d. Noise Figure (NF) - dB:", style={'fontWeight': 'bold', 'color': ACCENT_RED, 'marginTop': '10px'}),
                    dcc.Input(id='nf-input', type='text', value='2.0', placeholder='e.g., 2.0', style=INPUT_STYLE)
                ]),
                
                html.Div(id='link-budget-info', style={'padding': '12px', 'marginTop': '15px', 'backgroundColor': '#0a0e17', 'borderRadius': '6px', 'fontSize': '13px', 'border': f'1px solid {BORDER_COLOR}', 'color': '#cbd5e1'}),
            ], style={'backgroundColor': PANEL_BG, 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px', 'border': f'1px solid {BORDER_COLOR}', 'boxShadow': '0 4px 6px rgba(0,0,0,0.3)'}),

            html.H3("2. ENVIRONMENT SETTINGS", style={'color': H_COLOR, 'letterSpacing': '1px'}),
            html.Div([
                html.Label("True Coordinates (Lat, Lon):", style={'fontWeight': 'bold', 'color': TEXT_COLOR}),
                dcc.Input(id='coord-input', type='text', value='25.033964, 121.564468', style=INPUT_STYLE, debounce=True),
                html.Label("Tolerance Radius (m):", style={'fontWeight': 'bold', 'color': '#c084fc', 'marginTop': '10px'}),
                dcc.Input(id='tolerance-input', type='text', value='5.0', style={**INPUT_STYLE, 'color': '#c084fc'}),
            ]),
            
            html.Br(),
            
            html.Label("Map Display Style:", style={'fontWeight': 'bold', 'color': TEXT_COLOR}),
            dcc.RadioItems(
                id='map-style-radio',
                options=[
                    {'label': ' 🛰️ Satellite (Tech)', 'value': 'satellite'},
                    {'label': ' 🌃 Dark (Cyber)', 'value': 'carto-darkmatter'},
                    {'label': ' 🗺️ Light Street', 'value': 'open-street-map'}
                ],
                value='satellite', 
                style={'marginBottom': '15px', 'marginTop': '5px'},
                inputStyle={"marginRight": "5px"},
                labelStyle={"marginRight": "12px", "display": "inline-block", "color": "#94a3b8", "fontSize": "13px"}
            ),

            html.Label("Enabled Constellations:", style={'fontWeight': 'bold', 'color': TEXT_COLOR}),
            dcc.Checklist(
                id='sys-checklist', options=[{'label': ' GPS', 'value': 'GPS'}, {'label': ' Galileo', 'value': 'Galileo'}, {'label': ' GLONASS', 'value': 'GLONASS'}, {'label': ' BeiDou', 'value': 'BeiDou'}],
                value=['GPS'], style={'marginBottom': '10px'}, inputStyle={"marginRight": "5px"}, labelStyle={"display": "inline-block", "marginRight": "10px", "color": "#94a3b8"}
            ),
            
            html.Label("Frequency Band:", style={'fontWeight': 'bold', 'color': ACCENT_RED}),
            dcc.RadioItems(
                id='freq-mode-radio', options=[{'label': ' Single-Band (L1)', 'value': 'L1'}, {'label': ' Dual-Band (L1+L5)', 'value': 'L1+L5'}],
                value='L1', style={'marginBottom': '10px'}, inputStyle={"marginRight": "5px"}, labelStyle={"marginRight": "15px", "display": "inline-block", "color": "#94a3b8"}
            ),
            
            html.Label("Max Satellite Channels:", style={'color': TEXT_COLOR}),
            dcc.Slider(id='sat-count-slider', min=4, max=36, step=4, value=12, marks={i: {'label': f'{i}', 'style': {'color': '#64748b'}} for i in range(4, 37, 8)}),
            
            html.Div(id='metrics-panel', style={'marginTop': '20px', 'padding': '15px', 'backgroundColor': '#0a0e17', 'borderRadius': '8px', 'border': f'1px solid {BORDER_COLOR}', 'boxShadow': 'inset 0 2px 4px rgba(0,0,0,0.5)'})
            
        ], style={
            'width': '30%', 'padding': '20px', 'backgroundColor': PANEL_BG, 'borderRadius': '12px', 
            'boxShadow': '0 10px 15px -3px rgba(0,0,0,0.5)', 'height': '85vh', 'overflowY': 'auto',
            'border': f'1px solid {BORDER_COLOR}'
        }),
        
        # --- Right Panel ---
        html.Div([
            html.Div([
                html.Div([dcc.Graph(id='skyplot-graph', config={'displayModeBar': False}, style={'height': '100%'})], style={'width': '49%', 'backgroundColor': PANEL_BG, 'borderRadius': '12px', 'padding': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.3)', 'border': f'1px solid {BORDER_COLOR}'}),
                html.Div([dcc.Graph(id='waveform-graph', config={'displayModeBar': False}, style={'height': '100%'})], style={'width': '49%', 'backgroundColor': PANEL_BG, 'borderRadius': '12px', 'padding': '10px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.3)', 'border': f'1px solid {BORDER_COLOR}'})
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'height': '40vh', 'marginBottom': '20px'}),
            
            html.Div([
                dcc.Graph(id='map-graph', style={'height': '100%', 'borderRadius': '12px', 'overflow': 'hidden'})
            ], style={'flexGrow': '1', 'backgroundColor': PANEL_BG, 'borderRadius': '12px', 'boxShadow': '0 4px 6px rgba(0,0,0,0.3)', 'border': f'1px solid {BORDER_COLOR}', 'padding': '5px'})
            
        ], style={'width': '68%', 'display': 'flex', 'flexDirection': 'column', 'height': '85vh'})
        
    ], style={'display': 'flex', 'justifyContent': 'space-between'})

], style={'padding': '25px', 'fontFamily': '"Inter", "Segoe UI", Roboto, Helvetica, Arial, sans-serif', 'backgroundColor': BG_COLOR, 'minHeight': '100vh', 'color': TEXT_COLOR})

# --- 3. Callbacks ---
@app.callback(Output('coord-input', 'value'), [Input('coord-input', 'value'), Input('map-graph', 'clickData')])
def sync_input_and_click(manual_input, click_data):
    ctx = dash.callback_context
    if not ctx.triggered: return manual_input
    if ctx.triggered[0]['prop_id'].split('.')[0] == 'map-graph' and click_data:
        return f"{click_data['points'][0]['lat']:.6f}, {click_data['points'][0]['lon']:.6f}"
    return manual_input

@app.callback(
    [Output('map-graph', 'figure'), Output('skyplot-graph', 'figure'),
     Output('waveform-graph', 'figure'), Output('metrics-panel', 'children'),
     Output('link-budget-info', 'children')],
    [
     Input('sky-cn0-slider', 'value'),   
     Input('sat-count-slider', 'value'), 
     Input('coord-input', 'value'),      
     Input('freq-mode-radio', 'value'),  
     Input('sys-checklist', 'value'),    
     Input('tolerance-input', 'value'),  
     Input('device-type-radio', 'value'),
     Input('nf-input', 'value'),         
     Input('ant-eff-input', 'value'),
     Input('map-style-radio', 'value')
    ]
)
def update_dashboard(sky_cn0, max_sats, coord_str, freq_mode, selected_sys, tolerance_m, device_type, nf_val, ant_eff_percent, map_style):
    input_lat, input_lon = parse_coordinate_string(coord_str)
    
    try: tolerance_m = float(tolerance_m)
    except: tolerance_m = 5.0
    try: nf_val = float(nf_val)
    except: nf_val = 2.0
    try: ant_eff_percent = float(ant_eff_percent)
    except: ant_eff_percent = 85.0
    ant_eff_percent = max(0.1, min(100.0, ant_eff_percent))
    
    dev = DEVICE_CONFIGS[device_type]
    ant_loss_db = -10 * math.log10(ant_eff_percent / 100.0)
    total_loss = ant_loss_db + dev['mismatch_loss'] + dev['body_loss']
    effective_cn0 = max(10, sky_cn0 - total_loss - nf_val)
    
    link_budget_ui = html.Div([
        html.Div([f"Selected Device: ", html.Span(dev['name'], style={'fontWeight': 'bold', 'color': dev['color']})]),
        html.Hr(style={'margin': '8px 0', 'borderColor': '#334155'}),
        html.Div(f"Sky Signal (C/N0): {sky_cn0} dB-Hz"),
        html.Div(f"- Ant. Efficiency Loss ({ant_eff_percent:.1f}%): {ant_loss_db:.1f} dB", style={'color': '#f87171' if ant_loss_db > 2.0 else '#94a3b8'}),
        html.Div(f"- Mismatch Loss (RHCP->LP): {dev['mismatch_loss']:.1f} dB", style={'color': '#f87171' if dev['mismatch_loss']>0 else '#94a3b8'}),
        html.Div(f"- Body Absorption Loss: {dev['body_loss']:.1f} dB", style={'color': '#f87171' if dev['body_loss']>0 else '#94a3b8'}),
        html.Div(f"- Circuit Noise Figure (NF): {nf_val:.1f} dB", style={'color': '#f87171' if nf_val > 2.5 else '#94a3b8'}),
        html.Hr(style={'margin': '8px 0', 'borderColor': '#334155'}),
        html.Div([f"Effective Baseband C/N0: ", html.Span(f"{effective_cn0:.1f} dB-Hz", style={'fontWeight': '900', 'fontSize': '1.3em', 'color': '#00e5ff', 'textShadow': '0 0 5px rgba(0,229,255,0.5)'})]),
    ])
    
    L1_TRACKING_LIMIT = 22.0
    L5_TRACKING_LIMIT = 23.5  
    
    complete_loss = False
    l5_lost_lock = False
    
    if effective_cn0 < L1_TRACKING_LIMIT:
        complete_loss = True 
        
    if freq_mode == 'L1+L5': 
        if effective_cn0 >= L5_TRACKING_LIMIT:
            base_error, noise_factor = 0.000015, (50 - effective_cn0 + 1) / 30 
        else:
            l5_lost_lock = True 
            if not complete_loss:
                base_error, noise_factor = 0.00003, (50 - effective_cn0 + 1) / 5 
            else:
                base_error, noise_factor = 0.0001, 100 
    else: 
        if not complete_loss:
            base_error, noise_factor = 0.00003, (50 - effective_cn0 + 1) / 10 
        else:
            base_error, noise_factor = 0.0001, 100 

    pool_sats = [sat for sat in SATELLITE_ALMANAC if sat['sys'] in selected_sys]
    visible_sats = pool_sats[:max_sats]
    actual_sat_count = len(visible_sats) 
        
    if actual_sat_count < 4: dop_multiplier = 10.0
    else: dop_multiplier = max(0.5, 1.0 + (12 - actual_sat_count) * 0.1) * dev['mp_factor']
        
    std_dev = base_error * noise_factor * dop_multiplier
    simulated_lats = np.random.normal(input_lat, std_dev, 100)
    simulated_lons = np.random.normal(input_lon, std_dev, 100)
    avg_error_m = std_dev * 111320
    error_percent = (avg_error_m / tolerance_m) * 100
    
    ttff_val, ttff_str, ttff_color = 0, "", ""
    if complete_loss or actual_sat_count < 4:
        ttff_str, ttff_color = "Cannot Fix (Loss of Lock)", "#ff003c"
        error_percent = 999.9 
    else:
        if freq_mode == 'L1+L5' and not l5_lost_lock:
            if effective_cn0 >= 35: ttff_val = 35
            elif effective_cn0 >= 28: ttff_val = 35 + (35 - effective_cn0) * 2
            else: ttff_val = 49 + (28 - effective_cn0) * 5
        else:
            if effective_cn0 >= 38: ttff_val = 35
            elif effective_cn0 >= 32: ttff_val = 35 + (38 - effective_cn0) * 5
            elif effective_cn0 >= 28: ttff_val = 65 + (32 - effective_cn0) * 15
            else: ttff_val = 999 
        
        if actual_sat_count > 12 and ttff_val != 999: ttff_val = int(ttff_val * 0.85)
        if ttff_val == 999: ttff_str, ttff_color = "> 5 mins (Failed)", "#ff003c"
        else: ttff_str, ttff_color = f"~ {int(ttff_val)} sec", ("#39ff14" if ttff_val <= 45 else "#eab308")

    error_color = "#39ff14" if error_percent <= 100 else "#ff003c"
    warning_ui = None
    if complete_loss:
        warning_ui = html.Div("💀 FATAL: Signal below L1 tracking limit (22.0 dB). Receiver lost lock!", style={'color': '#fff', 'backgroundColor': '#7f1d1d', 'padding': '10px', 'borderRadius': '4px', 'marginBottom': '10px', 'fontSize': '13px', 'fontWeight': 'bold', 'border': '1px solid #ef4444'})
    elif freq_mode == 'L1+L5' and l5_lost_lock:
        warning_ui = html.Div("⚠️ WARNING: Signal below L5 tracking limit (23.5 dB). Downgraded to L1 only!", style={'color': '#fff', 'backgroundColor': '#9a3412', 'padding': '10px', 'borderRadius': '4px', 'marginBottom': '10px', 'fontSize': '13px', 'fontWeight': 'bold', 'border': '1px solid #f97316'})

    metrics_ui = html.Div([
        html.H4("📊 PERFORMANCE METRICS", style={'marginTop': '0', 'marginBottom': '15px', 'color': '#f8fafc', 'letterSpacing': '1px'}),
        warning_ui if warning_ui else html.Span(),
        html.Div([f"Avg. Error Distance: ", html.Span(f"{avg_error_m:.1f} m", style={'color': '#e2e8f0', 'fontWeight': 'bold'})], style={'marginBottom': '5px'}),
        html.Div([f"Positioning EVM: ", html.Span(f"{error_percent:.1f}%" if error_percent != 999.9 else "N/A", style={'color': error_color, 'fontWeight': 'bold', 'textShadow': f'0 0 5px {error_color}'})], style={'marginBottom': '5px'}),
        html.Div([f"Est. Cold Start TTFF: ", html.Span(ttff_str, style={'color': ttff_color, 'fontWeight': 'bold', 'textShadow': f'0 0 5px {ttff_color}'})]),
    ])

    chart_layout_base = dict(
        paper_bgcolor='#131a26',
        plot_bgcolor='#131a26',
        font=dict(color='#94a3b8'),
        margin={"r":20,"t":40,"l":20,"b":30}
    )

    circle_lats, circle_lons = generate_circle_polygon(input_lat, input_lon, tolerance_m)
    fig_map = go.Figure()
    
    fig_map.add_trace(go.Scattermapbox(lat=circle_lats, lon=circle_lons, mode='lines', fill='toself', fillcolor='rgba(192, 132, 252, 0.1)', line=dict(color='#c084fc', width=2), name=f'Tolerance ({tolerance_m}m)'))
    fig_map.add_trace(go.Scattermapbox(lat=simulated_lats, lon=simulated_lons, mode='markers', marker=dict(size=8, color=dev['color'], opacity=0.6), name='Simulated Fix'))
    fig_map.add_trace(go.Scattermapbox(lat=[input_lat], lon=[input_lon], mode='markers', marker=dict(size=18, color='#ff4081', symbol='star'), name='Ground Truth'))
    
    # 🛠️ 關鍵修復：用真實座標當作 uirevision_base
    # 只要座標沒改，你拉動任何滑桿，地圖的縮放和平移都會完美保留！
    uirevision_base = f"{input_lat:.6f}_{input_lon:.6f}"
    map_center = dict(lat=input_lat, lon=input_lon)
    map_zoom = 17
    
    if map_style == 'satellite':
        mapbox_args = dict(
            style="white-bg",
            center=map_center, zoom=map_zoom,
            layers=[dict(sourcetype="raster", source=["https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"], below="traces")]
        )
    else:
        mapbox_args = dict(style=map_style, center=map_center, zoom=map_zoom)

    fig_map.update_layout(
        mapbox=mapbox_args, 
        margin={"r":0,"t":0,"l":0,"b":0}, 
        uirevision=uirevision_base, # 🛡️ 守護地圖視角的魔法參數
        paper_bgcolor='#131a26'
    )

    fig_sky = go.Figure()
    for sys_name in ['GPS', 'Galileo', 'GLONASS', 'BeiDou']:
        sys_sats = [s for s in visible_sats if s['sys'] == sys_name]
        if sys_sats: fig_sky.add_trace(go.Scatterpolar(r=[90 - sat["el"] for sat in sys_sats], theta=[sat["az"] for sat in sys_sats], mode='markers+text', marker=dict(size=12, color=SYS_COLORS[sys_name], line=dict(color='#fff', width=1)), text=[sat["prn"] for sat in sys_sats], textposition="top center", name=sys_name))
    
    fig_sky.update_layout(
        **chart_layout_base,
        title=dict(text="Constellation Skyplot", font=dict(color='#e2e8f0', size=16)),
        showlegend=True, legend=dict(font=dict(color='#94a3b8'), orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        polar=dict(
            bgcolor='#0a0e17',
            angularaxis=dict(rotation=90, direction="clockwise", gridcolor='#334155', linecolor='#334155', tickvals=[0, 45, 90, 135, 180, 225, 270, 315], ticktext=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], tickfont=dict(color='#94a3b8')), 
            radialaxis=dict(range=[0, 90], gridcolor='#334155', showline=False, tickfont=dict(color='#94a3b8'))
        )
    )

    waveform_noise_std = (50 - effective_cn0) * 0.08 
    received_l1 = PRN_L1 + np.random.normal(0, waveform_noise_std, len(PRN_L1))
    received_l5 = PRN_L5 + np.random.normal(0, waveform_noise_std, len(PRN_L5))
    fig_wave = go.Figure()
    
    l5_is_active = (freq_mode == 'L1+L5') and not l5_lost_lock
    l5_opacity = 0.8 if l5_is_active else 0.2
    l5_color = '#ff003c' if l5_is_active else '#475569' 
    
    fig_wave.add_trace(go.Scatter(x=TIME_L1, y=received_l1 + 2, mode='lines+markers', line=dict(color='#475569', width=1), marker=dict(size=3, color='#475569'), name='L1 Rx'))
    fig_wave.add_trace(go.Scatter(x=TIME_L1, y=PRN_L1 + 2, mode='lines', line=dict(color='#00e5ff', width=2, shape='hv'), name='L1 Ideal'))
    fig_wave.add_trace(go.Scatter(x=TIME_L5, y=received_l5 - 2, mode='lines', line=dict(color='#475569', width=1), opacity=l5_opacity, name='L5 Rx'))
    fig_wave.add_trace(go.Scatter(x=TIME_L5, y=PRN_L5 - 2, mode='lines', line=dict(color=l5_color, width=2, shape='hv'), opacity=l5_opacity, name='L5 Ideal'))
    
    fig_wave.update_layout(
        **chart_layout_base,
        title=dict(text="Baseband Waveform (L1 vs L5)", font=dict(color='#e2e8f0', size=16)),
        xaxis=dict(gridcolor='#334155', zeroline=False),
        yaxis=dict(range=[-4.5, 4.5], tickvals=[-2, 2], ticktext=['L5 Band', 'L1 Band'], gridcolor='#334155', zeroline=True, zerolinecolor='#475569'),
        legend=dict(font=dict(color='#94a3b8'), orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig_map, fig_sky, fig_wave, metrics_ui, link_budget_ui

if __name__ == '__main__':
    app.run(debug=True)