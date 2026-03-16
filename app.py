import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. Data Loading & Preprocessing
# ==========================================

def load_and_preprocess_data():
    # Load csvs
    countries_df = pd.read_csv('countries.csv')
    forest_cover_df = pd.read_csv('forest_cover.csv')
    emissions_df = pd.read_csv('emissions.csv')
    climate_impacts_df = pd.read_csv('climate_impacts.csv')

    # Join fact tables on country_id and year
    df = forest_cover_df.merge(emissions_df, on=['year', 'country_id'], how='inner')
    df = df.merge(climate_impacts_df, on=['year', 'country_id'], how='inner')
    
    # Join with dimension table
    df = df.merge(countries_df, on='country_id', how='left')
    
    # Derived Metrics
    # Emissions intensity (CO2 per forest ha) - area is in millions
    df['emissions_intensity'] = df['co2_emissions_mtco2e'] / df['forest_area_ha_millions']
    
    # Sustainability Score (Simplified)
    # Higher forest %, lower emissions, lower temp anomaly is better
    # Normalize values for index calculation
    df['n_forest'] = (df['forest_percent'] - df['forest_percent'].min()) / (df['forest_percent'].max() - df['forest_percent'].min())
    df['n_emissions'] = (df['co2_emissions_mtco2e'] - df['co2_emissions_mtco2e'].min()) / (df['co2_emissions_mtco2e'].max() - df['co2_emissions_mtco2e'].min())
    df['n_temp'] = (df['temp_anomaly_c'] - df['temp_anomaly_c'].min()) / (df['temp_anomaly_c'].max() - df['temp_anomaly_c'].min())
    
    # Score: Weighted normalized values (Forest is positive, others negative)
    df['sustainability_score'] = (df['n_forest'] * 0.4 - df['n_emissions'] * 0.4 - df['n_temp'] * 0.2) * 100
    
    return df, countries_df

df, countries_df = load_and_preprocess_data()

# ==========================================
# 2. Design Tokens
# ==========================================

COLORS = {
    'primary': '#059669',    # Energetic Emerald
    'secondary': '#10B981',  # Vibrant Mint
    'accent': '#60A5FA',     # Sky Blue (Fresh complement)
    'background': '#F0FDF4', # Light Meadow Tint
    'text': '#064E3B',       # Deep Forest Text
    'white': '#FFFFFF'
}

PLOT_TEMPLATE = {
    'layout': {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': {'family': 'Inter, sans-serif', 'color': '#064E3B'},
        'margin': {'t': 40, 'b': 40, 'l': 40, 'r': 40},
        'hovermode': 'closest'
    }
}

# ==========================================
# 3. Dash App Setup
# ==========================================

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, 'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap'],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

server = app.server
# ==========================================
# 4. App Components
# ==========================================

def create_kpi_card(title, value, subtitle, icon=None):
    return dbc.Card(
        dbc.CardBody([
            html.H6(title, className="text-uppercase text-muted mb-2", style={'fontSize': '0.8rem', 'letterSpacing': '1px'}),
            html.H3(value, className="mb-1", style={'color': COLORS['primary'], 'fontWeight': '700'}),
            html.P(subtitle, className="text-muted mb-0", style={'fontSize': '0.8rem'})
        ]),
        style={'border': f'2px solid {COLORS["primary"]}', 'borderRadius': '12px', 'backgroundColor': COLORS['white'], 'boxShadow': '0 4px 15px rgba(0,0,0,0.05)'}
    )

# Header
header = html.Div([
    html.Div([
        html.H1("Impact of Deforestation on Climate Change", style={'fontWeight': '700', 'color': COLORS['primary'], 'marginBottom': '0'}),
        html.P("Analyzing Global Forest Cover, Emissions, and Climate Trends (2000-2025)", className="text-muted")
    ], className="mb-4"),
    dbc.Row([
        dbc.Col([
            html.Label("Select Country", className="text-muted mb-1", style={'fontSize': '0.9rem'}),
            dbc.Select(
                id='country-selector',
                options=[{'label': 'Global (All Countries)', 'value': 'all'}] + 
                        [{'label': c, 'value': c} for c in df['country'].unique()],
                value='all',
                style={'borderRadius': '8px', 'border': f'1px solid {COLORS["accent"]}'}
            )
        ], width=12, md=4),
        dbc.Col([
            html.Label("Select Year Range", className="text-muted mb-1", style={'fontSize': '0.9rem'}),
            dcc.RangeSlider(
                id='year-range',
                min=df['year'].min(),
                max=df['year'].max(),
                step=1,
                value=[df['year'].min(), df['year'].max()],
                marks={str(year): str(year) for year in range(df['year'].min(), df['year'].max() + 1, 5)},
                className="mt-2"
            )
        ], width=12, md=8)
    ], className="mb-4")
], className="container-fluid pt-4")

# Layout Structure
app.layout = html.Div([
    header,
    html.Div(id='kpi-row', className="container-fluid mb-4"),
    dbc.Container([
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Forest Cover & Emissions Trends", style={'backgroundColor': 'transparent', 'border': 'none', 'fontWeight': '600'}),
                dbc.CardBody(dcc.Loading(dcc.Graph(id='time-series-plot')))
            ], style={'borderRadius': '15px', 'border': 'none', 'boxShadow': '0 10px 30px rgba(0,0,0,0.03)'}), width=12, lg=8),
            
            dbc.Col(dbc.Card([
                dbc.CardHeader("Temperature Anomaly Heatmap", style={'backgroundColor': 'transparent', 'border': 'none', 'fontWeight': '600'}),
                dbc.CardBody(dcc.Loading(dcc.Graph(id='temp-heatmap')))
            ], style={'borderRadius': '15px', 'border': 'none', 'boxShadow': '0 10px 30px rgba(0,0,0,0.03)'}), width=12, lg=4),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Natural Disaster Frequency", style={'backgroundColor': 'transparent', 'border': 'none', 'fontWeight': '600'}),
                dbc.CardBody(dcc.Loading(dcc.Graph(id='disaster-bar')))
            ], style={'borderRadius': '15px', 'border': 'none', 'boxShadow': '0 10px 30px rgba(0,0,0,0.03)'}), width=12, lg=6),
            
            dbc.Col(dbc.Card([
                dbc.CardHeader("Forest Loss vs. Emissions Correlation", style={'backgroundColor': 'transparent', 'border': 'none', 'fontWeight': '600'}),
                dbc.CardBody(dcc.Loading(dcc.Graph(id='scatter-plot')))
            ], style={'borderRadius': '15px', 'border': 'none', 'boxShadow': '0 10px 30px rgba(0,0,0,0.03)'}), width=12, lg=6),
        ], className="mb-4"),
        
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Cross-Metric Correlation Matrix", style={'backgroundColor': 'transparent', 'border': 'none', 'fontWeight': '600'}),
                dbc.CardBody(dcc.Loading(dcc.Graph(id='corr-matrix')))
            ], style={'borderRadius': '15px', 'border': 'none', 'boxShadow': '0 10px 30px rgba(0,0,0,0.03)'}), width=12, lg=5),

            dbc.Col(dbc.Card([
                dbc.CardHeader("Sustainability Score Index", style={'backgroundColor': 'transparent', 'border': 'none', 'fontWeight': '600'}),
                dbc.CardBody(dcc.Loading(dcc.Graph(id='composite-index-plot')))
            ], style={'borderRadius': '15px', 'border': 'none', 'boxShadow': '0 10px 30px rgba(0,0,0,0.03)'}), width=12, lg=7)
        ], className="mb-5")
    ], fluid=True)
], style={'backgroundColor': COLORS['background'], 'minHeight': '100vh', 'fontFamily': 'Inter, sans-serif'})

# ==========================================
# 5. Callbacks
# ==========================================

@app.callback(
    [Output('kpi-row', 'children'),
     Output('time-series-plot', 'figure'),
     Output('temp-heatmap', 'figure'),
     Output('disaster-bar', 'figure'),
     Output('scatter-plot', 'figure'),
     Output('composite-index-plot', 'figure'),
     Output('corr-matrix', 'figure')],
    [Input('country-selector', 'value'),
     Input('year-range', 'value')]
)
def update_dashboard(selected_country, year_range):
    # Initial filter by country
    dff = df if selected_country == 'all' else df[df['country'] == selected_country]
    
    # Filter by year range
    dff = dff[(dff['year'] >= year_range[0]) & (dff['year'] <= year_range[1])]
    
    # ... (KPI calculations)
    recent_year = dff['year'].max()
    start_year = dff['year'].min()
    
    # Forest Loss % (Total 2000-2025)
    f_start = dff[dff['year'] == start_year]['forest_percent'].mean()
    f_end = dff[dff['year'] == recent_year]['forest_percent'].mean()
    forest_change = ((f_end - f_start) / f_start) * 100
    
    # Emissions Growth %
    e_start = dff[dff['year'] == start_year]['co2_emissions_mtco2e'].sum()
    e_end = dff[dff['year'] == recent_year]['co2_emissions_mtco2e'].sum()
    em_growth = ((e_end - e_start) / e_start) * 100
    
    # Temp Anomaly Avg
    temp_avg = dff['temp_anomaly_c'].mean()
    
    # Disaster Total
    dis_total = dff['disasters_count'].sum()
    
    # Emissions Intensity
    em_intensity = dff[dff['year'] == recent_year]['emissions_intensity'].mean()
    
    # Correlation (Forest vs Temp)
    corr_val = dff[['forest_percent', 'temp_anomaly_c']].corr().iloc[0, 1]
    
    kpis = dbc.Row([
        dbc.Col(create_kpi_card("Forest Loss", f"{forest_change:.1f}%", f"Since {start_year}"), width=12, sm=6, lg=2),
        dbc.Col(create_kpi_card("CO2 Growth", f"+{em_growth:.1f}%", "Emission Trajectory"), width=12, sm=6, lg=2),
        dbc.Col(create_kpi_card("Temp Anomaly", f"{temp_avg:.2f}°C", "Annual Average"), width=12, sm=6, lg=2),
        dbc.Col(create_kpi_card("Total Disasters", f"{dis_total}", f"{year_range[0]}-{year_range[1]} Period"), width=12, sm=6, lg=2),
        dbc.Col(create_kpi_card("CO2/Forest Ha", f"{em_intensity:.2f}", "Emissions Intensity"), width=12, sm=6, lg=2),
        dbc.Col(create_kpi_card("Forest-Temp", f"{corr_val:.2f}", "Correlation"), width=12, sm=6, lg=2),
    ])

    # 1. Time Series Plot
    fig_ts = make_subplots(specs=[[{"secondary_y": True}]])
    
    if selected_country == 'all':
        global_trends = dff.groupby('year').agg({'forest_percent': 'mean', 'co2_emissions_mtco2e': 'sum'}).reset_index()
        fig_ts.add_trace(go.Scatter(x=global_trends['year'], y=global_trends['forest_percent'], name="Forest Cover %", line=dict(color=COLORS['primary'], width=3)), secondary_y=False)
        fig_ts.add_trace(go.Bar(x=global_trends['year'], y=global_trends['co2_emissions_mtco2e'], name="CO2 Emissions", marker_color=COLORS['secondary'], opacity=0.6), secondary_y=True)
    else:
        fig_ts.add_trace(go.Scatter(x=dff['year'], y=dff['forest_percent'], name="Forest Cover %", line=dict(color=COLORS['primary'], width=3)), secondary_y=False)
        fig_ts.add_trace(go.Bar(x=dff['year'], y=dff['co2_emissions_mtco2e'], name="CO2 Emissions", marker_color=COLORS['secondary'], opacity=0.6), secondary_y=True)
        
    fig_ts.update_layout(height=400, **PLOT_TEMPLATE['layout'], legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    # 2. Heatmap
    heatmap_data = dff.pivot(index='country', columns='year', values='temp_anomaly_c')
    # If single country selected, ensure the index is preserved correctly for px.imshow
    if selected_country != 'all':
        heatmap_data = heatmap_data.loc[[selected_country]]
    fig_heatmap = px.imshow(
        heatmap_data,
        labels=dict(x="Year", y="Country", color="Temp Anomaly"),
        color_continuous_scale=[[0, '#F0FDF4'], [0.5, COLORS['secondary']], [1, '#EF4444']] # Cool background to heat
    )
    fig_heatmap.update_layout(height=400, **PLOT_TEMPLATE['layout'])

    # 3. Disaster Bar Chart
    fig_disaster = px.bar(dff, x='year', y='disasters_count', 
                         color='country' if selected_country == 'all' else None,
                         color_discrete_sequence=[COLORS['primary'], COLORS['secondary'], COLORS['accent'], '#3B82F6'] if selected_country == 'all' else [COLORS['primary']])
    fig_disaster.update_layout(height=350, **PLOT_TEMPLATE['layout'], showlegend=False if selected_country != 'all' else True)

    # 4. Scatter Plot
    fig_scatter = px.scatter(dff, x='forest_percent', y='co2_emissions_mtco2e', size='disasters_count', 
                            color='temp_anomaly_c', trendline="ols",
                            color_continuous_scale='Mint', labels={'forest_percent': 'Forest Cover %', 'co2_emissions_mtco2e': 'CO2 Emissions'})
    fig_scatter.update_layout(height=350, **PLOT_TEMPLATE['layout'])

    # 5. Correlation Matrix
    corr_matrix = dff[['forest_percent', 'co2_emissions_mtco2e', 'temp_anomaly_c', 'disasters_count']].corr()
    fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='Bluyl', aspect="auto")
    fig_corr.update_layout(height=350, **PLOT_TEMPLATE['layout'])

    # 6. Composite Sustainability Index
    fig_score = px.line(dff, x='year', y='sustainability_score', color='country' if selected_country == 'all' else None,
                       line_shape='spline')
    if selected_country != 'all':
        fig_score.update_traces(line_color=COLORS['primary'], fill='tozeroy', fillcolor='rgba(45, 64, 34, 0.1)')
    fig_score.update_layout(height=350, **PLOT_TEMPLATE['layout'])

    return kpis, fig_ts, fig_heatmap, fig_disaster, fig_scatter, fig_corr, fig_score

if __name__ == '__main__':
    app.run(port=8050)
