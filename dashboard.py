import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

# -------------------------------------------------------------------------------#
# 1. 加载数据
# -------------------------------------------------------------------------------#
# 加载regression_summary.csv和cluster_summary.csv
regression_summary = pd.read_csv('regression_summary.csv')
cluster_summary = pd.read_csv('cluster_summary.csv')

# 加载原始数据以支持时间序列图
aging_data_raw = pd.read_csv('API_SP.POP.65UP.TO.ZS_DS2_en_csv_v2_22652.csv', skiprows=4)
aging_data_raw = aging_data_raw.rename(columns={'Country Name': 'country', 'Country Code': 'country_code'})

years = [str(year) for year in range(2000, 2031)]
available_years = [y for y in years if y in aging_data_raw.columns]
aging_data = aging_data_raw[['country', 'country_code'] + available_years].copy()
aging_data = aging_data.dropna(subset=available_years, how='all')

# 数据转换：将aging_data转换为长格式以便绘图
aging_data_long = aging_data.melt(id_vars=['country', 'country_code'], 
                                  value_vars=available_years, 
                                  var_name='year', 
                                  value_name='aging_percent')
aging_data_long['year'] = aging_data_long['year'].astype(int)
aging_data_long = aging_data_long.dropna()

# 为cluster_summary添加聚类标签（低、中、高）
cluster_summary['cluster_label'] = cluster_summary['cluster'].map({0: 'Low', 1: 'Medium', 2: 'High'})


# 加载人口数据以计算加权平均
population_data = pd.read_csv('API_SP.POP.TOTL_DS2_en_csv_v2_389828.csv', skiprows=4)
population_data = population_data.rename(columns={'Country Name': 'country', 'Country Code': 'country_code'})
population_2023 = population_data[['country', 'country_code', '2023']].dropna()
population_2023['2023'] = population_2023['2023'].astype(float)


# -------------------------------------------------------------------------------#
# 2. 预测2024-2030年老龄化百分比
# -------------------------------------------------------------------------------#
# 预测每个国家的2024-2030年老龄化百分比
future_years = list(range(2024, 2031))
for year in future_years:
    regression_summary[f'aging_{year}'] = regression_summary['slope'] * year + regression_summary['intercept']

# 计算全球加权平均老龄化百分比
global_aging = []
merged_df = regression_summary.merge(population_2023[['country', '2023']], on='country', how='inner')
for year in future_years:
    merged_df[f'weighted_aging_{year}'] = merged_df[f'aging_{year}'] * merged_df['2023']
    global_avg = merged_df[f'weighted_aging_{year}'].sum() / merged_df['2023'].sum()
    global_aging.append({'Year': year, 'Global Aging %': round(global_avg, 2)})

global_aging_df = pd.DataFrame(global_aging)

# 创建预测数据长格式用于绘图
future_predictions = regression_summary[['country'] + [f'aging_{year}' for year in future_years]].copy()
future_predictions = future_predictions.melt(id_vars='country', 
                                            value_vars=[f'aging_{year}' for year in future_years],
                                            var_name='year', 
                                            value_name='aging_percent')
future_predictions['year'] = future_predictions['year'].str.replace('aging_', '').astype(int)
future_predictions['type'] = 'Predicted'

# 合并历史和预测数据
historical_data = aging_data_long[['country', 'year', 'aging_percent']].copy()
historical_data['type'] = 'Historical'
combined_data = pd.concat([historical_data, future_predictions], ignore_index=True)

# -------------------------------------------------------------------------------#
# 2. Dash应用
# -------------------------------------------------------------------------------#
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# 布局
app.layout = dbc.Container([
    html.H1("Global Aging Population Dashboard", className="text-center my-4"),
    
    # 国家选择下拉菜单
    dbc.Row([
        dbc.Col([
            html.Label("Select Countries:"),
            dcc.Dropdown(
                id='country-dropdown',
                options=[{'label': country, 'value': country} for country in aging_data['country'].unique()],
                value=['China', 'United States', 'Malaysia',],  # 默认国家
                multi=True
            )
        ], width=6)
    ], className="mb-4"),
    
    # 时间序列图
    dbc.Row([
        dbc.Col([
            html.H3("Aging Population Trends (2000-2023)"),
            dcc.Graph(id='time-series-plot')
        ], width=12)
    ], className="mb-4"),
    
    # 聚类散点图和表格
    dbc.Row([
        dbc.Col([
            html.H3("2023 Aging Population Clustering"),
            dcc.Graph(id='cluster-plot')
        ], width=6),
        dbc.Col([
            html.H3("Cluster and Regression Summary"),
            html.Div(id='summary-table')
        ], width=6)
    ], className="mb-4"),
    
    # 世界地图
    dbc.Row([
        dbc.Col([
            html.H3("Global Aging Map (2023)"),
            dcc.Graph(id='world-map')
        ], width=12)
    ]),
    
    # 预测表格（2024-2030）
    dbc.Row([
        dbc.Col([
            html.H3("Predicted Aging Population (2024-2030)"),
            html.Div(id='prediction-table')
        ], width=12)
    ], className="mb-4"),
    
    # 全球加权平均老龄化百分比
    dbc.Row([
        dbc.Col([
            html.H3("Global Aging Population Forecast (2024-2030)"),
            html.Div(id='global-forecast-table')
        ], width=12)
    ])
], fluid=True)

# -------------------------------------------------------------------------------#
# 3. 回调函数
# -------------------------------------------------------------------------------#
@app.callback(
    Output('time-series-plot', 'figure'),
    Input('country-dropdown', 'value')
)
def update_time_series(selected_countries):
    df = aging_data_long[aging_data_long['country'].isin(selected_countries)]
    fig = px.line(df, x='year', y='aging_percent', color='country',
                  title='Aging Population (% of Total) Over Time',
                  labels={'aging_percent': '% Population Aged 65+', 'year': 'Year'})
    
    # 添加回归趋势线
    for country in selected_countries:
        reg_data = regression_summary[regression_summary['country'] == country]
        if not reg_data.empty:
            slope = reg_data['slope'].iloc[0]
            intercept = reg_data['intercept'].iloc[0]
            years_range = [2000, 2023]
            trend_y = [slope * year + intercept for year in years_range]
            fig.add_trace(go.Scatter(
                x=years_range, y=trend_y, mode='lines', name=f'{country} Trend',
                line=dict(dash='dash')
            ))
    fig.update_layout(showlegend=True, xaxis_title="Year", yaxis_title="% Population Aged 65+")
    return fig

@app.callback(
    Output('cluster-plot', 'figure'),
    Input('country-dropdown', 'value')
)
def update_cluster_plot(selected_countries):
    fig = px.scatter(cluster_summary, x='2023', y='cluster_label', color='cluster_label',
                     hover_data=['country', '2023'],
                     title='K-Means Clustering of 2023 Aging Population',
                     labels={'2023': '% Population Aged 65+', 'cluster_label': 'Cluster'})
    fig.update_traces(marker=dict(size=10))
    # 高亮选定国家
    selected_df = cluster_summary[cluster_summary['country'].isin(selected_countries)]
    if not selected_df.empty:
        fig.add_trace(go.Scatter(
            x=selected_df['2023'], y=selected_df['cluster_label'],
            mode='markers', marker=dict(size=15, symbol='circle', line=dict(width=2, color='black')),
            text=selected_df['country'], name='Selected Countries'
        ))
    return fig

@app.callback(
    Output('summary-table', 'children'),
    Input('country-dropdown', 'value')
)
def update_summary_table(selected_countries):
    table_df = cluster_summary[cluster_summary['country'].isin(selected_countries)][['country', '2023', 'cluster_label']]
    table_df = table_df.merge(regression_summary[['country', 'slope', 'r2']], on='country', how='left')
    table_df = table_df.rename(columns={'2023': 'Aging % (2023)', 'cluster_label': 'Cluster', 'slope': 'Trend Slope', 'r2': 'R²'})
    table = dbc.Table.from_dataframe(table_df, striped=True, bordered=True, hover=True)
    return table

@app.callback(
    Output('world-map', 'figure'),
    Input('country-dropdown', 'value')
)
def update_world_map(selected_countries):
    fig = px.choropleth(cluster_summary, locations='country_code', color='2023',
                        hover_name='country', hover_data=['cluster_label'],
                        title='Global Aging Population (2023)',
                        color_continuous_scale=px.colors.sequential.Plasma,
                        labels={'2023': '% Population Aged 65+'})
    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=True, projection_type='equirectangular'),
        margin={"r":0,"t":50,"l":0,"b":0}
    )
    return fig

@app.callback(
    Output('prediction-table', 'children'),
    Input('country-dropdown', 'value')
)
def update_prediction_table(selected_countries):
    table_df = regression_summary[regression_summary['country'].isin(selected_countries)][
        ['country'] + [f'aging_{year}' for year in future_years] + ['slope', 'r2']
    ]
    table_df = table_df.merge(cluster_summary[['country', 'cluster_label']], on='country', how='left')
    table_df = table_df.rename(columns={
        'cluster_label': 'Cluster',
        'slope': 'Trend Slope',
        'r2': 'R²',
        **{f'aging_{year}': f'{year}' for year in future_years}
    })
    for year in future_years:
        table_df[str(year)] = table_df[str(year)].round(2)
    table = dbc.Table.from_dataframe(table_df, striped=True, bordered=True, hover=True)
    return table

@app.callback(
    Output('global-forecast-table', 'children'),
    Input('country-dropdown', 'value')
)
def update_global_forecast_table(_):
    table = dbc.Table.from_dataframe(global_aging_df, striped=True, bordered=True, hover=True)
    return table

# -------------------------------------------------------------------------------#
# 4. 运行应用
# -------------------------------------------------------------------------------#
if __name__ == '__main__':
    app.run(debug=True, port=8080)