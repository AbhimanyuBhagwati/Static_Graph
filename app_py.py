import warnings
warnings.filterwarnings("ignore")
from dash import Dash, dcc, html, Input, Output, callback, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
from PIL import Image
import io
import base64
from sklearn.preprocessing import StandardScaler
import plotly.graph_objs as go
global filtered_df
from dash_table import DataTable
from sklearn.decomposition import PCA
import numpy as np
from scipy.stats import boxcox
import dash
from scipy import stats


FILE_PATH = r"/Users/abhimanyubhagwati/Documents/MachineLearning Book/FINAL_TERM_ROJECT/new_df.csv"
DATA_INFP = r"/Users/abhimanyubhagwati/Documents/MachineLearning Book/FINAL_TERM_ROJECT/zillow-prize-1/zillow_data_dictionary.xlsx"
MY_IMG_PATH = r"/Users/abhimanyubhagwati/Documents/MachineLearning Book/FINAL_TERM_ROJECT/Screenshot 2023-12-02 at 1.27.41 AM.png"
IMG_PATH_BEFORE_OUTLIER = r"/Users/abhimanyubhagwati/Documents/MachineLearning Book/FINAL_TERM_ROJECT/before_otlier.png"
IMG_PATH_AFTER_OUTLIER = r"/Users/abhimanyubhagwati/Documents/MachineLearning Book/FINAL_TERM_ROJECT/after_outlier.png"


rnm_clms = {
    'airconditioningtypeid': 'aircon_type',
    'architecturalstyletypeid': 'architectural_style',
    'basementsqft': 'basement_area',
    'bathroomcnt': 'num_bathrooms',
    'bedroomcnt': 'num_bedrooms',
    'buildingclasstypeid': 'building_class',
    'buildingqualitytypeid': 'building_quality',
    'calculatedbathnbr': 'calculated_bathrooms',
    'decktypeid': 'deck_type',
    'finishedfloor1squarefeet': 'first_floor_area',
    'calculatedfinishedsquarefeet': 'calculated_finished_area',
    'finishedsquarefeet12': 'finished_living_area',
    'finishedsquarefeet13': 'perimeter_living_area',
    'finishedsquarefeet15': 'total_finished_area',
    'finishedsquarefeet50': 'finished_living_area_first',
    'finishedsquarefeet6': 'base_unfinished_finished_area',
    'fips': 'fips_code',
    'fireplacecnt': 'num_fireplaces',
    'fullbathcnt': 'num_full_bathrooms',
    'garagecarcnt': 'num_garage',
    'garagetotalsqft': 'garage_total_area',
    'heatingorsystemtypeid': 'heating_system_type',
    'latitude': 'latitude',
    'longitude': 'longitude',
    'lotsizesquarefeet': 'lot_area',
    'poolcnt': 'num_pools',
    'poolsizesum': 'pool_total_area',
    'pooltypeid10': 'pool_spa',
    'pooltypeid2': 'pool_spa',
    'pooltypeid7': 'pool',
    'propertylandusetypeid': 'property_landuse_type',
    'rawcensustractandblock': 'census_tract_block_raw',
    'regionidcity': 'region_city_id',
    'regionidcounty': 'region_county_id',
    'regionidneighborhood': 'region_neighborhood_id',
    'regionidzip': 'region_zip_id',
    'roomcnt': 'num_rooms',
    'storytypeid': 'story_type',
    'threequarterbathnbr': 'num_three_quarter_bathrooms',
    'typeconstructiontypeid': 'construction_material_type',
    'unitcnt': 'num_units',
    'yardbuildingsqft17': 'patio_area',
    'yardbuildingsqft26': 'shed_building_area',
    'yearbuilt': 'year_built',
    'numberofstories': 'num_stories',
    'structuretaxvaluedollarcnt': 'tax_building_value',
    'taxvaluedollarcnt': 'tax_total_value',
    'assessmentyear': 'assessment_year',
    'landtaxvaluedollarcnt': 'tax_land_value',
    'taxamount': 'tax_amount',
    'taxdelinquencyyear': 'tax_delinquency_year',
    'censustractandblock': 'census_tract_block'
}

data_info = pd.read_excel(DATA_INFP)
data = data_info.to_dict('records')
for i in range(len(data)):
    data[i]['Feature'] = data[i]['Feature'].replace("'", "")
    if data[i]['Feature'] in rnm_clms.keys():
        data[i]['Feature'] = rnm_clms[data[i]['Feature']]


data_info = pd.DataFrame(data)
new_df = pd.read_csv(FILE_PATH)
new_df['transactiondate'] = pd.to_datetime(new_df['transactiondate'], format='%Y-%m-%d')
new_df['month'] = new_df['transactiondate'].dt.month
new_df['year'] = new_df['transactiondate'].dt.year
new_df['month'] = new_df['month'].astype('int64')
new_df['year'] = new_df['year'].astype('int64')
new_df.drop(['transactiondate'], axis=1, inplace=True)
df = new_df.copy()
num_cols = new_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
_bd_cnt_mnthly = new_df.groupby('month')['num_bedrooms'].value_counts()
_bd_cnt_mnthly = _bd_cnt_mnthly.to_frame()
_bd_cnt_mnthly.reset_index(inplace=True)
list_of_cols_range_x = ["num_rooms", "month", "num_full_bathrooms", "num_bedrooms"]
new_df['house_age'] = new_df['year'] - new_df['year_built']
new_df['house_age'] = new_df['house_age'].astype('int64')
list_of_cols_y = new_df.columns.tolist()
for i in list_of_cols_range_x:
    list_of_cols_y.remove(i)
def get_otl_df(temp):
    for column_name in temp.columns:
        if pd.api.types.is_numeric_dtype(temp[column_name]):
            Q1 = temp[column_name].quantile(0.25)
            Q3 = temp[column_name].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            temp = temp[(temp[column_name] >= lower_bound) & (temp[column_name] <= upper_bound)]

    return temp

def calculate_statistics(column_data):
    mean_val = round(column_data.mean(), 2)
    median_val = round(column_data.median(), 2)
    std_dev = round(column_data.std(), 2)
    return mean_val, median_val, std_dev

mnth_list = _bd_cnt_mnthly['month'].unique().tolist()
mnth_list.sort()
_out_rm_df = get_otl_df(new_df)
fts = _out_rm_df.columns.tolist()
X = _out_rm_df[fts[:-1]]
scaler = StandardScaler()
X = scaler.fit_transform(X)
pca = PCA()
pca_result = pca.fit_transform(X)


cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
yr_name = {
    1: 'January',
    2: 'February',
    3: 'March',
    4: 'April',
    5: 'May',
    6: 'Jun',
    7: 'July',
    8: 'August',
    9: 'September',
    10: 'October',
    11: 'November',
    12: 'December'
}
dd_graph_list = [
    'bar',
    'line',
    'scatter',
    'pie',
    'box',
    'histogram',
    'violin',
    'strip',
    'density_contour',
    'density_mapbox',
]

_bd_cnt_mnthly = new_df.groupby('month')['num_bedrooms'].value_counts()
_bd_cnt_mnthly = _bd_cnt_mnthly.to_frame()
_bd_cnt_mnthly.reset_index(inplace=True)
app = Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
server = app.server
tabs_styles = {'zIndex': 99, 'display': 'inlineBlock', 'height': '4vh', 'width': '12vw',
               'position': 'fixed', "background": "#323130", 'top': '12.5vh', 'left': '7.5vw',
               'border': 'grey', 'border-radius': '4px'}

tab_style = {
    "background": "#323130",
    'text-transform': 'uppercase',
    'color': 'white',
    'border': 'grey',
    'font-size': '11px',
    'font-weight': 600,
    'align-items': 'center',
    'justify-content': 'center',
    'border-radius': '4px',
    'padding':'6px'
}

tab_selected_style = {
    "background": "grey",
    'text-transform': 'uppercase',
    'color': 'white',
    'font-size': '11px',
    'font-weight': 600,
    'align-items': 'center',
    'justify-content': 'center',
    'border-radius': '4px',
    'padding':'6px'
}
app.layout = html.Div([
html.H1("Information Visualization CS 5764 FINAL TERM PROJECT", style={'text-align': 'center', 'fontSize': 70,
               'font-weight': 'bold'}),
    dcc.Tabs(id='tabs', value='tab1', children=[
        dcc.Tab(label='About Project', value='info',style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Dataset Description', value='data_desc',style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Outlier Visualization', value='outlier',style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Pca Analysis', value='pca',style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Bar Graph', value='tab1',style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Heatmap', value='tab2',style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Scatter Plot', value='tab3',style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Multi interactive Graph', value='tab4',style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Image Download', value='tab5',style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='BoxCox Test', value='normality',style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='Normality Test', value='all_normality',style=tab_style, selected_style=tab_selected_style),
        dcc.Tab(label='About Me', value='about_me',style=tab_style, selected_style=tab_selected_style),
    ],style=tab_style),

    html.Div(id='tabs-content'),
    html.Br(),
html.Footer("Made By: Abhi (Abhimanyu Bhagwati) ", style={'text-align': 'center', 'padding': '10px', 'background': '#f0f0f0', 'position': 'fixed', 'bottom': '0', 'width': '100%'})
])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))
def update_tab_content(selected_tab):
    if selected_tab == 'info':
        return info_layout
    elif selected_tab == 'data_desc':
        return data_desc
    elif selected_tab == 'outlier':
        return outlier_layout
    elif selected_tab == 'tab1':
        return tab1_layout
    elif selected_tab == 'pca':
        return pca_layout
    elif selected_tab == 'tab2':
        return tab2_layout
    elif selected_tab == 'tab3':
        return tab3_layout
    elif selected_tab == 'tab4':
        return tab4_layout
    elif selected_tab == 'tab5':
        return tab5_layout
    elif selected_tab == 'normality':
        return normality_layout
    elif selected_tab == 'all_normality':
        return all_normality_layout
    elif selected_tab == 'about_me':
        return about_me_layout

all_normality_layout = html.Div([
    html.H3('Normality Tests', style={'text-align': 'center', 'fontSize': 70,
               'font-weight': 'bold'}),
    html.Br(),
    html.H4('Select a Test', style={'fontSize': 50}),
    dcc.Dropdown(
        id='test-dropdown',
        options=[
            {'label': "D'Agostino's K-squared test", 'value': 'dagostino'},
            {'label': 'Kolmogorov-Smirnov test', 'value': 'kolmogorov'},
            {'label': 'Shapiro-Wilk test', 'value': 'shapiro'}
        ],
        value='dagostino',
style={'width': '70%', 'height': '50px', 'fontSize': 30}
    ),
    html.Br(),
    html.H4('Dependent Variable', style={'fontSize': 50}),
    html.Br(),
    dcc.Dropdown(
        id='dependent-dropdown',
        options=[{'label': i, 'value': i} for i in df.columns],
        value=df.columns[1],
style={'width': '70%', 'height': '50px', 'fontSize': 30}
    ),
html.Br(),
    html.H4('Independent Variable', style={'fontSize': 50}),
    html.Br(),
    dcc.Dropdown(
        id='independent-dropdown',
        options=[{'label': i, 'value': i} for i in df.columns],
        value=df.columns[1],
        style={'width': '70%', 'height': '50px', 'fontSize': 30}
    ),
    html.Div(id='test-result', style={'margin-top': 20, 'font-size': 30})
])

# Define the callback function
@app.callback(
    Output('test-result', 'children'),
    [Input('test-dropdown', 'value'),
     Input('dependent-dropdown', 'value'),
     Input('independent-dropdown', 'value')]
)
def perform_test(test, dependent, independent):
    if test == 'dagostino':
        k2, p = stats.normaltest(df[dependent])
        return f"D'Agostino's K-squared test on {dependent}: K^2 = {k2}, p = {p}"
    elif test == 'kolmogorov':
        d, p = stats.kstest(df[dependent], 'norm')
        return f"Kolmogorov-Smirnov test on {dependent}: D = {d}, p = {p}"
    elif test == 'shapiro':
        W, p = stats.shapiro(df[dependent])
        return f"Shapiro-Wilk test on {dependent}: W = {W}, p = {p}"


pca_layout = html.Div([
    html.H3('PCA Analysis', style={'text-align': 'center', 'fontSize': 70,
               'font-weight': 'bold'}),
    html.P("Select Number of Components:", style={'fontSize': 30,}),
    dcc.Slider(
        id='no-component-slider',
        min=1,
        max=len(cumulative_explained_variance),
        step=1,
        value=8,
    ),
    html.Br(),
    dcc.Loading(
        id="loading-graph",
        type="circle",
        children=[
            dcc.Graph(id='PCA-graph')
        ]
    ),
])

@app.callback(
    Output(component_id='PCA-graph', component_property='figure'),
    [Input(component_id='no-component-slider', component_property='value')],
)
def plot_pca(no_component):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(1, no_component + 1),
                             y=cumulative_explained_variance[:no_component],
                             mode='lines',
                             name='Cumulative Explained Variance',
                             ))
    threshold_component = np.where(cumulative_explained_variance >= 0.95)[0][0] + 1
    if threshold_component <= no_component:
        fig.add_shape(
            go.layout.Shape(
                type='line',
                x0=1,
                x1=len(cumulative_explained_variance),
                y0=0.95,
                y1=0.95,
                line=dict(color='red', width=6, dash='dash'),
                name='95% Threshold'
            )
        )
        fig.add_shape(
            go.layout.Shape(
                type='line',
                x0=threshold_component,
                x1=threshold_component,
                y0=0,
                y1=1,
                line=dict(color='red', width=6, dash='dash',),
                name=f'Component #{threshold_component}'
            )
        )
    # Customize layout
    fig.update_layout(
        title='PCA Analysis',
        title_font_size=40,
        title_font_family='sans-serif',
        title_font_color='white',
        template='plotly_dark',
        xaxis=dict(title='Number of components'),
        yaxis=dict(title='Cumulative explained variance'),
        xaxis_showgrid=True,
        yaxis_showgrid=True,
        showlegend=True,
    )
    return fig

data_desc = html.Div([
    html.H1("Dataset Description", style={'text-align': 'center', 'fontSize': 70,
               'font-weight': 'bold'}),
    DataTable(
        id='table',
        columns=[{'name': col, 'id': col} for col in data_info.columns],
        data=data_info.to_dict('records'),
        style_table={'height': '1000px', 'overflowY': 'auto', 'overflowX': 'auto',
                        'width': '100%', 'minWidth': '100%', 'maxWidth': '100%'},
        style_cell={'textAlign': 'left', 'padding': '30px', 'whiteSpace': 'auto',
                    'fontSize': 42, 'font-family': 'sans-serif', 'color': 'black',
                    },
        style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white', 'fontWeight': 'bold'},
    ),
])
_clm_list = new_df.columns.tolist()
normality_layout =dcc.Loading(
    id="loading-layout",
    type="default",
    children=[
        html.Div([
            html.H1('Box-Cox Transformation', style={'text-align': 'center', 'fontSize': 70, 'font-weight': 'bold'}),

            dcc.Dropdown(
                id='column-dropdown',
                options=[{'label': col, 'value': col} for col in _clm_list[3:]],
                value=df.columns[4],
                style={'fontSize': 30, 'font-weight': 'bold', 'margin-bottom': '10px'}
            ),

            dcc.Dropdown(
                id='dataframe-dropdown',
                style={'width': '0%', 'fontSize': 0, 'margin-bottom': '0px',
                       'background-color': 'rgba(255, 255, 255, 0.1)',
                       'position': 'absolute',
                       'right': '0',
                       'bottom': '0'
                       }
            ),

            html.Button('Show Original Graph', id='original-button', n_clicks=0, style={'fontSize': 30, 'margin-right': '20px'}),
            html.Button('Show Transformed Graph', id='transformed-button', n_clicks=0, style={'fontSize': 30, 'margin-bottom': '20px'}),

            dcc.Graph(id='data-plot'),

            html.Div(id='lambda-display', style={'margin-top': 20, 'fontSize': 30, 'font-weight': 'bold'}),
html.Br(),
        html.H3("Normality Test",style={'text-align': 'center', 'fontSize': 30,
                'font-weight': 'bold'}),
        html.Br(),
        html.H3("Implemented the Box-Cox transformation, a statistical technique that stabilizes variance and approximates a normal distribution. The page features two buttons allowing users to visualize the dataset before and after transformation, providing insights into the impact of the Box-Cox technique.", style={'text-align': 'center', 'fontSize': 24 }),

        ])
    ],
    style={'margin-top': '50px'}  # Adjust the margin as needed
)

# Define callback to update graph based on button selection
@app.callback(
    [Output('data-plot', 'figure'),
     Output('lambda-display', 'children')],
    [Input('original-button', 'n_clicks'),
     Input('transformed-button', 'n_clicks')],
    [State('column-dropdown', 'value'),
     State('dataframe-dropdown', 'value')]
)
def update_plot(original_clicks, transformed_clicks, selected_column, selected_dataframe):
    ctx = dash.callback_context
    if ctx.triggered_id == 'original-button':
        column_data = new_df[selected_column]
        print(f"Original Data: {column_data.head()}")
        fig = px.histogram(new_df, x=column_data, title=f'Original {selected_column}', template='plotly_dark',
        height=700)
        fig.update_layout(
            title_font_size=40,
            title_font_family='sans-serif',
            title_font_color='white',
            template='plotly_dark',
            xaxis=dict(title=selected_column),
            yaxis=dict(title='Count'),
            xaxis_showgrid=True,
            yaxis_showgrid=True,
            showlegend=True,
        )

        return fig, ''

    elif ctx.triggered_id == 'transformed-button':
        column_data = new_df[selected_column]
        transformed_data, lambda_value = boxcox(column_data)
        print(f"Transformed Data: {transformed_data[:5]}")
        fig = px.histogram(x=transformed_data, title=f'Box-Cox Transformed {selected_column}', template='plotly_dark')
        fig.update_layout(
            title_font_size=40,
            title_font_family='sans-serif',
            title_font_color='white',
            template='plotly_dark',
            xaxis=dict(title=selected_column),
            yaxis=dict(title='Count'),
            xaxis_showgrid=True,
            yaxis_showgrid=True,
            showlegend=True,
        )
        return fig, f'Box-Cox Lambda: {lambda_value:.4f}'
    return px.histogram(), ''


outlier_layout = html.Div([
    html.H1("Outlier Visualization", style={'text-align': 'center', 'fontSize': 70}),
    html.Br(),
    dcc.Tabs([
        dcc.Tab(label='Before Outlier Removal', style=tab_style, selected_style=tab_selected_style, children=[
            html.Br(),
            html.Div([
                dcc.Dropdown(
                    id='feature-dropdown-before',
                    options=[{'label': col, 'value': col} for col in df.columns],
                    value='target',
                    multi=False,
                    style={'fontSize': 25}
                ),
                html.Br(),
                html.Button('Display Boxplot', id='btn-display-boxplot-before', n_clicks=0, style={'margin-left': '10px', 'fontSize': 25}),
            ]),
            html.Br(),
            dcc.Graph(
                id='boxplot-before-outlier-removal',
            ),
        ],),
        dcc.Tab(label='After Outlier Removal',style=tab_style, selected_style=tab_selected_style, children=[
            html.Br(),
            html.Div([
                dcc.Dropdown(
                    id='feature-dropdown-after',
                    options=[{'label': col, 'value': col} for col in df.columns],
                    value='target',
                    multi=False,
                    style={'fontSize': 25}
                ),
                html.Br(),
                html.Button('Remove Outliers', id='btn-remove-outliers-after', n_clicks=0, style={'margin-left': '10px', 'fontSize': 25}),
            ]),
            html.Br(),
            dcc.Graph(
                id='boxplot-after-outlier-removal',
            ),
        ],),
    ],style=tab_style),
])

@app.callback(
    Output('boxplot-before-outlier-removal', 'figure'),
    [Input('btn-display-boxplot-before', 'n_clicks')],
    [State('feature-dropdown-before', 'value')]
)
def display_boxplot_before(n_clicks, selected_feature):
    temp_df = df.copy()  # Make a copy to avoid modifying the original DataFrame

    # Create box plot for the selected feature
    fig = {
        'data': [
            go.Box(y=temp_df[selected_feature], name=selected_feature),
            go.Layout(template='plotly_dark', height=700),

        ],
        'layout': go.Layout(title=f'Boxplot - Before Outlier Removal ({selected_feature})'),
    }


    return fig

@app.callback(
    Output('boxplot-after-outlier-removal', 'figure'),
    [Input('btn-remove-outliers-after', 'n_clicks')],
    [State('feature-dropdown-after', 'value')]
)
def remove_outliers_after(n_clicks, selected_feature):
    temp_df = df.copy()  # Making a copy to avoid modifying the original DataFrame

    if n_clicks > 0:
        temp_df = get_otl_df(temp_df)

    fig = {
        'data': [
            go.Box(y=temp_df[selected_feature], name=selected_feature),
            go.Layout(template='plotly_dark',height=700,)
        ],
        'layout': go.Layout(title=f'Boxplot - After Outlier Removal ({selected_feature})' if n_clicks > 0 else f'Boxplot - Before Outlier Removal ({selected_feature})'),
    }
    return fig


info_layout = html.Div([
    html.H1("About Project", style={'text-align': 'center', 'fontSize': 70,
                'font-weight': 'bold'}),
    html.Br(),
    html.Div([
        html.H3("Dataset: Zillow Prize: Zillow’s Home Value Prediction (Zestimate)", style={'text-align': 'center', 'fontSize': 30,
                'font-weight': 'bold'}),
        html.H3("Link: https://www.kaggle.com/c/zillow-prize-1", style={'text-align': 'center', 'fontSize': 30,
                'font-weight': 'bold'}),
html.H3(
    [
        "Kaggle Link: ",
        html.A("Zillow Prize: Zillow’s Home Value Prediction (Zestimate)", href="https://www.kaggle.com/c/zillow-prize-1"),
    ],
    style={'text-align': 'center', 'fontSize': 30}
),

    ]),
    html.Div([
        html.H3("Dataset Description", style={'text-align': 'center', 'fontSize': 30,
                'font-weight': 'bold'}),
        html.Br(),
        html.H3("The dataset consists of two main files: properties_2016.csv and train_2016_v2.csv. properties_2016.csv contains property information for various parcels. train_2016_v2.csv includes transactions data for properties with sales in the year 2016.", style={'text-align': 'center', 'fontSize': 24 }),
        html.Br(),
        html.H3("Objective:",style={'text-align': 'center', 'fontSize': 30,
                'font-weight': 'bold'}),
        html.Br(),
        html.H3("In the context of the Zillow Prize competition, where we are dealing with property data, EDA helps us better comprehend the distribution of important variables like home prices, square footage, and the number of bedrooms and bathrooms.", style={'text-align': 'center', 'fontSize': 24 }),
        html.Br(),
        html.H3("Iterative Development Journey: Unveiling the Enhanced Version of Our Application",style={'text-align': 'center', 'fontSize': 30,}),
html.Br(),
html.H3(
            [
                "LinkedIn Post link: ",
                html.A("Post link",
                       href="https://www.linkedin.com/posts/abhimanyubhagwati_dash-activity-7134822917019713538-ADb1?utm_source=share&utm_medium=member_desktop"),
            ],
            style={'text-align': 'center', 'fontSize': 30}
        ),
        html.Br(),
html.H3(
            [
                "App Link: ",
                html.A("App",
                       href="https://dashapp-nd335decna-nn.a.run.app/"),
            ],
            style={'text-align': 'center', 'fontSize': 30}
        )
    ]),
])



tab1_layout = html.Div([
    html.Br(),
    html.H1(id='heading', style={'text-align': 'center', 'fontSize': 70, 'font-weight': 'bold'}),
    html.Div(id='slider-output-container'),
    html.Br(),
    dcc.Graph(id='graph', style={'width': '100%', 'display': 'flex', 'align-items': 'center'}),
    html.Br(),
    dcc.Slider(
        id='mth_slider',
        min=mnth_list[0],
        max=mnth_list[-1],
        step=None,
        marks={str(mnth): {'label': yr_name[mnth], 'style': {'fontSize': 25, 'fontWeight': 'bold',
                                                                'margin-left': '20px'

                                                             }} for mnth in mnth_list},
        value=mnth_list[0],
    ),
html.Br(),
        html.H3("Bar Graph Page",style={'text-align': 'center', 'fontSize': 30,
                'font-weight': 'bold'}),
        html.Br(),
        html.H3("This page features a bar plot displaying the popularity of different numbers of bedrooms for sale. A slider allows users to navigate through months to observe variations in popularity.", style={'text-align': 'center', 'fontSize': 24 }),

])

@app.callback(
    [Output('graph', 'figure'),
     Output('heading', 'children')],
    [Input('mth_slider', 'value')]
)
def update_figure(selected_mnth):
    filtered_df = _bd_cnt_mnthly[_bd_cnt_mnthly.month == selected_mnth]
    popular_room = filtered_df.loc[filtered_df['count'].idxmax()]['num_bedrooms']

    fig = px.bar(filtered_df, x="num_bedrooms", y="count", color="count", template='plotly_dark',
                  height=700
                 )
    fig.update_layout(
        xaxis_title="Number of Bedrooms",
        yaxis_title="Count",
        transition_duration=500
    )

    heading_text = f"Sale of number of Bedrooms in {yr_name[selected_mnth]} - Most Popular No. of Room: {popular_room}"
    return fig, heading_text
tab2_layout = html.Div([
    html.H1("Heatmap", style={'text-align': 'center',
                              'color': '#black',
                              'font-size': '100px',
                              'font-weight': 'bold',
                              }),
    dcc.Loading(
        id="loading",
        type='circle',
        children=[
            dcc.Graph(id='corr_graph',
                      style={'height': '70vh',
                             'width': '70vw',
                             'display': 'inline-block',
                             'padding': '0 20',
                             'float': 'left'
                             }
                      ),
        ],
        style={"width": "800px", "height": "900px", "margin": "auto"},
    ),
    html.Div(
        id='Selected_Featured', style={'fontSize': 20, 'margin-top': '20px', 'font-weight': 'bold'}),
html.H3("Select Features",style={'text-align': 'right',
                                         'color': '#black',
                                         'font-size': '50px',
                                         'font-weight': 'bold',
                                         }),
    dbc.Tooltip("Hover for checklist tooltip", target="ft_checklist", placement="bottom"),
    dcc.Checklist(
        id='ft_checklist',
        options=[{'label': i, 'value': i} for i in num_cols],
        value=num_cols[:],
        style={'fontSize': 40,
               'font-weight': 'bold',
               'float': 'right',
               'margin-top': '20px'
               }
    ),
])


@app.callback(
    [Output('corr_graph', 'figure'),
     Output('Selected_Featured', 'children')],
    Input('ft_checklist', 'value')
)
def update_graph(ft_checklist):
    selected_df = new_df[ft_checklist]
    corr_df = selected_df.corr()
    fig = px.imshow(corr_df,
                    labels=dict(x='Features', y='Features', color='Correlation'
                                ),
                    x=selected_df.columns,
                    y=selected_df.columns,
                    color_continuous_scale='Viridis',
                    template='plotly_dark',
                    )
    selected_features_text = f"Selected Features: {', '.join(ft_checklist)}"

    return fig, selected_features_text

tab3_layout = html.Div([
    html.H1("Scatter Plot", style={'text-align': 'center', 'fontSize': 70,
               'font-weight': 'bold'}),
    html.Br(),
    html.Label("Select Column for X-axis:", style={'fontSize': 30,
               'font-weight': 'bold'}),
    dcc.RadioItems(
        id='x-axis-radio',
        options=[{'label': col, 'value': col} for col in list_of_cols_range_x],
        value=list_of_cols_range_x[0],
        labelStyle={'display': 'block'},
        style={'fontSize': 30}
    ),
    html.Br(),
    html.Label("Select Columns for Y-axis:", style={'text-align': 'center', 'fontSize': 30,
               'font-weight': 'bold'}),
    dcc.Dropdown(
        id='y-axis-dropdown',
        options=[{'label': col, 'value': col} for col in list_of_cols_y],
        value=[list_of_cols_y[1]],
        multi=True,
        style={'fontSize': 25}
    ),
    html.Br(),
    html.Label("Select Range for X-axis:", style={'text-align': 'center', 'fontSize': 30, 'font-weight': 'bold'}),
    dcc.RangeSlider(
        id='x-axis-range',
        min=new_df[list_of_cols_range_x[0]].min(),
        max=new_df[list_of_cols_range_x[0]].max(),
        step=1,
        marks={i: str(i) for i in range(int(new_df[list_of_cols_range_x[0]].min()), int(new_df[list_of_cols_range_x[0]].max())+1)},
        value=[new_df[list_of_cols_range_x[0]].min(), new_df[list_of_cols_range_x[0]].max()]
    ),
    html.Br(),
    html.Br(),
    dcc.Loading(
        id='loading-output',
        type='circle',
        children=[
            dcc.Graph(id='dynamic-graph'),
        html.Button("Download CSV", id="btn_csv",style={'position': 'absolute', 'top': '400px', 'right': '10px', 'fontSize': 25}),
        dcc.Download(id="download-dataframe-csv"),
        ],
    ),
html.Br(),
        html.H3("Scatter Plot",style={'text-align': 'center', 'fontSize': 30,
                'font-weight': 'bold'}),
        html.Br(),
        html.H3("The scatter plot functionality enables users to plot any column value (Y-axis) against various parameters such as month, number of rooms, total bathrooms, and number of bedrooms (X-axis). This provides a comprehensive view of the popularity of different categories.", style={'text-align': 'center', 'fontSize': 24 }),
    html.Br(),
    html.Br()

])

@app.callback(
    [Output('x-axis-range', 'min'),
     Output('x-axis-range', 'max')],
    [Input('x-axis-radio', 'value')]
)
def update_slider_range(selected_x_column):
    return new_df[selected_x_column].min(), new_df[selected_x_column].max()

@app.callback(
    Output('dynamic-graph', 'figure'),
    [Input('y-axis-dropdown', 'value'),
     Input('x-axis-range', 'value'),
     Input('x-axis-radio', 'value')]
)
def update_graph(selected_y_columns, x_range, selected_x_column):
    filtered_data = new_df[(new_df[selected_x_column] >= x_range[0]) & (new_df[selected_x_column] <= x_range[1])]
    global filtered_df
    filtered_df = filtered_data
    figure = px.scatter(filtered_data, x=selected_x_column, y=selected_y_columns,height=700,
                        title=f'{", ".join(selected_y_columns)} vs {selected_x_column}',
                        labels={selected_x_column: f'{selected_x_column}', 'value': 'Y-axis',},
                        template='plotly_dark')
    figure.update_layout(
title_font_size=40,
        title_font_family='sans-serif',
        title_font_color='white',
        template='plotly_dark',
        xaxis=dict(title=selected_x_column),
        yaxis=dict(title='Y-axis'),
        xaxis_showgrid=True,
        yaxis_showgrid=True,
        showlegend=True,
    )
    return figure

@callback(
    Output("download-dataframe-csv", "data"),
    Input("btn_csv", "n_clicks"),
    prevent_initial_call=True,
)
def func(n_clicks):
    return dcc.send_data_frame(filtered_df.to_csv, "mydf.csv")

my_img = Image.open(MY_IMG_PATH)
my_img_byte_array = io.BytesIO()
my_img.save(my_img_byte_array, format='PNG')
my_img_base64 = 'data:image/png;base64,' + str(base64.b64encode(my_img_byte_array.getvalue()), 'utf-8')


about_me_layout = html.Div([
    html.H1("About Me", style={'text-align': 'center', 'fontSize': 70,
               'font-weight': 'bold'}),
    html.Br(),
    html.Div([
        html.H3("Name: Abhimanyu Bhagwati", style={'text-align': 'center', 'fontSize': 30,
               'font-weight': 'bold'}),
        html.H3("Email: abhimanyu@vt.edu", style={'text-align': 'center', 'fontSize': 30,
               'font-weight': 'bold'}),
        #add hyperlink to linkedin
        html.H3(
            [
                "LinkedIn: ",
                html.A("Abhimanyu Bhagwati",
                       href="https://www.linkedin.com/in/abhimanyubhagwati/"),
            ],
            style={'text-align': 'center', 'fontSize': 30}
        ),        html.H3("Major: Computer Science", style={'text-align': 'center', 'fontSize': 30,
                'font-weight': 'bold'}),
        html.Img(src=my_img_base64, style={'width': '20%', 'display': 'flex',
                                           #align image top left
                                              'align-items': 'left', 'margin-left': '10px',},),
        html.H1("This is Me, Thanks for Visiting", style={'align-items': 'left', 'margin-left': '10px', 'fontSize': 70,
               'font-weight': 'bold'}),
    ])
])


img_before_outlier = Image.open(IMG_PATH_BEFORE_OUTLIER)
img_after_outlier = Image.open(IMG_PATH_AFTER_OUTLIER)

img_before_outlier_byte_array = io.BytesIO()
img_before_outlier.save(img_before_outlier_byte_array, format='PNG')
img_before_outlier_base64 = 'data:image/png;base64,' + str(base64.b64encode(img_before_outlier_byte_array.getvalue()), 'utf-8')

img_after_outlier_byte_array = io.BytesIO()
img_after_outlier.save(img_after_outlier_byte_array, format='PNG')
img_after_outlier_base64 = 'data:image/png;base64,' + str(base64.b64encode(img_after_outlier_byte_array.getvalue()), 'utf-8')

image_dropdown_value = 'before_outlier'  # Add a variable to store the selected image

tab5_layout = html.Div([
    html.H1("QQ plot Image", style={'text-align': 'center', 'fontSize': 70, 'font-weight': 'bold'}),
    html.Br(),
    dcc.Dropdown(
        id='image-dropdown',
        options=[
            {'label': 'Before Outlier', 'value': 'before_outlier'},
            {'label': 'After Outlier', 'value': 'after_outlier',}
        ],
        value='before_outlier',
        style={'height': '50px', 'fontSize': 30}
    ),
    html.Br(),
    html.Div([
        html.Img(id='selected-image', style={'width': '50%'}),
        html.Button("Download Image", id='btn-download', n_clicks=0),
        dcc.Download(id="download-image")
    ]),
])

@app.callback(
    Output('selected-image', 'src'),
    Input('image-dropdown', 'value')
)
def update_image(selected_image):
    global image_dropdown_value
    image_dropdown_value = selected_image

    if selected_image == 'before_outlier':
        return img_before_outlier_base64
    elif selected_image == 'after_outlier':
        return img_after_outlier_base64
    else:
        return ""


@app.callback(
    Output("download-image", "data"),
    Input("btn-download", "n_clicks"),
    prevent_initial_call=True
)
def download_image(n_clicks):
    global image_dropdown_value

    if n_clicks > 0:
        if image_dropdown_value == 'before_outlier':
            img_data = img_before_outlier_byte_array.getvalue()
        elif image_dropdown_value == 'after_outlier':
            img_data = img_after_outlier_byte_array.getvalue()
        else:
            raise PreventUpdate

        return dcc.send_bytes(img_data, f"image_{image_dropdown_value}.png")

tab4_layout =html.Div([
    html.H1("Interactive Plotting", style={'text-align': 'center', 'fontSize': 70, 'font-weight': 'bold'}),
    html.Label("Select Graph Type:", style={'fontSize': 30, 'font-weight': 'bold'}),
    dcc.Dropdown(
        id='graph-type-dropdown',
        options=[{'label': graph_type, 'value': graph_type} for graph_type in dd_graph_list],
        value='scatter',
        style={'fontSize': 40, 'margin-bottom': '20px', 'margin-top': '10px', 'width': '100%'}
    ),

    html.Label("Select X-axis:", style={'fontSize': 30, 'font-weight': 'bold'}),
    dcc.Dropdown(
        id='x-axis-dropdown',
        options=[{'label': col, 'value': col} for col in new_df.columns],
        value=new_df.columns[0],
        style={'fontSize': 40, 'margin-bottom': '20px', 'margin-top': '10px', 'width': '100%'}
    ),

    html.Label("Select Y-axis:", style={'fontSize': 30, 'font-weight': 'bold'}),
    dcc.Dropdown(
        id='y-axis-dropdown',
        options=[{'label': col, 'value': col} for col in new_df.columns],
        value=new_df.columns[1],
        style={'fontSize': 40, 'margin-bottom': '20px', 'margin-top': '10px', 'width': '100%'}
    ),
    html.Br(),
dcc.Loading(
                    id="loading",
                    type="default",
                    children=[
                        dcc.Graph(id='selected-plot')
                    ]
                ),
html.Br(),
        html.H3("Multi-Interactive Graph - Open Play Group for Plot",style={'text-align': 'center', 'fontSize': 30,
                'font-weight': 'bold'}),
        html.Br(),
        html.H3("This feature involves three drop-down menus. The first menu allows users to select the type of graph, the second lets them choose the feature for the X-axis, and the third for the Y-axis. This enables users to dynamically plot and explore relationships between different data points based on their preferences.", style={'text-align': 'center', 'fontSize': 24 }),

])

@app.callback(
    Output('selected-plot', 'figure'),
    [Input('graph-type-dropdown', 'value'),
     Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value')]
)
def update_selected_plot(graph_type, x_axis_column, y_axis_column):

    if graph_type == 'scatter':
        fig = px.scatter(new_df, x=new_df[x_axis_column], y=new_df[y_axis_column],height=700, labels={'x': x_axis_column, 'y': y_axis_column}, template='plotly_dark')
    elif graph_type == 'bar':
        fig = px.bar(new_df, x=new_df[x_axis_column], y=new_df[y_axis_column],height=700, labels={'x': x_axis_column, 'y': y_axis_column}, template='plotly_dark')
    elif graph_type == 'line':
        fig = px.line(new_df, x=new_df[x_axis_column], y=new_df[y_axis_column],height=700, labels={'x': x_axis_column, 'y': y_axis_column}, template='plotly_dark')
    elif graph_type == 'pie':
        fig = px.pie(new_df, values=new_df[y_axis_column], names=new_df[x_axis_column],height=700, title=f'{graph_type} plot', template='plotly_dark')
    elif graph_type == 'box':
        fig = px.box(new_df, x=new_df[x_axis_column], y=new_df[y_axis_column],height=700, labels={'x': x_axis_column, 'y': y_axis_column}, template='plotly_dark')
    elif graph_type == 'histogram':
        fig = px.histogram(new_df, x=new_df[x_axis_column], y=new_df[y_axis_column],height=700, labels={'x': x_axis_column, 'y': y_axis_column}, template='plotly_dark')
    elif graph_type == 'violin':
        fig = px.violin(new_df, x=new_df[x_axis_column], y=new_df[y_axis_column],height=700, labels={'x': x_axis_column, 'y': y_axis_column}, template='plotly_dark')
    elif graph_type == 'strip':
        fig = px.strip(new_df, x=new_df[x_axis_column], y=new_df[y_axis_column],height=700, labels={'x': x_axis_column, 'y': y_axis_column}, template='plotly_dark')
    elif graph_type == 'density_contour':
        fig = px.density_contour(new_df, x=new_df[x_axis_column], y=new_df[y_axis_column],height=700, labels={'x': x_axis_column, 'y': y_axis_column}, template='plotly_dark')
    else:
        return "Invalid graph type"
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8080)
