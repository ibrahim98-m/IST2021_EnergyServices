import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_table
import base64

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
df = pd.read_csv('raw_data_total.csv')

df6 = pd.read_csv('data_NT.csv')
available_years = df6['year'].unique()

df1 = pd.read_csv('North_Tower_Total_Hourly_Model.csv')

image_filename = 'linear regression.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

image_filename2 = 'random forest.png' # replace with your own image
encoded_image2 = base64.b64encode(open(image_filename2, 'rb').read())

image_filename3 = 'extrem gradient boosting.png' # replace with your own image
encoded_image3 = base64.b64encode(open(image_filename3, 'rb').read())

image_filename4 = 'neural networks.png' # replace with your own image
encoded_image4 = base64.b64encode(open(image_filename4, 'rb').read())

image_filename5 = 'Clust_image3.png' # replace with your own image
encoded_image5 = base64.b64encode(open(image_filename5, 'rb').read())

image_filename6 = 'Clust_image4.png' # replace with your own image
encoded_image6 = base64.b64encode(open(image_filename6, 'rb').read())

image_filename7 = 'Clust_image5.png' # replace with your own image
encoded_image7 = base64.b64encode(open(image_filename7, 'rb').read())

image_filename8 = 'Clust_image1.png' # replace with your own image
encoded_image8 = base64.b64encode(open(image_filename8, 'rb').read())

image_filename9 = '3D.png' # replace with your own image
encoded_image9 = base64.b64encode(open(image_filename9, 'rb').read())

image_filename10 = 'ist_logo.png' # replace with your own image
encoded_image10 = base64.b64encode(open(image_filename10, 'rb').read())

def generate_page():
        return html.Div(children=[
        html.Div([
            html.H3(children='''
                    Visualization of hourly electricity consumption at North Tower over the last years
                    '''),

            dcc.Graph(
                id='yearly-data',
                figure={
                    'data': [
                        {'x': df['Date'], 'y': df['Power_kW'], 'type': 'line', 'name': 'Power'},
                        {'x': df['Date'], 'y': df['Temperature (C)'], 'type': 'line', 'name': 'Temperature'},
                        
                        ],
                    'layout': {
                        'title': 'North Tower hourly electricity consumption (kWh)'
                        }
                    }
                ),
            ]),

            html.H2(children='   '),

            html.Div([
                html.H3(children='''
                     Visualization of total electricity consumption at North Tower over the last years
                     '''),

            dcc.Graph(
                id='yearly',
                figure={
                    'data': [
                        {'x': df6.year, 'y': df6.NorthTower, 'type': 'bar', 'name': 'North Tower'},
                        #{'x': df6.year, 'y': df6.Total, 'type': 'bar', 'name': 'Total'},
                        ],
                    'layout': {
                        'title': 'North Tower yearly electricity consumption (MWh)'
                        }
                    }
                ),
            
            html.H3(children='Summary Table'),

        dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in df6.columns],
                data=df6.to_dict('records'),
              
                )
        
      # html.Table([
       #     html.Thead(
       #         html.Tr([html.Th(col) for col in df6.columns])
        #        ),
         #   html.Tbody([
          #      html.Tr([
           #         html.Td(df6.iloc[i][col]) for col in df6.columns
            #        ]) for i in range(len(df6))
             #   ])
            #])
        
        ])
        ])
        
    

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server=app.server

app.layout = html.Div([
    html.Img(src='data:image/png;base64,{}'.format(encoded_image10.decode()), style={'height':'20%', 'width':'20%'}),
    html.H1('Project 2 : Dashboard - North Tower data'),
    html.H6('by Ibrahim Minta - ist1100838'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Raw data', value='tab-1'),
        dcc.Tab(label='Exploratory data analysis', value='tab-2'),
        dcc.Tab(label='Clustering', value='tab-4'),
        dcc.Tab(label='Feature selection', value='tab-3'),
        dcc.Tab(label='Regression', value='tab-5'),
    ]),
    html.Div(id='tabscontent')
])

@app.callback(Output('tabscontent', 'children'),
              Input('tabs', 'value'))

def render_content(tab):
    
    if tab == 'tab-2':
        return html.Div([
            generate_page()
        ]),

    elif tab == 'tab-1':
        return    html.Div([
            html.H3('Raw data'),
            dash_table.DataTable(
                id='tble',
                columns=[{"name": i, "id": i} for i in df.columns],
                data=df.to_dict('records'),
                sort_action='native',
                filter_action='native'
                )
            ])

    elif tab == 'tab-5':
        return    html.Div(children=[
            html.Div([
                html.H4('Linear regression results :'),
                html.H6(' MAE_LR = 3.9590377236004777            MSE_LR = 24.131991734230432            RMSE_LR = 4.912432364341562         '),
                html.H6(' cvRMSE_LR = 0.3009657846101855 '),
                html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), style={'height':'50%', 'width':'50%'})
             ]),
    
            html.Div([
                html.H4('Random forest results :'),
                html.H6(' MAE_LR = 3.1098235844621716            MSE_LR = 16.762044451121458           RMSE_LR = 4.094147585410357      '),
                html.H6('    cvRMSE_LR = 0.25102687365365123 '),
                html.Img(src='data:image/png;base64,{}'.format(encoded_image2.decode()), style={'height':'50%', 'width':'50%'})
             ]),
             
            html.Div([
                html.H4('Extrem gradient boosting results :'),
                html.H6(' MAE_LR = 3.198246846110806            MSE_LR = 17.85045840704362           RMSE_LR = 4.2249802848112346          '),
                html.H6('cvRMSE_LR = 0.2590486957343465 '),
                html.Img(src='data:image/png;base64,{}'.format(encoded_image3.decode()), style={'height':'50%', 'width':'50%'})
           ]),
    
            html.Div([
                html.H4('Neural networks results :'),
                html.H6('MAE_LR = 3.5153580234266792           MSE_LR = 20.15767603107264            RMSE_LR = 4.489730062161047          '),
                html.H6('cvRMSE_LR = 0.2752814542077962 '),
                html.Img(src='data:image/png;base64,{}'.format(encoded_image4.decode()), style={'height':'50%', 'width':'50%'})
             ])
            
        ])
            
    elif tab == 'tab-4':
        return    html.Div(children=[
            
            html.Div([
                html.H3('Cluster analysis'),
                ]),
                html.Div([
                html.Img(src='data:image/png;base64,{}'.format(encoded_image5.decode()), style={'height':'80%', 'width':'80%'})
             ],className="six columns"),
    
            html.Div([
                html.Div([
                html.H3(' '),
                html.Img(src='data:image/png;base64,{}'.format(encoded_image8.decode()), style={'height':'80%', 'width':'80%'})
             ],className="six columns"),
             
            ], className="row"),
            
            html.Div([
                html.Img(src='data:image/png;base64,{}'.format(encoded_image6.decode()), style={'height':'80%', 'width':'80%'})
           ],className="six columns"),
    
            html.Div([
              html.Div([
                html.Img(src='data:image/png;base64,{}'.format(encoded_image7.decode()), style={'height':'80%', 'width':'80%'})
             ],className="six columns"),
            ], className="row"),
            
            html.Div([
              
                html.Img(src='data:image/png;base64,{}'.format(encoded_image9.decode()), style={'height':'80%', 'width':'80%'})
             ],className="six columns")
            ])
                
    elif tab == 'tab-3':
        return    html.Div([
            html.H3('The features selected are :'),
            html.H5('- Power_kW'),
            html.H5('- Power-1'),
            html.H5('- Temperature'),
            html.H5('- Hour'),
            html.H5('- Week day'),
            html.H5('- HDH'),
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in df1.columns],
                data=df1.to_dict('records'),
                sort_action='native',
                filter_action='native'
                )
            ])

if __name__ == '__main__':
    app.run_server(debug=False)