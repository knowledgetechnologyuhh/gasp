#!/usr/bin/env python
# remember to mount sshfs icub@10.0.0.3:repos/a5-task3-gazenet/logs/metrics metrics
from collections import deque
import random

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly
import plotly.graph_objs as go
import pandas as pd
import plotly.express as px
from sklearn.metrics import accuracy_score

from flask import Flask, Response

from gazenet.applications.audiovisual_gaze_congruence_scenario.utils import VideoStreamer

server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])
vid_streamer = VideoStreamer()  # must be defined in main.py
# stim_streamer = VideoStreamer()  # must be defined in main.py


def Header(name, app):
    title = html.H2(name, style={"margin-top": 5})
    logo = html.Img(
        src=app.get_asset_url("kt-icon.png"), style={"float": "right", "height": 50}
    )

    return dbc.Row([dbc.Col(title, md=9), dbc.Col(logo, md=3)])


def LabeledSelect(label, **kwargs):
    return dbc.FormGroup([dbc.Label(label), dbc.Select(**kwargs)])


human_acc = 0.97
robot_acc = 0.61

human_incon_x = deque(maxlen = 20)
human_incon_x.append(0)
human_incon_y = deque(maxlen = 20)
human_incon_y.append(0)
human_con_x = deque(maxlen = 20)
human_con_x.append(0)
human_con_y = deque(maxlen = 20)
human_con_y.append(0)
human_neut_x = deque(maxlen = 20)
human_neut_x.append(0)
human_neut_y = deque(maxlen = 20)
human_neut_y.append(0)

robot_incon_x = deque(maxlen = 20)
robot_incon_x.append(0)
robot_incon_y = deque(maxlen = 20)
robot_incon_y.append(0)
robot_con_x = deque(maxlen = 20)
robot_con_x.append(0)
robot_con_y = deque(maxlen = 20)
robot_con_y.append(0)
robot_neut_x = deque(maxlen = 20)
robot_neut_x.append(0)
robot_neut_y = deque(maxlen = 20)
robot_neut_y.append(0)

# Card components
cards = [
    dbc.Card(
        [
            html.H2(f"{human_acc*100:.2f}%", className="card-title"),
            html.P("Human Accuracy", className="card-text"),
        ],
        body=True,
        color="light",
    ),
    dbc.Card(
        [
            html.H2(f"{robot_acc*100:.2f}%", className="card-title"),
            html.P("Robot Accuracy", className="card-text"),
        ],
        body=True,
        color="dark",
        inverse=True,
    ),
]



def gen(resource):
    while True:
        frame = resource.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.server.before_first_request
def initialization():
    global vid_source, stim_source

    vid_source = vid_streamer
    #stim_source = stim_streamer


@server.route('/video_feed')
def video_feed():
    return Response(gen(vid_source),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@server.route('/stimuli_feed')
def stimuli_feed():
    return Response(gen(stim_source),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

app.layout = dbc.Container(
    [
        html.Hr(),
        Header("iCub HRI Streamer", app),
        html.Hr(),
        dbc.Row([dbc.Col(card) for card in cards]),
        html.Br(),
        html.Div([
            html.Div([
                html.Img(src="/video_feed", width=540, height=380),
                dcc.Interval(
                    id='interval-vid-component',
                    interval=1000,
                    n_intervals=0),
                html.H5('FPS: 0', id='fps-label')],
                style={'width': '49%', 'display': 'inline-block'}),
            html.Div([
                html.Div(
                [html.H5('Robot Results', id='robot-graph-label'),
                 dcc.Graph(id ='live-robot-graph', animate = True),
                 dcc.Interval(
                      id = 'graph-robot-update',
                      interval=10000,
                      n_intervals=0
                  ),
                 html.H5('Human Results', id='human-graph-label'),
                 dcc.Graph(id='live-human-graph', animate=True),
                 dcc.Interval(
                     id='graph-human-update',
                     interval=10000,
                     n_intervals=0
                 )
                 ], style={'height': '200px'})
            ],
                style={'width': '49%', 'display': 'inline-block'}),
        ]),
        #dbc.Row([dbc.Col(graph) for graph in graphs]),
    ],
    fluid=False,
)



@app.callback(Output('fps-label', 'children'),
              [Input('interval-vid-component', 'n_intervals')])
def update_fps(n):
    return 'FPS: ' + str(vid_source.get_fps())


@app.callback(
    Output('live-robot-graph', 'figure'),
    [Input('graph-robot-update', 'n_intervals')]
)
def update_graph_robot(n):
    robot_data = pd.read_csv("logs/metrics/hriavcongruence.csv")

    robot_con_x.append(robot_con_x[-1] + 1)

    robot_con_y.append(robot_data.loc[robot_data.condition_congruency == "congruent"].tail(10)["pred_accuracy"].mean())
    data_con = plotly.graph_objs.Scatter(
        x=list(robot_con_x),
        y=list(robot_con_y),
        name='Congruent',
        mode='lines+markers'
    )
    robot_incon_y.append(robot_data.loc[robot_data.condition_congruency == "incongruent"].tail(10)["pred_accuracy"].mean())
    data_incon = plotly.graph_objs.Scatter(
        x=list(robot_con_x),
        y=list(robot_incon_y),
        name='Incongruent',
        mode='lines+markers'
    )

    robot_neut_y.append(robot_data.loc[robot_data.condition_congruency == "neutral"].tail(10)["pred_accuracy"].mean())
    data_neut = plotly.graph_objs.Scatter(
        x=list(robot_con_x),
        y=list(robot_neut_y),
        name='Neutral',
        mode='lines+markers'
    )

    return {'data': [data_con, data_incon, data_neut],
            'layout': go.Layout(xaxis=dict(range=[min(robot_con_x), max(robot_con_x)]),
                                yaxis=dict(range=[0, 1]),
                                height=300)}

@app.callback(
    Output('live-human-graph', 'figure'),
    [Input('graph-human-update', 'n_intervals')]
)
def update_graph_human(n):
    human_data = pd.read_csv("logs/app_metrics/hriavcongruence_human.csv")

    human_con_x.append(human_con_x[-1] + 1)

    human_con_y.append(human_data.loc[human_data.conditions_congruency == "congruent"].tail(10)["pred_accuracy"].mean())
    data_con = plotly.graph_objs.Scatter(
        x=list(human_con_x),
        y=list(human_con_y),
        name='Congruent',
        mode='lines+markers'
    )
    human_incon_y.append(human_data.loc[human_data.conditions_congruency == "incongruent"].tail(10)["pred_accuracy"].mean())
    data_incon = plotly.graph_objs.Scatter(
        x=list(human_con_x),
        y=list(human_incon_y),
        name='Incongruent',
        mode='lines+markers'
    )

    human_neut_y.append(human_data.loc[human_data.conditions_congruency == "neutral"].tail(10)["pred_accuracy"].mean())
    data_neut = plotly.graph_objs.Scatter(
        x=list(human_con_x),
        y=list(human_neut_y),
        name='Neutral',
        mode='lines+markers'
    )

    return {'data': [data_con, data_incon, data_neut],
            'layout': go.Layout(xaxis=dict(range=[min(human_con_x), max(human_con_x)]),
                                yaxis=dict(range=[0, 1]),
                                height=300)}

if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port=8050, debug=True, use_reloader=False)