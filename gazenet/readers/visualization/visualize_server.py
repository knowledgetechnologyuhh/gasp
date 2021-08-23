# TODO (fabawi): The audio is not actually streamed but played locally. Need to change that
from io import *
import base64
import os
import time
import re

import sounddevice as sd
from flask import Flask, Response
import numpy as np
import cv2
import dash
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from gazenet.utils.registrar import *
from gazenet.utils.helpers import encode_image
from gazenet.utils.annotation_plotter import OpenCV
from gazenet.utils.dataset_processors import DataSplitter
import gazenet.utils.sample_processors as sp

width, height = 800, 400

reader = "FindWhoSampleReader"
sampler = "FindWhoSample"
play_mode = 'play'
DEBUG = False
sp.SERVER_MODE = True

SampleRegistrar.scan()
ReaderRegistrar.scan()
dspl = DataSplitter()

video_source = ReaderRegistrar.registry[reader](mode="d")
video = SampleRegistrar.registry[sampler](video_source, w_size=1, width=width, height=height)


FONT_AWESOME = "https://use.fontawesome.com/releases/v5.7.2/css/all.css"

server = Flask(__name__)
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, FONT_AWESOME], server=server)

plotter = OpenCV()

dataset_info = video_source.dataset_info()
dataset_name = dataset_info["name"]
dataset_summary = dataset_info["summary"]
dataset_link = dataset_info["link"]


reset_sample = True

# TODO (fabawi): pass initialization arguments to the preprocess for tracing the properties
preprocessed_data = video.preprocess_frames()
if preprocessed_data is not None:
    extracted_data_list = video.extract_frames(**preprocessed_data)
else:
    extracted_data_list = video.extract_frames()
_, _, _, _, _, properties = video.annotate_frame(input_data=next(zip(*extracted_data_list)), plotter=plotter)
video_properties = {k: v[0] for k,v in properties.items()}

dummy_img_vid = encode_image(np.random.randint(255, size=(height,width,3),dtype=np.uint8), raw=True)
dummy_img = encode_image(np.random.randint(255, size=(height,width,3),dtype=np.uint8))

def gen_video():
    while True:
        ts = time.time()
        with video.read_lock:
            video.buffer.put(play_mode)
        grabbed_video, video_frame, grabbed_audio, audio_frames, _, _ = video.read()

        time.sleep(0.01)
        # time.sleep(sample_streamer.frames_per_sec() * 0.00106)
        if video_frame is not None:
            if grabbed_audio:
                sd.play(audio_frames, samplerate=video.audio_cap.get(sp.AUCAP_PROP_SAMPLE_RATE))
            video_frame = encode_image(video_frame.copy(), True)
            orig_fps = video.frames_per_sec() * 1.06  # used to be multiplied by 1.51
            td = time.time() - ts
            try:
                if td < 1.0 / orig_fps:
                    time.sleep(1.0 / orig_fps - td)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + video_frame + b'\r\n\r\n')
            except:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + dummy_img_vid + b'\r\n\r\n')
        else:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + dummy_img_vid + b'\r\n\r\n')

@server.route('/video_feed')
def video_feed():
    time.sleep(1)
    stream_img = gen_video()
    return Response(stream_img,
                    mimetype='multipart/x-mixed-replace; boundary=frame')

dynamic_elements = {'toggle': [], 'multilist': []}
def generate_elements(properties):
    all_elements = []
    for property_name, property in properties.items():
        if property[1] == 'toggle':
            all_elements.append(
                html.Div(
                    className="control-element",
                    children=[
                    html.Div(
                        children=["Switch property:"]
                    ),
                    dbc.Checklist(options=[
                        {"label": property_name.replace("_", " "), "value": 1},
                    ],
                    value=[],
                    id="property-"+property_name,
                    switch=True,
                    )
                    ])
            )
            dynamic_elements['toggle'].append("property-"+property_name)
    return all_elements


cards = []
for i, sample in enumerate(video_source.samples):
    audio_symbol = "\U0001F508" if sample["has_audio"] else " "
    split, category = dspl.sample(sample["id"], video_source.short_name , mode="r")
    card = dbc.Card(
            [
                dbc.CardImg(src=encode_image(sample['video_thumbnail']),
                    id="video-"+str(i)+"-card-img", top=True), #  style={"display": "none"}
                dbc.CardBody([
                    dbc.Button(
                        os.path.basename(sample["video_name"]) + audio_symbol,
                        id="video-" + str(i) + "-card-button",
                        color="dark",
                        style={'display': 'inline-block'}),
                    dbc.DropdownMenu(
                        label="",
                        children=[
                            dbc.DropdownMenuItem("Split", header=True),
                            dbc.DropdownMenuItem("train", id="video-" + str(i) + "-split-train-button"),
                            dbc.DropdownMenuItem("val", id="video-" + str(i) + "-split-val-button"),
                            dbc.DropdownMenuItem("test", id="video-" + str(i) + "-split-test-button"),
                            dbc.DropdownMenuItem("None", id="video-" + str(i) + "-split-None-button"),
                            dbc.DropdownMenuItem(divider=True),
                            dbc.DropdownMenuItem("Category", header=True),
                            dbc.DropdownMenuItem("Social", id="video-" + str(i) + "-cat-Social-button"),
                            dbc.DropdownMenuItem("Nature", id="video-" + str(i) + "-cat-Nature-button"),
                            dbc.DropdownMenuItem("Other", id="video-" + str(i) + "-cat-Other-button"),
                            dbc.DropdownMenuItem("None", id="video-" + str(i) + "-cat-None-button"),
                        ],
                        style={'display': 'inline-block', 'float': 'right'},
                    ),
                    html.Div([
                        dbc.Badge(split if split is not None else "None",
                                  id="video-" + str(i) + "-split-bdg",
                                  color="success", className="mr-1"),
                        dbc.Badge(category if category is not None else "None",
                                  id="video-" + str(i) + "-category-bdg",
                                  color="info", className="mr-1"),
                    ], style={"display": "block"})

                ])],
        color="primary", inverse=True)

    cards.append(card)

# Main App
app.layout = html.Div(
    children=[
        dcc.Interval(id="video-player", interval=100, n_intervals=0),
        html.Div(id="top-bar", className="row"),
        html.Div(
            className="page-content",
            children=[
                html.Div(
                    id="left-side-column",
                    className="four columns",
                    children=[
                        dbc.CardColumns(cards),
                              html.Img(src='', id="magic-img")
                    ],
                ),
                html.Div(
                    id="right-side-column",
                    className="eight columns",
                    children=[
                        html.Div(
                            id="header-section",
                            children=[
                                html.H4(dataset_name),
                                html.P(
                                    dataset_summary
                                ),
                                dcc.Link(html.Button(
                                    "Learn More", id="learn-more-button", n_clicks=0
                                ), href=dataset_link,  target='_blank'),
                                html.Div(
                                    [
                                        dbc.Button("Save", id="open-save-dialog-button",
                                            style={'display': 'inline-block', 'float': 'right'}),
                                        dbc.Modal(
                                            [
                                                dbc.ModalHeader("Save"),
                                                dbc.ModalBody("Are you sure you want to save the dataset splits and categories?"),
                                                dbc.ModalFooter([
                                                    dbc.Button(
                                                        "Yes", id="save-split-cat-button", className="ml-auto", color="success",
                                                    ),
                                                    dbc.Button(
                                                        "No", id="close-save-dialog-button", className="ml-auto", color="dark",
                                                    )
                                                    ]
                                                ),
                                            ],
                                            backdrop=False,
                                            zIndex=1000000,
                                            id="save-dialog",
                                            centered=True,
                                        ),
                                    ])
                            ],
                        ),
                        html.Div(children=html.Hr()),
                        html.Div(
                            id="video-name",
                            children=[]
                        ),
                        html.Div(
                            className="video-outer-container",
                            children=[
                                html.Div(
                                className="video-container",
                                children=[
                                            html.Img(id="video-container",
                                                     src="/video_feed"),
                                          ]
                            ),
                                html.Div(
                                    className="video-control-section",
                                    children=[
                                        html.Div(
                                            className="video-control-element",
                                            children=[
                                                html.Button(
                                                    "",
                                                    className="fas fa-pause-circle fa-lg",
                                                    style={'border': 'none', 'height': '15px', 'padding':'0px'},
                                                    id="video-control-button",
                                                    n_clicks=0
                                                ),
                                                html.Button(
                                                    "",
                                                    className="fas fa-stop-circle fa-lg",
                                                    style={'border': 'none', 'height': '15px', 'padding':'0px'},
                                                    id="video-stop-button",
                                                    n_clicks=0
                                                ),
                                                dcc.Slider(
                                                    id="video-frame-slider",
                                                    min=20,
                                                    max=80,
                                                    value=0),
                                            ],
                                        ),
                                    ],
                                ),
                            ]
                        ),
                        html.Div(id="enabled-properties", style={"display": "none"}, children=[]),
                        html.Div(
                            className="control-section",
                            children= generate_elements(properties) ,
                        ),
                    ],
                ),

            ],
        ),
    ]
)

@app.callback([Output("video-stop-button", "className")],
              [Input("video-stop-button", "n_clicks")],
              [State("video-stop-button", "className")])
def stop_video(n, class_name):
    global play_mode
    global reset_sample
    if n > 0:
            video.goto_frame(0)
            video.stop()
            video.goto(video.index)
            video.set_annotation_properties(video_properties.copy())
            reset_sample = True
            video.start(plotter)
            play_mode = 'pause'
            sd.stop()
    return [class_name]


@app.callback([Output("video-control-button", "className")],
              [Input("video-control-button", "n_clicks")],
              [State("video-control-button", "className")])
def control_video(n, class_name):
    global play_mode
    if n > 0:
        if class_name == 'fas fa-play-circle fa-lg':
            play_mode = 'play'
            return ['fas fa-pause-circle fa-lg']
        else:
            play_mode = 'pause'
            sd.stop()
            return ['fas fa-play-circle fa-lg']
    return [class_name]


@app.callback([Output("video-frame-slider", "value")],
              [Input("video-player", "n_intervals")],
              [State("video-frame-slider", "value"),
               State("video-control-button", "className")])
def start_video(n, prev_frame_idx, class_name):
    global reset_sample
    if n > 0 and not reset_sample and prev_frame_idx is not None:
        new_frame_idx = video.frame_index()
        # TODO (fabawi): Find a better way to seek
        if np.abs((prev_frame_idx - new_frame_idx)) > video.frames_per_sec() + 10:
            video.goto_frame(prev_frame_idx)
        return [video.frame_index()]
    else:
        reset_sample = False
        return [0]


@app.callback(
    [Output("video-name", "children"), Output("video-frame-slider","min"), Output("video-frame-slider","max")],
    [Input(f"video-{i}-card-button", "n_clicks_timestamp") for i in range(len(cards))] + [Input(f"video-{i}-card-button", "n_clicks") for i in range(len(cards))] ,
    [State(f"video-{i}-card-img", "src") for i in range(len(cards))]
)
def activate_video(*args):
    global reset_sample
    h_args = args[:len(cards)+1]
    if h_args and h_args is not None:
        f_h_args = list(filter(None.__ne__, h_args))
        if f_h_args:
            index = h_args.index(max(f_h_args)) + (len(cards)*2)
            if args[index - len(cards)] is not None and args[index - len(cards)] > 0:
                video.goto_frame(0)
                video.stop()
                video.goto(index - (len(cards) * 2))
                video.set_annotation_properties(video_properties.copy())
                reset_sample = True
                video.start(plotter)
                curr_sample = video.reader.samples[video.index]
                video_name = os.path.basename(curr_sample['video_name'])
                time.sleep(1)
                return [html.H5(video_name), 0, video.len_frames() - 2]
        else:
            return [html.H5("No video selected"), 1, 2]


# Toggle properties
@app.callback(
    [Output("enabled-properties", "children")],
    [Input(f"{i}", "value") for i in dynamic_elements['toggle']],
    [State(f"{i}", "id") for i in dynamic_elements['toggle']]
)
def set_property(*args):
    values = args[:len(dynamic_elements['toggle'])]
    ids = args[len(dynamic_elements['toggle']):]
    for idx, value in enumerate(values):
        video_properties[ids[idx].replace('property-', '')] = True if len(value) == 1 else False
    video.set_annotation_properties(video_properties.copy())
    return ["None"]

@app.callback(
    [Output(f"video-{i}-split-bdg", "children") for i in range(len(cards))],
    [Input(f"video-{i}-split-train-button", "n_clicks") for i in range(len(cards))] +
    [Input(f"video-{i}-split-val-button", "n_clicks") for i in range(len(cards))] +
    [Input(f"video-{i}-split-test-button", "n_clicks") for i in range(len(cards))] +
    [Input(f"video-{i}-split-None-button", "n_clicks") for i in range(len(cards))] ,
    [State(f"video-{i}-split-bdg", "children") for i in range(len(cards))]
)
def control_video_split(*args):
    # we performed the button checking before in a different way, but this is the most recently recommended method
    # https://dash.plotly.com/dash-html-components/button
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    changed_id = changed_id
    outputs = [*args[-len(cards):]]
    if changed_id is not None or changed_id != ".":
        for split_name in ("train", "val", "test", "None"):
            match = re.match(r'video-(.*)-split-' + split_name + '-button(.*)', changed_id)
            if match is not None:
                index = int(match.group(1))
                dspl.sample(video_source.samples[index]["id"], video_source.short_name,
                            fps=video_source.samples[index].get("video_fps", 30), split=split_name, mode="d")
                outputs[index] = split_name
    return outputs

@app.callback(
    [Output(f"video-{i}-category-bdg", "children") for i in range(len(cards))],
    [Input(f"video-{i}-cat-Social-button", "n_clicks") for i in range(len(cards))] +
    [Input(f"video-{i}-cat-Nature-button", "n_clicks") for i in range(len(cards))] +
    [Input(f"video-{i}-cat-Other-button", "n_clicks") for i in range(len(cards))] +
    [Input(f"video-{i}-cat-None-button", "n_clicks") for i in range(len(cards))] ,
    [State(f"video-{i}-category-bdg", "children") for i in range(len(cards))]
)
def control_video_category(*args):
    # we performed the button checking before in a different way, but this is the most recently recommended method
    # https://dash.plotly.com/dash-html-components/button
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    changed_id = changed_id
    outputs = [*args[-len(cards):]]
    if changed_id is not None or changed_id != ".":
        for cat_name in ("Social", "Nature", "Other", "None"):
            match = re.match(r'video-(.*)-cat-' + cat_name + '-button(.*)', changed_id)
            if match is not None:
                index = int(match.group(1))
                dspl.sample(video_source.samples[index]["id"], video_source.short_name,
                            fps=video_source.samples[index].get("video_fps", None), scene_type=cat_name, mode="d")
                outputs[index] = cat_name
    return outputs

@app.callback(
    Output("save-dialog", "is_open"),
    [Input("open-save-dialog-button", "n_clicks"), Input("save-split-cat-button", "n_clicks"), Input("close-save-dialog-button", "n_clicks")],
    [State("save-dialog", "is_open")],
)
def save_split_category(nopen, nsave, nclose, is_open):
    if nopen or nsave or nclose:
        if nsave is not None and nsave > 0:
            dspl.save()
        return not is_open
    return is_open


if __name__ == '__main__':
    # run on flask
    # app.run_server(debug=DEBUG)

    # run on waitress
    from waitress import serve
    serve(server, host="0.0.0.0", port=8080)