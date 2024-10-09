import argparse
import json
import requests
from datetime import datetime
from flask import Flask, jsonify, abort, make_response, request, Response
from flask_cors import CORS

from model import WhisperSegmenter, WhisperSegmenterFast
#import librosa
# import pandas as pd
import numpy as np
# from tqdm import tqdm
from pathlib import Path
import datetime as dt

import multiprocessing as mp
import base64
import io
import traceback
import queue

model_paths = []
model_names = []
model_times = []
workers = {}
model_root_path = ''
segment_config = {}

# Make Flask application
app = Flask(__name__)
CORS(app)
# maintain the returned order of keys!
app.json.sort_keys = False

def decimal_to_seconds( decimal_time ):
    splits = decimal_time.split(":")
    if len(splits) == 2:
        hours = 0
        minutes, seconds = splits
    elif len(splits) == 3:
        hours, minutes, seconds = splits
    else:
        assert False

    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)

def seconds_to_decimal( seconds ):
    hours = int(seconds // 3600)
    minutes = int(seconds // 60)
    seconds = seconds % 60

    if hours > 0:
        return "%d:%02d:%06.3f"%( hours, minutes, seconds )
    else:
        return "%d:%06.3f"%( minutes, seconds )

def bytes_to_base64_string(f_bytes):
    return base64.b64encode(f_bytes).decode('ASCII')

def base64_string_to_bytes(base64_string):
    return base64.b64decode(base64_string)

@app.route('/models', methods=['POST'])
def list_models_handler():
    """Request handler for list_models.

    Returns:
        str: Response to query

    """
    response =  list_models()
    return jsonify(response), 201

def list_models():
    """Send client a list of available trained models.

    Returns:
        dict: List of trained models

    """
    global model_names, model_times
    response = {'model_names':model_names, 'model_timestamps':model_times}
    return response

@app.route('/update', methods=['POST'])
def update_models_handler():
    """Request handler for update_models.

    Returns:
        str: Response to query

    """
    response = update_models()
    return jsonify(response), 201

def update_models():
    """Search for new models in the root directory and load them.

    Returns:
        dict: Predicted segmentation and labeling response

    """
    global model_paths, model_names, model_times, workers, model_root_path

    # Reload config
    segment_config = load_config(args.config_file_path)

    # Find available networks
    updated_model_paths, updated_model_names, updated_model_times = find_models(model_root_path)

    # For any networks that aren't already assigned to a worker, assign one
    addedNetworks = 0
    errors = 0
    errorMessages = []
    for model_path, model_name, model_time in zip(updated_model_paths, updated_model_names, updated_model_times):
        if model_path not in model_paths      \
            or model_name not in model_names  \
            or model_time not in model_times:
            try:
                workers[model_name] = create_worker(model_name, model_path, device, device_ids, batch_size, segment_config)
                addedNetworks += 1
            except:
                errors += 1
                errorMessages.append(traceback.format_exc())

    # Update global list of loaded models
    model_names = updated_model_names
    model_times = updated_model_times
    model_paths = updated_model_paths

    if addedNetworks == 0:
        response = 'No updated models added'
    else:
        response = 'Updated models - added {n} new workers.'.format(n=addedNetworks)

    if errors > 0:
        response = response + '\nErrors:\n' + ', '.join(errorMessages)
    return response

@app.route('/segment/<model_name>', methods=['POST'])
def dispatch_segmenter(model_name):
    """Send the audio data to thre requested worker for segmentation.

    Args:
        model_name (str): Name of the trained model to use

    Returns:
        str: Segmentation and labeling prediction

    """
    print('Got request for worker {name}'.format(name=model_name))
    if model_name not in workers:
        print('No worker found')
        prediction = make_empty_prediction()
        prediction['message'] = "Error: no segmentation worker found for the name '{name}'".format(name=model_name)
    else:
        request_info = request.json
        worker = workers[model_name]
        worker.request_queue.put(request_info, block=True)
        print('Sent job to worker')
        try:
            prediction = worker.prediction_queue.get(block=True, timeout=100)
            print('Got response from worker')
        except queue.Empty:
            print('Error: Worker timed out!')
            prediction = make_empty_prediction()
            prediction['message'] = "Error: segmentation worker '{name}' timed out!".format(name=model_name)
    return jsonify(prediction), 201

def make_empty_prediction():
    prediction = {
        "onset":[],
        "offset":[],
        "cluster":[],
        "message":''
    }
    return prediction

class Segmenter(mp.Process):
    """Worker class for segmenting and labeling audio.

    Args:
        model_name (str): Name of the trained model to use
        model_path (str or Path): Path to the trained model
        device (str): Type of compute device to use to run model
        device_ids (list of int): GPU devices IDs available
        batch_size (int): ?
        segment_config (dict): Loaded version of config.json for model

    Attributes:
        request_queue (mp.Queue): A queue for receiving requests from client
        prediction_queue (mp.Queue): A queue for sending predictions back to the
            client
        segmenter (?): Loaded segmenter model object
        model_name
        model_path
        device
        device_ids
        segment_config
        batch_size

    """
    EXIT_CODE='exit'
    def __init__(self, model_name, model_path, device, device_ids, batch_size, segment_config):
        mp.Process.__init__(self, daemon=True)

        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.device_ids = device_ids
        self.segment_config = segment_config
        self.batch_size = batch_size
        self.request_queue = mp.Queue()
        self.prediction_queue = mp.Queue()
        self.segmenter = None
        self.name = 'Segmentation worker for model {model_name}'.format(model_name=self.model_name)

    def run(self):
        """Main run loop for segmentation workers

        Returns:
            None

        """

        # Initialize trained model
        try:
            self.segmenter = WhisperSegmenterFast(self.model_path, device=self.device, device_ids=self.device_ids)
            print("The loaded model is the Ctranslated version.")
        except:
            self.segmenter = WhisperSegmenter(self.model_path, device=self.device, device_ids=self.device_ids)
            print("The loaded model is the original huggingface version.")

        while True:
            # Wait for a request from the client
            request_info = self.request_queue.get(block=True)
            if request_info == Segmenter.EXIT_CODE:
                # Got an exit code
                break
            # Segment the provided audio data, and return it
            prediction = self.segment(request_info)
            # Send the predicted segmentation and labels through a queue to be
            #   returned to the client
            self.prediction_queue.put(prediction)

    def segment(self, request_info):
        """Method to segment the provided audio.

        Args:
            request_info (dict): A dictionary of POST request info from the
                client.

        Returns:
            None (return data is put into a multiprocessing queue)

        """
        try:
            audio_file_base64_string = request_info["audio_file_base64_string"]
            ### drop all the key-value pairs whose value is None, since we will determine the default value within this function.
            request_info = { k:v for k,v in request_info.items() if v is not None}

            if "species" in request_info and request_info["species"] in self.segment_config:
                cfg = self.segment_config[request_info["species"]]
                default_sr = cfg["sr"]
                default_min_frequency = cfg["min_frequency"]
                default_spec_time_step = cfg["spec_time_step"]
                default_min_segment_length = cfg["min_segment_length"]
                default_eps = cfg["eps"]
                default_num_trials = cfg["num_trials"]
            else:
                default_sr = 32000
                default_min_frequency = 0
                default_spec_time_step = 0.0025
                default_min_segment_length = 0.01
                default_eps = 0.02
                default_num_trials = 3

            sr = request_info.get("sr", default_sr)
            min_frequency = request_info.get("min_frequency", default_min_frequency)
            spec_time_step = request_info.get( "spec_time_step", default_spec_time_step )
            min_segment_length = request_info.get( "min_segment_length", default_min_segment_length )
            eps = request_info.get( "eps", default_eps )
            num_trials = request_info.get( "num_trials", default_num_trials )

            channel_id = request_info.get( "channel_id", 0 )
            adobe_audition_compatible = request_info.get( "adobe_audition_compatible", False )

            audioBytes = base64_string_to_bytes(audio_file_base64_string)
            # print('first 32 bytes:')
            # print(audioBytes[:16])
            audio = np.frombuffer(audioBytes, dtype='float').astype('float32')
            # print('Got audio:')
            # print(audio[:4])
            # print('shape:', audio.shape)
    #        audio, _ = librosa.load( io.BytesIO(base64_string_to_bytes(audio_file_base64_string)),
    #                                 sr = sr, mono=False )
            ### for multiple channel audio, choose the desired channel
            if len(audio.shape) == 2:
                audio = audio[channel_id]

            prediction = self.segmenter.segment(
                audio,
                sr = sr,
                min_frequency = min_frequency,
                spec_time_step = spec_time_step,
                min_segment_length = min_segment_length,
                eps = eps,
                num_trials = num_trials,
                batch_size = self.batch_size
                )
            prediction["message"] = 'Success'
        except:
            message = "Segmentation Error! Returning an empty prediction ...\n" + traceback.format_exc()
            prediction = make_empty_prediction()
            prediction["message"] = message

            adobe_audition_compatible = False

        if adobe_audition_compatible:
            Start_list = [ seconds_to_decimal( seconds ) for seconds in prediction["onset"] ]
            Duration_list = [ seconds_to_decimal( end - start ) for start, end in zip( prediction["onset"], prediction["offset"] )  ]
            Format_list = [ "decimal" ] * len(Start_list)
            Type_list = [ "Cue" ] * len(Start_list)
            Description_list = [ "" for _ in range(len(Start_list))]
            Name_list = [ "" for _ in range( len(Start_list) )  ]

            prediction = {
                "\ufeffName":Name_list,
                "Start":Start_list,
                "Duration":Duration_list,
                "Time Format":Format_list,
                "Type":Type_list,
                "Description":Description_list
            }

        return prediction

def find_models(model_root_path):
    """Find available networks in the given root path

    Args:
        model_root_path (str or Path): Root path in which to look for trained
            model subdirectories

    Returns:
        Paths, strs, strs: A list of discovered trained model paths, and
            corresponding lists of model names and creation timestamps

    """
    # Find available networks
    model_root = Path(model_root_path)
    model_paths = [f.resolve() for f in model_root.iterdir() if f.is_dir()]
    model_names = [p.stem for p in model_paths]
    model_times = [str(dt.datetime.fromtimestamp(p.stat().st_mtime)) for p in model_paths]
    return model_paths, model_names, model_times

def create_worker(model_name, model_path, device, device_ids, batch_size, segment_config):
    """Create and start a segmentation worker process.

    Args:
        model_name (str): name of a trained model to use
        model_path (str or Path): path to the trained model to use.
        device (?): GPU index I think?
        device_ids (?): List of GPUs maybe?.
        batch_size (int): ?
        segment_config (dict): Loaded version of config.json file to use.

    Returns:
        mp.Process: Process object representing the worker process

    """
    print('Starting worker for:', model_path)
    worker = Segmenter(model_name, str(model_path), device, device_ids, batch_size, segment_config)
    worker.start()
    return worker

def load_config(config_file_path):
    with open(config_file_path, "r") as config_file:
        segment_config = json.load(config_file)
    return segment_config

def print_models():
    global model_paths, model_names, model_times
    print('model_paths:')
    print(model_paths)
    print('model_names:')
    print(model_names)
    print('model_times:')
    print(model_times)

if __name__ == '__main__':

    try:
        mp.set_start_method('spawn', force=True)
    except:
        print('Unable to set process start method to spawn')
        pass

    # Get startup arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--flask_port", help="The port of the flask app.", default=8050, type=int)
    parser.add_argument("--model_root_path")
    parser.add_argument("--config_file_path", help="The file for recommended segment parameters of different species.", default="config/segment_config.json")
    parser.add_argument("--device", help="cpu or cuda", default = "cuda")
    parser.add_argument("--device_ids", help="a list of GPU ids", type = int, nargs = "+", default = [0,])
    parser.add_argument("--batch_size", default=8, type=int)
    args = parser.parse_args()

    model_root_path = args.model_root_path
    device = args.device
    device_ids = args.device_ids
    batch_size = args.batch_size

    # Start workers for all the available trained models
    update_models()

    print("Waiting for requests...")

    app.run(host='0.0.0.0', port=args.flask_port, threaded = True )
