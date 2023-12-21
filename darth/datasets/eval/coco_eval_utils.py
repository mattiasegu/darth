from email.policy import default
import numpy as np
from collections import defaultdict
from pandas import DataFrame


def get_default_trackeval_dataset_config():
    """Default class config values"""
    default_config = {
        "GT_FOLDER": None,
        "TRACKERS_FOLDER": None,
        "OUTPUT_FOLDER":
        None,  # Trackers location,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
        "TRACKERS_TO_EVAL":
        None,  # Filenames of trackers to eval (if None, all in folder)
        "CLASSES_TO_EVAL": None,  # Classes to eval (if None, all classes)
        "SPLIT_TO_EVAL": "training",  # Valid: 'training', 'val'
        "INPUT_AS_ZIP": False,
        "PRINT_CONFIG": False,  # Whether to print current config
        "TRACKER_SUB_FOLDER":
        "data",  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
        "OUTPUT_SUB_FOLDER":
        "",  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
        "TRACKER_DISPLAY_NAMES":
        None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
    }
    return default_config


def get_default_trackeval_config():
    """Returns the default config values for evaluation."""
    default_config = {
        "USE_PARALLEL": True,
        "NUM_PARALLEL_CORES": 8,
        "BREAK_ON_ERROR": True,
        "RETURN_ON_ERROR": False,
        "LOG_ON_ERROR": None,
        "PRINT_RESULTS": False,
        "PRINT_ONLY_COMBINED": True,
        "PRINT_CONFIG": False,
        "TIME_PROGRESS": True,
        "DISPLAY_LESS_PROGRESS": True,
        "OUTPUT_SUMMARY": True,
        "OUTPUT_EMPTY_CLASSES": True,
        "OUTPUT_DETAILED": True,
        "PLOT_CURVES": True,
    }
    return default_config


def get_track_id_str(ann):
    """Get name of track ID in annotation."""
    if "track_id" in ann:
        tk_str = "track_id"
    elif "instance_id" in ann:
        tk_str = "instance_id"
    elif "scalabel_id" in ann:
        tk_str = "scalabel_id"
    else:
        assert False, "No track/instance ID."
    return tk_str

