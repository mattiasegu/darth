"""Postprocessing utils for tracking results."""
from collections import defaultdict
import pandas as pd
import numpy as np
import statistics


def majority_vote(results):
    """Postprocess tracking results by majority class voting.

    Args:
        - results (list(list(list())))
    """
    num_cats = len(results[0][0])
    post_results = []
    for video_result in results:
        video_track_id_to_result = []
        video_track_id_to_cat = defaultdict(lambda: [])
        for img_result in video_result:
            img_bboxes = np.concatenate(img_result)
            img_track_id_to_result = {
                int(b[0]): np.expand_dims(b, 0) for b in img_bboxes
            }
            for cat, bboxes in enumerate(img_result):
                for bbox in bboxes:
                    if bbox.shape[0] > 0:
                        track_id = int(bbox[0])
                        video_track_id_to_cat[track_id].append(cat)
            video_track_id_to_result.append(img_track_id_to_result)
        
        # get mode
        video_track_id_to_mode = {}
        for track_id, cats in video_track_id_to_cat.items():
            video_track_id_to_mode[track_id] = statistics.mode(cats)

        # postprocess
        post_video_result = []
        for img_track_id_to_result in video_track_id_to_result:
            img_result = [np.ndarray((0,6)) for _ in range(num_cats)]
            for id, res in img_track_id_to_result.items():
                img_result[video_track_id_to_mode[id]] = np.concatenate(
                    [img_result[video_track_id_to_mode[id]], res])
            post_video_result.append(img_result)
        post_results.append(post_video_result)
        
    return post_results
