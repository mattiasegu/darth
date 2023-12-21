"""BDD100K Dataset in COCO format."""
import itertools
import json
import os
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment
from trackeval.utils import TrackEvalException
from trackeval.datasets._base_dataset import _BaseDataset
from trackeval import _timing

from .coco_eval_utils import get_track_id_str


class BDD100K(_BaseDataset):
    """BDD100K tracking dataset in COCO format."""

    @staticmethod
    def get_default_dataset_config():
        # not used in our case
        pass

    def __init__(self, config):
        """Initialize dataset, checking that all required files are present."""
        super().__init__()
        # Fill non-given config values with defaults
        self.config = config
        self.gt_fol = self.config["GT_FOLDER"]
        self.tracker_fol = self.config["TRACKERS_FOLDER"]
        self.should_classes_combine = True
        self.use_super_categories = False

        self.tracker_sub_fol = self.config["TRACKER_SUB_FOLDER"]
        self.output_fol = self.config["OUTPUT_FOLDER"]
        if self.output_fol is None:
            self.output_fol = self.tracker_fol
        self.output_sub_fol = self.config["OUTPUT_SUB_FOLDER"]

        if self.gt_fol.endswith(".json"):
            self.gt_data = json.load(open(self.gt_fol, "r"))
        else:
            gt_dir_files = [
                file for file in os.listdir(self.gt_fol)
                if file.endswith(".json")
            ]
            if len(gt_dir_files) != 1:
                raise TrackEvalException(
                    f"{self.gt_fol} does not contain exactly one json file.")

            with open(os.path.join(self.gt_fol, gt_dir_files[0])) as f:
                self.gt_data = json.load(f)

        # fill missing video ids
        self._fill_video_ids_inplace(self.gt_data["annotations"])

        # get sequences to eval and sequence information
        self.seq_list = [
            vid["name"].replace("/", "-") for vid in self.gt_data["videos"]
        ]
        self.seq_name2seqid = {
            vid["name"].replace("/", "-"): vid["id"]
            for vid in self.gt_data["videos"]
        }
        # compute mappings from videos to annotation data
        self.video2gt_track, self.video2gt_image = self._compute_vid_mappings(
            self.gt_data["annotations"])
        # compute sequence lengths
        self.seq_lengths = {vid["id"]: 0 for vid in self.gt_data["videos"]}
        for img in self.gt_data["images"]:
            self.seq_lengths[img["video_id"]] += 1
        self.seq2images2timestep = self._compute_image_to_timestep_mappings()
        self.seq2cls = {
            vid["id"]: {
                "pos_cat_ids":
                list({
                    track["category_id"]
                    for track in self.video2gt_track[vid["id"]]
                }),
            }
            for vid in self.gt_data["videos"]
        }

        # Get classes to eval
        considered_vid_ids = [
            self.seq_name2seqid[vid] for vid in self.seq_list
        ]
        seen_cats = set([
            cat_id for vid_id in considered_vid_ids
            for cat_id in self.seq2cls[vid_id]["pos_cat_ids"]
        ])
        # only classes with ground truth are evaluated in TAO
        self.valid_classes = [
            cls["name"] for cls in self.gt_data["categories"]
            if cls["id"] in seen_cats
        ]
        cls_name2clsid_map = {
            cls["name"]: cls["id"]
            for cls in self.gt_data["categories"]
        }

        if self.config["CLASSES_TO_EVAL"]:
            self.class_list = [
                cls.lower() if cls.lower() in self.valid_classes else None
                for cls in self.config["CLASSES_TO_EVAL"]
            ]
            if not all(self.class_list):
                valid_cls = ", ".join(self.valid_classes)
                raise TrackEvalException(
                    "Attempted to evaluate an invalid class. Only classes "
                    f"{valid_cls} are valid (classes present in ground truth"
                    " data).")
        else:
            self.class_list = [cls for cls in self.valid_classes]
        self.cls_name2clsid = {
            k: v
            for k, v in cls_name2clsid_map.items() if k in self.class_list
        }
        self.clsid2cls_name = {
            v: k
            for k, v in cls_name2clsid_map.items() if k in self.class_list
        }
        # get trackers to eval
        if self.config["TRACKERS_TO_EVAL"] is None:
            self.tracker_list = os.listdir(self.tracker_fol)
        else:
            self.tracker_list = self.config["TRACKERS_TO_EVAL"]

        if self.config["TRACKER_DISPLAY_NAMES"] is None:
            self.tracker_to_disp = dict(
                zip(self.tracker_list, self.tracker_list))
        elif (self.config["TRACKERS_TO_EVAL"] is not None) and (len(
                self.config["TRACKER_DISPLAY_NAMES"]) == len(
                    self.tracker_list)):
            self.tracker_to_disp = dict(
                zip(self.tracker_list, self.config["TRACKER_DISPLAY_NAMES"]))
        else:
            raise TrackEvalException(
                "List of tracker files and tracker display names do not match."
            )

        self.tracker_data = {tracker: dict() for tracker in self.tracker_list}

        for tracker in self.tracker_list:
            if self.tracker_sub_fol.endswith(".json"):
                with open(os.path.join(self.tracker_sub_fol)) as f:
                    curr_data = json.load(f)
            else:
                tr_dir = os.path.join(self.tracker_fol, tracker,
                                      self.tracker_sub_fol)
                tr_dir_files = [
                    file for file in os.listdir(tr_dir)
                    if file.endswith(".json")
                ]
                if len(tr_dir_files) != 1:
                    raise TrackEvalException(
                        f"{tr_dir} does not contain exactly one json file.")
                with open(os.path.join(tr_dir, tr_dir_files[0])) as f:
                    curr_data = json.load(f)

            # fill missing video ids
            self._fill_video_ids_inplace(curr_data)

            # make track ids unique over whole evaluation set
            self._make_tk_ids_unique(curr_data)

            # get tracker sequence information
            curr_vids2tracks, curr_vids2images = self._compute_vid_mappings(
                curr_data)
            self.tracker_data[tracker]["vids_to_tracks"] = curr_vids2tracks
            self.tracker_data[tracker]["vids_to_images"] = curr_vids2images

    def get_display_name(self, tracker):
        return self.tracker_to_disp[tracker]

    def _load_raw_file(self, tracker, seq, is_gt):
        """Load a file (gt or tracker) in the Coco format

        If is_gt, this returns a dict which contains the fields:
        [gt_ids, gt_classes]:
            list (for each timestep) of 1D NDArrays (for each det).
        [gt_dets, gt_crowd_ignore_regions]: list (for each timestep) of lists of detections.

        if not is_gt, this returns a dict which contains the fields:
        [tracker_ids, tracker_classes, tracker_confidences]:
            list (for each timestep) of 1D NDArrays (for each det).
        [tracker_dets]: list (for each timestep) of lists of detections.
        """
        seq_id = self.seq_name2seqid[seq]
        # file location
        if is_gt:
            imgs = self.video2gt_image[seq_id]
        else:
            imgs = self.tracker_data[tracker]["vids_to_images"][seq_id]

        # convert data to required format
        num_timesteps = self.seq_lengths[seq_id]
        img_to_timestep = self.seq2images2timestep[seq_id]
        data_keys = ["ids", "classes", "dets"]
        if is_gt:
            data_keys += ["gt_crowd_ignore_regions"]
        else:
            data_keys += ["tracker_confidences"]
        raw_data = {key: [None] * num_timesteps for key in data_keys}

        for img in imgs:
            # some tracker data contains images without any ground truth info,
            # these are ignored
            if img["id"] not in img_to_timestep:
                continue
            t = img_to_timestep[img["id"]]
            anns = img["annotations"]

            ig_ids = []
            keep_ids = []
            for i, ann in enumerate(anns):
                if is_gt and ann["iscrowd"] == 1:
                    ig_ids.append(i)
                else:
                    keep_ids.append(i)

            tk_str = get_track_id_str(anns[0])
            if keep_ids:
                raw_data["dets"][t] = np.atleast_2d(
                    [anns[i]["bbox"] for i in keep_ids]).astype(float)
                raw_data["ids"][t] = np.atleast_1d(
                    [anns[i][tk_str] for i in keep_ids]).astype(int)
                raw_data["classes"][t] = np.atleast_1d(
                    [anns[i]["category_id"] for i in keep_ids]).astype(int)
                if not is_gt:
                    raw_data["tracker_confidences"][t] = np.atleast_1d(
                        [anns[i]["score"] for i in keep_ids]).astype(float)
            else:
                raw_data["dets"][t] = np.empty((0, 4)).astype(float)
                raw_data["ids"][t] = np.empty(0).astype(int)
                raw_data["classes"][t] = np.empty(0).astype(int)

            if is_gt:
                if ig_ids:
                    raw_data["gt_crowd_ignore_regions"][t] = np.atleast_2d(
                        [anns[i]["bbox"] for i in ig_ids]).astype(float)
                else:
                    raw_data["gt_crowd_ignore_regions"][t] = np.empty(
                        (0, 4)).astype(float)

        for t, d in enumerate(raw_data["dets"]):
            if d is None:
                raw_data["dets"][t] = np.empty((0, 4)).astype(float)
                raw_data["ids"][t] = np.empty(0).astype(int)
                raw_data["classes"][t] = np.empty(0).astype(int)
                if not is_gt:
                    raw_data["tracker_confidences"][t] = np.empty(0)

            if is_gt and raw_data["gt_crowd_ignore_regions"][t] is None:
                raw_data["gt_crowd_ignore_regions"][t] = np.empty(
                    (0, 4)).astype(float)

        if is_gt:
            key_map = {
                "ids": "gt_ids",
                "classes": "gt_classes",
                "dets": "gt_dets"
            }
        else:
            key_map = {
                "ids": "tracker_ids",
                "classes": "tracker_classes",
                "dets": "tracker_dets",
            }
        for k, v in key_map.items():
            raw_data[v] = raw_data.pop(k)

        raw_data["num_timesteps"] = num_timesteps
        raw_data["seq"] = seq
        return raw_data

    @_timing.time
    def get_preprocessed_seq_data(self, raw_data, cls):
        """Preprocess data for a single sequence for a single class ready for evaluation.
        Inputs:
            - raw_data is a dict containing the data for the sequence already read in by get_raw_seq_data().
            - cls is the class to be evaluated.
        Outputs:
            - data is a dict containing all of the information that metrics need to perform evaluation.
                It contains the following fields:
                    [num_timesteps, num_gt_ids, num_tracker_ids, num_gt_dets, num_tracker_dets] : integers.
                    [gt_ids, tracker_ids, tracker_confidences]: list (for each timestep) of 1D NDArrays (for each det).
                    [gt_dets, tracker_dets]: list (for each timestep) of lists of detections.
                    [similarity_scores]: list (for each timestep) of 2D NDArrays.
        Notes:
            General preprocessing (preproc) occurs in 4 steps. Some datasets may not use all of these steps.
                1) Extract only detections relevant for the class to be evaluated (including distractor detections).
                2) Match gt dets and tracker dets. Remove tracker dets that are matched to a gt det that is of a
                    distractor class, or otherwise marked as to be removed.
                3) Remove unmatched tracker dets if they fall within a crowd ignore region or don't meet a certain
                    other criteria (e.g. are too small).
                4) Remove gt dets that were only useful for preprocessing and not for actual evaluation.
            After the above preprocessing steps, this function also calculates the number of gt and tracker detections
                and unique track ids. It also relabels gt and tracker ids to be contiguous and checks that ids are
                unique within each timestep.
        BDD100K:
            In BDD100K, the 4 preproc steps are as follow:
                1) There are eight classes (pedestrian, rider, car, bus, truck, train, motorcycle, bicycle)
                    which are evaluated separately.
                2) For BDD100K there is no removal of matched tracker dets.
                3) Crowd ignore regions are used to remove unmatched detections.
                4) No removal of gt dets.
        """

        cls_id = self.cls_name2clsid[cls]

        data_keys = [
            "gt_ids",
            "tracker_ids",
            "gt_dets",
            "tracker_dets",
            "similarity_scores",
        ]
        data = {key: [None] * raw_data["num_timesteps"] for key in data_keys}
        unique_gt_ids = []
        unique_tracker_ids = []
        num_gt_dets = 0
        num_tracker_dets = 0
        for t in range(raw_data["num_timesteps"]):

            # Only extract relevant dets for this class for preproc and eval (cls)
            gt_class_mask = np.atleast_1d(raw_data["gt_classes"][t] == cls_id)
            gt_class_mask = gt_class_mask.astype(np.bool)
            gt_ids = raw_data["gt_ids"][t][gt_class_mask]
            gt_dets = raw_data["gt_dets"][t][gt_class_mask]

            tracker_class_mask = np.atleast_1d(
                raw_data["tracker_classes"][t] == cls_id)
            tracker_class_mask = tracker_class_mask.astype(np.bool)
            tracker_ids = raw_data["tracker_ids"][t][tracker_class_mask]
            tracker_dets = raw_data["tracker_dets"][t][tracker_class_mask]
            similarity_scores = raw_data["similarity_scores"][t][
                gt_class_mask, :][:, tracker_class_mask]

            # Match tracker and gt dets (with hungarian algorithm)
            unmatched_indices = np.arange(tracker_ids.shape[0])
            if gt_ids.shape[0] > 0 and tracker_ids.shape[0] > 0:
                matching_scores = similarity_scores.copy()
                matching_scores[matching_scores < 0.5 -
                                np.finfo("float").eps] = 0
                match_rows, match_cols = linear_sum_assignment(
                    -matching_scores)
                actually_matched_mask = (
                    matching_scores[match_rows, match_cols] >
                    0 + np.finfo("float").eps)
                match_cols = match_cols[actually_matched_mask]
                unmatched_indices = np.delete(
                    unmatched_indices, match_cols, axis=0)

            # For unmatched tracker dets, remove those that are greater than 50% within a crowd ignore region.
            unmatched_tracker_dets = tracker_dets[unmatched_indices, :]
            crowd_ignore_regions = raw_data["gt_crowd_ignore_regions"][t]
            intersection_with_ignore_region = self._calculate_box_ious(
                unmatched_tracker_dets,
                crowd_ignore_regions,
                do_ioa=True,
            )
            is_within_crowd_ignore_region = np.any(
                intersection_with_ignore_region > 0.5 + np.finfo("float").eps,
                axis=1)

            # Apply preprocessing to remove unwanted tracker dets.
            to_remove_tracker = unmatched_indices[
                is_within_crowd_ignore_region]
            data["tracker_ids"][t] = np.delete(
                tracker_ids, to_remove_tracker, axis=0)
            data["tracker_dets"][t] = np.delete(
                tracker_dets, to_remove_tracker, axis=0)
            similarity_scores = np.delete(
                similarity_scores, to_remove_tracker, axis=1)

            data["gt_ids"][t] = gt_ids
            data["gt_dets"][t] = gt_dets
            data["similarity_scores"][t] = similarity_scores

            unique_gt_ids += list(np.unique(data["gt_ids"][t]))
            unique_tracker_ids += list(np.unique(data["tracker_ids"][t]))
            num_tracker_dets += len(data["tracker_ids"][t])
            num_gt_dets += len(data["gt_ids"][t])

        # Re-label IDs such that there are no empty IDs
        if len(unique_gt_ids) > 0:
            unique_gt_ids = np.unique(unique_gt_ids)
            gt_id_map = np.nan * np.ones((np.max(unique_gt_ids) + 1))
            gt_id_map[unique_gt_ids] = np.arange(len(unique_gt_ids))
            for t in range(raw_data["num_timesteps"]):
                if len(data["gt_ids"][t]) > 0:
                    data["gt_ids"][t] = gt_id_map[data["gt_ids"][t]].astype(
                        np.int)
        if len(unique_tracker_ids) > 0:
            unique_tracker_ids = np.unique(unique_tracker_ids)
            tracker_id_map = np.nan * np.ones((np.max(unique_tracker_ids) + 1))
            tracker_id_map[unique_tracker_ids] = np.arange(
                len(unique_tracker_ids))
            for t in range(raw_data["num_timesteps"]):
                if len(data["tracker_ids"][t]) > 0:
                    data["tracker_ids"][t] = tracker_id_map[data["tracker_ids"]
                                                            [t]].astype(np.int)

        # Record overview statistics.
        data["num_tracker_dets"] = num_tracker_dets
        data["num_gt_dets"] = num_gt_dets
        data["num_tracker_ids"] = len(unique_tracker_ids)
        data["num_gt_ids"] = len(unique_gt_ids)
        data["num_timesteps"] = raw_data["num_timesteps"]

        # Ensure that ids are unique per timestep.
        self._check_unique_ids(data)

        return data

    def _calculate_similarities(self, gt_dets_t, tk_dets_t):
        """Compute similarity scores."""
        sim_scores = self._calculate_box_ious(gt_dets_t, tk_dets_t)
        return sim_scores

    def _compute_vid_mappings(self, annotations):
        """Computes mappings from videos to corresponding tracks and images."""
        vids_to_tracks = {}
        vids_to_imgs = {}
        vid_ids = [vid["id"] for vid in self.gt_data["videos"]]

        # compute an mapping from image IDs to images
        images = {}
        for image in self.gt_data["images"]:
            images[image["id"]] = image

        tk_str = get_track_id_str(annotations[0])
        for ann in annotations:
            ann["area"] = ann["bbox"][2] * ann["bbox"][3]

            vid = ann["video_id"]
            if ann["video_id"] not in vids_to_tracks.keys():
                vids_to_tracks[ann["video_id"]] = list()
            if ann["video_id"] not in vids_to_imgs.keys():
                vids_to_imgs[ann["video_id"]] = list()

            # fill in vids_to_tracks
            tid = ann[tk_str]
            exist_tids = [track["id"] for track in vids_to_tracks[vid]]
            try:
                index1 = exist_tids.index(tid)
            except ValueError:
                index1 = -1
            if tid not in exist_tids:
                curr_track = {
                    "id": tid,
                    "category_id": ann["category_id"],
                    "video_id": vid,
                    "annotations": [ann],
                }
                vids_to_tracks[vid].append(curr_track)
            else:
                vids_to_tracks[vid][index1]["annotations"].append(ann)

            # fill in vids_to_imgs
            img_id = ann["image_id"]
            exist_img_ids = [img["id"] for img in vids_to_imgs[vid]]
            try:
                index2 = exist_img_ids.index(img_id)
            except ValueError:
                index2 = -1
            if index2 == -1:
                curr_img = {"id": img_id, "annotations": [ann]}
                vids_to_imgs[vid].append(curr_img)
            else:
                vids_to_imgs[vid][index2]["annotations"].append(ann)

        # sort annotations by frame index and compute track area
        for vid, tracks in vids_to_tracks.items():
            for track in tracks:
                track["annotations"] = sorted(
                    track["annotations"],
                    key=lambda x: images[x["image_id"]]["frame_id"],
                )
                # compute average area
                track["area"] = sum(x["area"]
                                    for x in track["annotations"]) / len(
                                        track["annotations"])

        # ensure all videos are present
        for vid_id in vid_ids:
            if vid_id not in vids_to_tracks.keys():
                vids_to_tracks[vid_id] = []
            if vid_id not in vids_to_imgs.keys():
                vids_to_imgs[vid_id] = []

        return vids_to_tracks, vids_to_imgs

    def _compute_image_to_timestep_mappings(self):
        """Computes a mapping from images to timestep in sequence."""
        images = {}
        for image in self.gt_data["images"]:
            images[image["id"]] = image

        seq_to_imgs_to_timestep = {
            vid["id"]: dict()
            for vid in self.gt_data["videos"]
        }
        for vid in seq_to_imgs_to_timestep:
            curr_imgs = [img["id"] for img in self.video2gt_image[vid]]
            curr_imgs = sorted(curr_imgs, key=lambda x: images[x]["frame_id"])
            seq_to_imgs_to_timestep[vid] = {
                curr_imgs[i]: i
                for i in range(len(curr_imgs))
            }

        return seq_to_imgs_to_timestep

    def _fill_video_ids_inplace(self, annotations):
        """Fills in missing video IDs inplace.

        Adapted from https://github.com/TAO-Dataset/.
        """
        missing_video_id = [x for x in annotations if "video_id" not in x]
        if missing_video_id:
            image_id_to_video_id = {
                x["id"]: x["video_id"]
                for x in self.gt_data["images"]
            }
            for x in missing_video_id:
                x["video_id"] = image_id_to_video_id[x["image_id"]]

    @staticmethod
    def _make_tk_ids_unique(annotations):
        """Makes track IDs unique over the whole annotation set.

        Adapted from https://github.com/TAO-Dataset/.
        """
        track_id_videos = {}
        track_ids_to_update = set()
        max_track_id = 0

        tk_str = get_track_id_str(annotations[0])
        for ann in annotations:
            t = int(ann[tk_str])
            if t not in track_id_videos:
                track_id_videos[t] = ann["video_id"]

            if ann["video_id"] != track_id_videos[t]:
                # track id is assigned to multiple videos
                track_ids_to_update.add(t)
            max_track_id = max(max_track_id, t)

        if track_ids_to_update:
            print("true")
            next_id = itertools.count(max_track_id + 1)
            new_tk_ids = defaultdict(lambda: next(next_id))
            for ann in annotations:
                t = ann[tk_str]
                v = ann["video_id"]
                if t in track_ids_to_update:
                    ann[tk_str] = new_tk_ids[t, v]
        return len(track_ids_to_update)
