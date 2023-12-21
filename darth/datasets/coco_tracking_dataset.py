from copy import deepcopy
import os
import os.path as osp

import json
import tempfile
import shutil
from collections import defaultdict

from pandas import DataFrame
import mmcv
from mmcv.utils import print_log
from mmdet.datasets import DATASETS

from mmtrack.datasets import CocoVideoDataset

from .parsers import COCO, CocoVID
from .eval import (
    get_default_trackeval_config,
    get_default_trackeval_dataset_config,
    BDD100K,
    accumulate_results,
    rename_dict,
    pretty_logging,
    eval_mot,
    multiply_dict,
)
from .utils import convert_coco_results_to_scalabel
from darth.core.track.postprocessing import majority_vote


@DATASETS.register_module()
class CocoTrackingDataset(CocoVideoDataset):
    """Extended coco video dataset for VID, MOT and SOT tasks. Supporting
    additional tracking metrics from the CLEAR MOT, HOTA, and TETA evalluation.
    Args:
        attributes (Optional[Dict[str, ...]]): a dictionary containing the
            allowed attributes. Dataset samples will be filtered based on
            the allowed attributes. If None, load all samples. Default: None.
        cat_name_to_label (Optional[Dict[str, id]]): maps class names to label 
            ids. If None, self.cat2labels is inferred from classes. An example
            is cat_name_to_label = {'car': 0, 'pedestrian': 1}. A use case is
            when a model trained on another dataset based its output order on
            the cat_name_to_label inferred from the other dataset. To test that
            model on a new dataset with a different category order, set
            cat_name_to_label based on the checkpoint metadata. Default: None.
        cat_name_mapping (Optional[Dict[str, str]]): defines a mapping to
            change category names in cat_name_to_label. Can be used for example
            to define a mapping between anming conventions across different
            datasets. If None, it does not affect the naming convention. 
            Default: None.
    """

    def __init__(self,
                 attributes=None,
                 cat_name_to_label=None,
                 cat_name_mapping=None,
                 *args,
                 **kwargs):
        self.attributes = attributes
        super().__init__(*args, **kwargs)
        # override cat2label and cat_name_to_label
        if cat_name_to_label is not None:
            if cat_name_mapping is not None:
                cat_name_mapping_dummy = {
                    name: name
                    for name in cat_name_to_label
                    if name not in cat_name_mapping
                }
                cat_name_mapping.update(cat_name_mapping_dummy)
                cat_name_to_label = {
                    cat_name_mapping[name]: label
                    for (name, label) in cat_name_to_label.items()
                }
            self.cat2label = {
                self.cat_name_to_id[name]: label
                for (name, label) in cat_name_to_label.items()
                if name in self.CLASSES
            }
            self.cat_name_to_label = cat_name_to_label

    def set_category_mappings(self):
        """Sets category mappings:
        - self.cat_ids
        - self.cat2label
        - self.cat_name_to_label
        """
        # The order of cat_ids will not change with the order of CLASSES
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_name_to_label = {
            name: self.cat2label[self.coco.cat_name_to_id[name]]
            for name in self.CLASSES
        }
        self.label_to_cat_name = dict(
            (v,k) for k,v in self.cat_name_to_label.items())
        if hasattr(self.coco, "cat_to_vids"):
            self.eval_classes = [
                self.label_to_cat_name[self.cat2label[cat]]
                for cat in self.coco.cat_to_vids.keys()
                if cat in self.cat2label]
        elif hasattr(self.coco, "catToImgs"):
            self.eval_classes = [
                self.label_to_cat_name[self.cat2label[cat]]
                for cat in self.coco.catToImgs.keys()
                if cat in self.cat2label]
        self.eval_classes = list(set(self.eval_classes) & set(self.CLASSES)) 


    def load_image_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """
        self.coco = COCO(ann_file)
        self.set_category_mappings()
        self.img_ids = self.coco.get_img_ids(attributes=self.attributes)

        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i], cat_ids=self.cat_ids)
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def load_video_anns(self, ann_file):
        """Load annotations from COCOVID style annotation file.
        Args:
            ann_file (str): Path of annotation file.
        Returns:
            list[dict]: Annotation information from COCOVID api.
        """
        self.coco = CocoVID(ann_file)
        self.set_category_mappings()

        data_infos = []
        self.vid_ids = self.coco.get_vid_ids(attributes=self.attributes)
        self.img_ids = []
        for vid_id in self.vid_ids:
            img_ids = self.coco.get_img_ids_from_vid(vid_id)
            if self.key_img_sampler is not None:
                img_ids = self.key_img_sampling(img_ids,
                                                **self.key_img_sampler)
            self.img_ids.extend(img_ids)
            for img_id in img_ids:
                info = self.coco.load_imgs([img_id])[0]
                info['filename'] = info['file_name']
                data_infos.append(info)
        return data_infos

    def load_annotations(self, ann_file):
        """Load annotations from COCO/COCOVID style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation information from COCO/COCOVID api.
        """
        if not self.load_as_video:
            data_infos = self.load_image_annotations(ann_file)
        else:
            data_infos = self.load_video_anns(ann_file)
        return data_infos

    def get_default_teta_configs(self, output_dir, result_files):
        """Returns default teta evaluation and dataset configs."""
        import teta
        teta_eval_config = teta.config.get_default_eval_config()
        teta_eval_config["DISPLAY_LESS_PROGRESS"] = True
        teta_eval_config["OUTPUT_SUMMARY"] = False
        teta_eval_config["PRINT_CONFIG"] = False
        teta_eval_config["NUM_PARALLEL_CORES"] = 8
        teta_dataset_config = teta.config.get_default_dataset_config()
        teta_dataset_config["TRACKERS_TO_EVAL"] = ['TETA']
        teta_dataset_config["OUTPUT_FOLDER"] = output_dir
        teta_dataset_config["GT_FOLDER"] = self.ann_file
        teta_dataset_config["PRINT_CONFIG"] = False
        teta_dataset_config["TRACKER_SUB_FOLDER"] = result_files[
            "track_coco"]
        return teta_eval_config, teta_dataset_config

    def get_default_trackeval_configs(self, output_dir, result_files):
        """Returns default trackeval evaluation and dataset configs."""
        trackeval_config = get_default_trackeval_config()
        trackeval_config["DISPLAY_LESS_PROGRESS"] = True
        trackeval_config["NUM_PARALLEL_CORES"] = 8
        trackeval_dataset_config = get_default_trackeval_dataset_config(
        )
        trackeval_dataset_config["CLASSES_TO_EVAL"] = self.eval_classes
        trackeval_dataset_config["TRACKERS_TO_EVAL"] = ['track']
        trackeval_dataset_config["OUTPUT_FOLDER"] = output_dir
        trackeval_dataset_config["GT_FOLDER"] = self.ann_file
        trackeval_dataset_config["TRACKER_SUB_FOLDER"] = result_files[
            "track_coco"]
        return trackeval_config, trackeval_dataset_config

    def _track2json(self, results):
        """Convert tracking results to TAO json style."""
        inds = [
            i for i, info in enumerate(self.data_infos)
            if info["frame_id"] == 0
        ]
        num_vids = len(inds)
        inds.append(len(self.data_infos))
        results = [results[inds[i]:inds[i + 1]] for i in range(num_vids)]
        img_infos = [
            self.data_infos[inds[i]:inds[i + 1]] for i in range(num_vids)
        ]

        json_results = []
        max_track_id = 0
        for _img_infos, _results in zip(img_infos, results):
            track_ids = []
            for img_info, result in zip(_img_infos, _results):
                img_id = img_info["id"]
                for label in range(len(result)):
                    bboxes = result[label]
                    for i in range(bboxes.shape[0]):
                        data = dict()
                        data["image_id"] = img_id
                        data["bbox"] = self.xyxy2xywh(bboxes[i, 1:])
                        data["score"] = float(bboxes[i][-1])
                        if len(result) != len(self.cat_ids):
                            data["category_id"] = label + 1
                        else:
                            data["category_id"] = self.cat_ids[label]
                        data["video_id"] = img_info["video_id"]
                        data["track_id"] = max_track_id + int(bboxes[i][0])
                        track_ids.append(int(bboxes[i][0]))
                        json_results.append(data)
            track_ids = list(set(track_ids))
            if track_ids:
                max_track_id += max(track_ids) + 1

        return json_results

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data["image_id"] = img_id
                    data["bbox"] = self.xyxy2xywh(bboxes[i])
                    data["score"] = float(bboxes[i][4])
                    # if the object detector is trained on 1230 classes(lvis 0.5)
                    if len(result) != len(self.cat_ids):
                        data["category_id"] = label + 1
                    else:
                        data["category_id"] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def postprocess_track(self, results, majority_voting=False):
        """Applies postprocessing operations to track_results."""
        if results:
            # split frames by videos
            inds = [
                i for i, info in enumerate(self.data_infos)
                if info["frame_id"] == 0
            ]
            num_vids = len(inds)
            inds.append(len(self.data_infos))
            video_results = [results[inds[i]:inds[i + 1]] for i in range(num_vids)]

            # applied post-processing
            if majority_voting:
                video_results = majority_vote(video_results)

            # concatenate video frames back together
            results = []
            for result in video_results:
                results += result
        return results

    def format_results(
        self,
        results,
        results_path=None,
        scalabel=False,
        majority_voting=False,
        append_empty_preds=True,
    ):
        """Format the results to json (standard format for COCO evaluation).
        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            results_path (Optional[str]): optional directory where to save the
                results.
            scalabel (bool): if True, return results also in the Scalabel
                format. Default: False.
            majority_voting (bool): if True, apply majority class voting for
                postprocessing of tracklets.
            append_empty_preds: if True, append scalabel frames for which no
                prediction was provided. Necessary for submitting to the 
                evaluation benchmark. Default: True.
        Returns:
            tuple: (results, result_files, output_dir), results are the results
                passed as inputs to which postpreocessing have been applied;
                result_files is a dict containing the json filepaths; output_dir
                is the temporary directory created for saving json files when
                results_path is not specified.
        """
        assert isinstance(results, dict), "results must be a dict"

        if results_path is None:
            output_dir = tempfile.mkdtemp()
            results_path = output_dir
        else:
            output_dir = results_path
        os.makedirs(results_path, exist_ok=True)

        result_files = dict()

        if scalabel:
            # get complete dataset info from the annotation file
            bdd_scalabel_gt = json.load(open(self.ann_file))
            bdd_cat_id_to_info = {}
            for cat in bdd_scalabel_gt["categories"]:
                if cat["id"] not in bdd_cat_id_to_info:
                    bdd_cat_id_to_info[cat["id"]] = cat
            # image id to info mapping
            image_id_to_info = {}
            for image in bdd_scalabel_gt["images"]:
                if image["id"] not in image_id_to_info:
                    image_id_to_info[image["id"]] = image
            # video id to info mapping
            video_id_to_info = {}
            for video in bdd_scalabel_gt["videos"]:
                if video["id"] not in video_id_to_info:
                    video_id_to_info[video["id"]] = video

        if "track_bboxes" in results:
            results["track_bboxes"] = self.postprocess_track(
                results["track_bboxes"], majority_voting=majority_voting
            )
            track_results = self._track2json(results["track_bboxes"])

            if scalabel:
                track_results_scalabel = convert_coco_results_to_scalabel(
                    track_results,
                    bdd_cat_id_to_info,
                    image_id_to_info,
                    video_id_to_info,
                    append_empty_preds=append_empty_preds,
                )
                result_files[
                    "track_scalabel"] = f"{results_path}/track_scalabel.json"
                mmcv.dump(track_results_scalabel,
                            result_files["track_scalabel"])

            result_files["track_coco"] = f"{results_path}/track_coco.json"
            mmcv.dump(track_results, result_files["track_coco"])

        if "det_bboxes" in results:
            bbox_results = self._det2json(results["det_bboxes"])

            if scalabel:
                bbox_results_scalabel = convert_coco_results_to_scalabel(
                    bbox_results,
                    bdd_cat_id_to_info,
                    image_id_to_info,
                    video_id_to_info,
                    append_empty_preds=append_empty_preds,
                )
                result_files[
                    "bbox_scalabel"] = f"{results_path}/bbox_scalabel.json"
                mmcv.dump(bbox_results_scalabel,
                            result_files["bbox_scalabel"])

            result_files["bbox_coco"] = f"{results_path}/bbox_coco.json"
            mmcv.dump(bbox_results, result_files["bbox_coco"])

        vid_infos = self.coco.load_vids(self.vid_ids)
        names = [_['name'] for _ in vid_infos]

        return results, result_files, names, output_dir

    def evaluate_coco(self,
                      results,
                      metric='bbox',
                      logger=None,
                      jsonfile_prefix=None,
                      classwise=False,
                      proposal_nums=(100, 300, 1000),
                      iou_thrs=None,
                      metric_items=None):
        """Evaluation in COCO protocol. This method is based on the evaluate 
        method from mmdet.datasets.coco.CocoDataset, and is needed to call its
        format_results method which is here overridden.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """

        metrics = metric if isinstance(metric, list) else [metric]
        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        coco_gt = self.coco
        self.cat_ids = coco_gt.get_cat_ids(cat_names=self.CLASSES)

        result_files, tmp_dir = super(CocoVideoDataset, self).format_results(
            results, jsonfile_prefix)
        eval_results = self.evaluate_det_segm(results, result_files, coco_gt,
                                              metrics, logger, classwise,
                                              proposal_nums, iou_thrs,
                                              metric_items)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

    def evaluate(
        self,
        results,
        metric=["bbox", "track"],
        logger=None,
        results_path=None,
        results_kwargs=dict(
            scalabel=False,
            append_empty_preds=True,
        ),
        bbox_kwargs=dict(
            classwise=True,
            proposal_nums=(100, 300, 1000),
            iou_thrs=None,
            metric_items=None,
        ),
        track_kwargs=dict(
            iou_thr=0.5,
            ignore_iof_thr=0.5,
            ignore_by_classes=False,
            nproc=1,
            majority_voting=False,
        ),
    ):
        """Evaluation in COCO protocol, CLEAR MOT metrics (e.g. MOTA, IDF1),
        HOTA metrics and TETA metrics.
        Args:
            results (dict): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'track', 'hota', 'teta'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            results_path (Optional[str]): optional path to a directory or to
                a json file. If it is a json file, the json file will be
                loaded and used as results to evaluate; if it is a
                directory, results will be saved to that directory; if None,
                a temporary directory will be used instead. Default: None.
            results_kwargs (dict): Configuration for results loading and saving.
                scalabel (bool): Whether to save results also in scalabel format
                    for submission to BDD eval. Default: False
                append_empty_preds (bool): must be True for submission to BDD
                    evaluation benchmark. Default: True.
            bbox_kwargs (dict): Configuration for COCO styple evaluation.
            track_kwargs (dict): Configuration for CLEAR MOT evaluation.
        Returns:
            dict[str, float]: COCO style and CLEAR MOT evaluation metric.
        """
        if isinstance(metric, list):
            metrics = metric
        elif isinstance(metric, str):
            metrics = [metric]
        else:
            raise TypeError("metric must be a list or a str.")
        allowed_metrics = ["bbox", "track", "hota", "teta"]
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f"metric {metric} is not supported.")

        results, result_files, _, output_dir = self.format_results(
            results,
            results_path,
            majority_voting=track_kwargs["majority_voting"],
            scalabel=results_kwargs["scalabel"],
            append_empty_preds=results_kwargs["append_empty_preds"])

        eval_results = defaultdict(lambda: {})

        if "track" in metrics:
            print_log('Evaluating CLEAR MOT metrics...', logger=logger)
            assert len(self.data_infos) == len(results['track_bboxes'])
            inds = [
                i for i, _ in enumerate(self.data_infos) if _['frame_id'] == 0
            ]
            num_vids = len(inds)
            inds.append(len(self.data_infos))

            track_bboxes = [
                results['track_bboxes'][inds[i]:inds[i + 1]]
                for i in range(num_vids)
            ]
            ann_infos = [self.get_ann_info(_) for _ in self.data_infos]
            ann_infos = [
                ann_infos[inds[i]:inds[i + 1]] for i in range(num_vids)
            ]

            mot_kwargs = deepcopy(track_kwargs)
            mot_kwargs.pop("majority_voting")
            mot_results = eval_mot.evaluate_mot(
                results=track_bboxes,
                annotations=ann_infos,
                logger=logger,
                classes=self.CLASSES,
                ignore_empty_classes=True,
                **mot_kwargs)
            rename_dict(mot_results, eval_mot.METRIC_MAPS)
            multiply_dict(
                mot_results, keys=eval_mot.MULTIPLY_KEYS, factor=100
            )
            eval_results['track'].update(mot_results)

        if json.load(open(result_files["track_coco"], "r")):
            if "teta" in metrics:
                try:
                    import teta
                except ImportError:
                    raise ImportError(
                        'Please run'
                        'pip install git+https://github.com/SysCV/tet.git#egg=teta\&subdirectory=teta'  # noqa
                        'to manually install teta')

                print_log('Evaluating TETA metrics...', logger=logger)
                eval_config, dataset_config = self.get_default_teta_configs(
                    output_dir, result_files
                )
                teta_evaluator = teta.Evaluator(eval_config)
                teta_dataset = teta.datasets.COCO(dataset_config)
                teta_results, _ = teta_evaluator.evaluate(
                    [teta_dataset], [teta.metrics.TETA(exhaustive=True)])
                        
                keep_metrics = [
                    "TETA", "LocA", "AssocA", "ClsA", "LocRe", "LocPr", "AssocRe",
                    "AssocPr", "ClsRe", "ClsPr"
                ]
                metric_mapping = {
                    "AssocRe": "AssRe", "AssocPr": "AssPr", "AssocA": "AssA"
                }
                teta_results, _ = accumulate_results(
                    teta_results['COCO']['TETA']['COMBINED_SEQ'], keep_metrics
                )
                rename_dict(teta_results, metric_mapping)

                eval_results['teta'].update(teta_results)

            if "track" in metrics or "hota" in metrics:
                try:
                    import trackeval
                except ImportError:
                    raise ImportError(
                        'Please run'
                        'pip install git+https://github.com/JonathonLuiten/TrackEval.git'  # noqa
                        'to manually install trackeval')

                print_log('Evaluating HOTA metrics...', logger=logger)
                eval_config, dataset_config = self.get_default_trackeval_configs(
                    output_dir, result_files
                )
                # (using BDD dataset as default)
                trackeval_dataset = BDD100K(dataset_config)
                trackeval_evaluator = trackeval.Evaluator(eval_config)
                trackeval_metrics = [trackeval.metrics.HOTA({"PRINT_CONFIG": False})]
                hota_results, _ = trackeval_evaluator.evaluate(
                    [trackeval_dataset], trackeval_metrics)

                keep_metrics = [
                    "HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA",
                    "Dets", "IDs"
                ]
                hota_results, _ = accumulate_results(
                    hota_results['BDD100K']['track']['COMBINED_SEQ'], keep_metrics
                )
                rename_dict(hota_results, metric_mapping={})
                eval_results['hota'].update(hota_results)

        # evaluate detectors without tracker
        if "bbox" in metrics:
            if isinstance(results, dict):
                super_results = results['det_bboxes']
            elif isinstance(results, list):
                super_results = results
            else:
                raise TypeError('Results must be a dict or a list.')
            print_log('Evaluating COCO metrics...', logger=logger)
            super_eval_results = self.evaluate_coco(
                results=super_results,
                metric=["bbox"],
                logger=logger,
                **bbox_kwargs)
            metric_mapping = {
                'bbox_mAP': 'mAP',
                'bbox_mAP_50': 'mAP_50',
                'bbox_mAP_75': 'mAP_75',
                'bbox_mAP_s': 'mAP_s',
                'bbox_mAP_m': 'mAP_m',
                'bbox_mAP_l': 'mAP_l',
                }
            super_eval_results = {
                metric_mapping[k]: {'OVERALL': 100*v}
                for k, v in super_eval_results.items()
                if k in metric_mapping}
            eval_results["bbox"].update(super_eval_results)

        summary_dict = defaultdict(lambda: {})
        if (
            "track" in eval_results and
            "MOTA" in eval_results["track"] and
            "IDF1" in eval_results["track"] and
            "hota" in eval_results and
            "LocA" in eval_results["hota"] and
            "DetA" in eval_results["hota"] and
            "AssA" in eval_results["hota"] and
            "HOTA" in eval_results["hota"]
        ):
            summary_dict["LocA"] = eval_results["hota"]["LocA"]
            summary_dict["DetA"] = eval_results["hota"]["DetA"]
            summary_dict["MOTA"] = eval_results["track"]["MOTA"]
            summary_dict["HOTA"] = eval_results["hota"]["HOTA"]
            summary_dict["IDF1"] = eval_results["track"]["IDF1"]
            summary_dict["AssA"] = eval_results["hota"]["AssA"]
            eval_results["summary"].update(summary_dict)

        count = {"track": 2, "hota": 2, "summary": 2, 'teta': 1, 'bbox': 0}
        if output_dir is not None:
            for key, results_dict in eval_results.items():
                pd_frame = DataFrame.from_dict(results_dict)            
                pretty_logging(
                    pd_frame=pd_frame,
                    global_count=count[key],
                    logger=logger
                )
                pd_frame.to_csv(os.path.join(output_dir, f"{key}_metrics.csv"))
                
        # remove output_dir if it was temporary
        if results_path is None and output_dir is not None:
            shutil.rmtree(output_dir)

        # prepare results for logging
        log_dict = {}
        for metric, results in eval_results.items():
            for submetric, result in results.items():
                for cat, res in result.items():
                    log_dict[f"{metric}/{cat}/{submetric}"] = res
        return log_dict
