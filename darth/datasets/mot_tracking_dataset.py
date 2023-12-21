import os
import os.path as osp
import shutil
from collections import defaultdict
import tempfile

from pandas import DataFrame
import motmetrics as mm

import mmcv
from mmcv.utils import print_log
from mmdet.core import eval_map
from mmdet.datasets import DATASETS
from mmtrack.datasets import CocoVideoDataset
from mmtrack.datasets.mot_challenge_dataset import MOTChallengeDataset

try:
    import trackeval
except ImportError:
    trackeval = None

from darth.datasets.eval import (
    accumulate_results,
    rename_dict,
    pretty_logging,
    eval_mot,
    multiply_dict,
)


@DATASETS.register_module()
class MOTTrackingDataset(MOTChallengeDataset):
    """Dataset for the MOT Challenge datasets: https://www.bdd100k.com/."""

    CLASSES = ("pedestrian", )

    def __init__(self,
                 gt_root=None,
                 *args,
                 **kwargs):
        self.gt_root = gt_root
        super().__init__(*args, **kwargs)

    def format_results(self, results, results_path=None, metrics=['track']):
        """Format the results to txts (standard format for MOT Challenge).

        Args:
            results (dict(list[ndarray])): Testing results of the dataset.
            results_path (str, optional): Path to save the formatted results.
                Defaults to None.
            metrics (list[str], optional): The results of the specific metrics
                will be formatted. Defaults to ['track'].

        Returns:
            tuple: (results_path, resfiles, names, tmp_dir), results_path is
            the path to save the formatted results, resfiles is a dict
            containing the filepaths, names is a list containing the name of
            the videos, tmp_dir is the temporal directory created for saving
            files.
        """
        assert isinstance(results, dict), 'results must be a dict.'
        if results_path is None:
            output_dir = tempfile.mkdtemp()
            results_path = output_dir
        else:
            output_dir = results_path

        resfiles = dict()
        for metric in metrics:
            resfiles[metric] = osp.join(results_path, metric)
            os.makedirs(resfiles[metric], exist_ok=True)

        inds = [i for i, _ in enumerate(self.data_infos) if _['frame_id'] == 0]
        num_vids = len(inds)
        assert num_vids == len(self.vid_ids)
        inds.append(len(self.data_infos))
        vid_infos = self.coco.load_vids(self.vid_ids)
        names = [_['name'] for _ in vid_infos]

        for i in range(num_vids):
            if 'bbox' in resfiles:
                self.format_bbox_results(
                    results['det_bboxes'][inds[i]:inds[i + 1]],
                    self.data_infos[inds[i]:inds[i + 1]],
                    f"{resfiles['bbox']}/{names[i]}.txt")
            # TODO: add majority vote in format_track_results
            if 'track' in resfiles:
                self.format_track_results(
                    results['track_bboxes'][inds[i]:inds[i + 1]],
                    self.data_infos[inds[i]:inds[i + 1]],
                    f"{resfiles['track']}/{names[i]}.txt")

        return results_path, resfiles, names, output_dir

    def evaluate(self,
                 results,
                 metric='track',
                 logger=None,
                 results_path=None,
                 bbox_kwargs=dict(
                     classwise=True,
                     proposal_nums=(100, 300, 1000),
                     iou_thrs=0.5,
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
        """Evaluation in MOT Challenge.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'track'. Defaults to 'track'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            results_path (str, optional): Path to save the formatted results.
                Defaults to None.
            bbox_iou_thr (float, optional): IoU threshold for detection
                evaluation. Defaults to 0.5.
            track_iou_thr (float, optional): IoU threshold for tracking
                evaluation.. Defaults to 0.5.

        Returns:
            dict[str, float]: MOTChallenge style evaluation metric.
        """
        eval_results = defaultdict(lambda: {})
        if isinstance(metric, list):
            metrics = metric
        elif isinstance(metric, str):
            metrics = [metric]
        else:
            raise TypeError('metric must be a list or a str.')
        allowed_metrics = ['bbox', 'track']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported.')

        results_path, resfiles, names, output_dir = self.format_results(
            results, results_path, metrics)

        if 'track' in metrics:
            if trackeval is None:
                raise ImportError(
                    'Please run'
                    'pip install git+https://github.com/JonathonLuiten/TrackEval.git'  # noqa
                    'to manually install trackeval')

            print_log('Evaluate CLEAR MOT results.', logger=logger)
            distth = 1 - track_kwargs['iou_thr']
            accs = []
            # support loading data from ceph
            local_dir = tempfile.TemporaryDirectory()

            for name in names:
                if 'half-train' in self.ann_file:
                    gt_file = osp.join(self.img_prefix,
                                       f'{name}/gt/gt_half-train.txt')
                elif 'half-val' in self.ann_file:
                    gt_file = osp.join(self.img_prefix,
                                       f'{name}/gt/gt_half-val.txt')
                else:
                    gt_file = osp.join(self.img_prefix, f'{name}/gt/gt.txt')

                if self.gt_root is not None:
                    gt_file = osp.join(self.gt_root, gt_file)

                res_file = osp.join(resfiles['track'], f'{name}.txt')
                # copy gt file from ceph to local temporary directory
                gt_dir_path = osp.join(local_dir.name, name, 'gt')
                os.makedirs(gt_dir_path)
                copied_gt_file = osp.join(
                    local_dir.name,
                    gt_file.replace(gt_file.split(name)[0], ''))

                f = open(copied_gt_file, 'wb')
                gt_content = self.file_client.get(gt_file)
                if hasattr(gt_content, 'tobytes'):
                    gt_content = gt_content.tobytes()
                f.write(gt_content)
                f.close()
                # copy sequence file from ceph to local temporary directory
                copied_seqinfo_path = osp.join(local_dir.name, name,
                                               'seqinfo.ini')
                f = open(copied_seqinfo_path, 'wb')
                info_file = osp.join(self.img_prefix, name, 'seqinfo.ini')
                if self.gt_root is not None:
                    info_file = osp.join(self.gt_root, info_file)
                seq_content = self.file_client.get(info_file)
                if hasattr(seq_content, 'tobytes'):
                    seq_content = seq_content.tobytes()
                f.write(seq_content)
                f.close()

                gt = mm.io.loadtxt(copied_gt_file)
                res = mm.io.loadtxt(res_file)
                if osp.exists(copied_seqinfo_path
                              ) and 'MOT15' not in self.img_prefix:
                    acc, ana = mm.utils.CLEAR_MOT_M(
                        gt, res, copied_seqinfo_path, distth=distth)
                else:
                    acc = mm.utils.compare_to_groundtruth(
                        gt, res, distth=distth)
                accs.append(acc)

            mh = mm.metrics.create()
            summary = mh.compute_many(
                accs,
                names=names,
                metrics=mm.metrics.motchallenge_metrics,
                generate_overall=True)
            
            mot_results = {
                eval_mot.METRIC_MAPS[k]: {'pedestrian': v["OVERALL"]} 
                for k, v in summary.to_dict().items()
                if k in eval_mot.METRIC_MAPS
            }
            multiply_dict(
                mot_results, keys=eval_mot.MULTIPLY_KEYS, factor=100
            )
            eval_results['track'].update(mot_results)

            seqmap = osp.join(results_path, 'videoseq.txt')
            with open(seqmap, 'w') as f:
                f.write('name\n')
                for name in names:
                    f.write(name + '\n')
                f.close()

            print_log('Evaluate HOTA results.', logger=logger)
            eval_config = trackeval.Evaluator.get_default_eval_config()
            evaluator = trackeval.Evaluator(eval_config)

            # tracker's name is set to 'track',
            # so this word needs to be splitted out
            output_folder = resfiles['track'].rsplit(os.sep, 1)[0]
            dataset_config = self.get_dataset_cfg_for_hota(
                local_dir.name, output_folder, seqmap)
            dataset = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
            hota_metrics = [
                trackeval.metrics.HOTA(
                    dict(METRICS=['HOTA'], THRESHOLD=0.5, PRINT_CONFIG=False)
                )
            ]
            output_res, _ = evaluator.evaluate(dataset, hota_metrics)
            res = output_res['MotChallenge2DBox']['track']['COMBINED_SEQ']['pedestrian']  # no-qa
            output_res['MotChallenge2DBox']['track']['COMBINED_SEQ']['pedestrian'] = {  # no-qa
                **res["HOTA"], **res["Count"]
            }
            keep_metrics = [
                "HOTA", "DetA", "AssA", "DetRe", "DetPr", "AssRe", "AssPr", "LocA",
                "Dets", "IDs"
            ]
            hota_results, _ = accumulate_results(
                output_res['MotChallenge2DBox']['track']['COMBINED_SEQ'], keep_metrics
            )
            rename_dict(hota_results, metric_mapping={})
            eval_results['hota'].update(hota_results)

            local_dir.cleanup()

        if 'bbox' in metrics:
            if isinstance(results, dict):
                bbox_results = results['det_bboxes']
            elif isinstance(results, list):
                bbox_results = results
            else:
                raise TypeError('results must be a dict or a list.')

            print_log('Evaluate mAP results.', logger=logger)
            annotations = [self.get_ann_info(info) for info in self.data_infos]
            mean_ap, _ = eval_map(
                bbox_results,
                annotations,
                iou_thr=bbox_kwargs['iou_thrs'],
                dataset=self.CLASSES,
                logger=logger)
            eval_results['bbox'] = {'mAP': {'pedestrian': 100*mean_ap}}

        summary_dict = defaultdict(lambda: {})
        summary_dict["LocA"] = eval_results["hota"]["LocA"]
        summary_dict["DetA"] = eval_results["hota"]["DetA"]
        summary_dict["MOTA"] = eval_results["track"]["MOTA"]
        summary_dict["HOTA"] = eval_results["hota"]["HOTA"]
        summary_dict["IDF1"] = eval_results["track"]["IDF1"]
        summary_dict["AssA"] = eval_results["hota"]["AssA"]
        eval_results["summary"].update(summary_dict)

        count = {"track": 0, "hota": 0, "bbox": 0, "summary": 0}
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