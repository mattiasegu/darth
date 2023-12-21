
import time
from multiprocessing import Pool

import motmetrics as mm
import numpy as np
import pandas as pd

from mmcv.utils import print_log
from mmtrack.core.evaluation.eval_mot import (
    acc_single_video,
    aggregate_accs,
    eval_single_class,
    METRIC_MAPS
)

MULTIPLY_KEYS = [
    "IDF1",
    "MOTA",
    "MOTP",
    "Rcll",
    "Prcn",
]


def evaluate_mot(results,
                 annotations,
                 logger=None,
                 classes=None,
                 iou_thr=0.5,
                 ignore_iof_thr=0.5,
                 ignore_by_classes=False,
                 ignore_empty_classes=True,
                 nproc=4):
    """Evaluation CLEAR MOT metrics.

    Args:
        results (list[list[list[ndarray]]]): The first list indicates videos,
            The second list indicates images. The third list indicates
            categories. The ndarray indicates the tracking results.
        annotations (list[list[dict]]): The first list indicates videos,
            The second list indicates images. The third list indicates
            the annotations of each video. Keys of annotations are

            - `bboxes`: numpy array of shape (n, 4)
            - `labels`: numpy array of shape (n, )
            - `instance_ids`: numpy array of shape (n, )
            - `bboxes_ignore` (optional): numpy array of shape (k, 4)
            - `labels_ignore` (optional): numpy array of shape (k, )
        logger (logging.Logger | str | None, optional): The way to print the
            evaluation results. Defaults to None.
        classes (list, optional): Classes in the dataset. Defaults to None.
        iou_thr (float, optional): IoU threshold for evaluation.
            Defaults to 0.5.
        ignore_iof_thr (float, optional): Iof threshold to ignore results.
            Defaults to 0.5.
        ignore_by_classes (bool, optional): Whether ignore the results by
            classes or not. Defaults to False.
        ignore_empty_classes (bool, optional): Whether to ignore empty classes
            in AVERAGE computation. Defaults to False.
        nproc (int, optional): Number of the processes. Defaults to 4.

    Returns:
        dict[str, float]: Evaluation results.
    """
    print_log('---CLEAR MOT Evaluation---', logger)
    t = time.time()
    gts = annotations.copy()
    if classes is None:
        classes = [i + 1 for i in range(len(results[0]))]
    assert len(results) == len(gts)
    metrics = METRIC_MAPS.keys()

    print_log('Accumulating...', logger)

    pool = Pool(nproc)
    accs = pool.starmap(
        acc_single_video,
        zip(results, gts, [iou_thr for _ in range(len(gts))],
            [ignore_iof_thr for _ in range(len(gts))],
            [ignore_by_classes for _ in range(len(gts))]))
    names, accs, items = aggregate_accs(accs, classes)

    # ignore classes without any instance
    if ignore_empty_classes:
        valid_ids = [i for i, name in enumerate(names) if len(name) > 0]
        names = [names[i] for i in valid_ids]
        accs = [accs[i] for i in valid_ids]
        items = [items[i] for i in valid_ids]
    
    print_log('Evaluating...', logger)
    eval_results = pd.DataFrame(columns=metrics)
    summaries = pool.starmap(eval_single_class, zip(names, accs))
    pool.close()

    # category and overall results
    for i, item in enumerate(items):
        eval_results.loc[item] = summaries[i]

    dtypes = {m: type(d) for m, d in zip(metrics, summaries[0])}
    # average results
    avg_results = []
    for i, m in enumerate(metrics):
        v = np.array([s[i] for s in summaries[:len(classes)]])
        v = np.nan_to_num(v, nan=0)
        if dtypes[m] == int:
            avg_results.append(int(v.sum()))
        elif dtypes[m] == float:
            avg_results.append(float(v.mean()))
        else:
            raise TypeError()
    eval_results.loc['AVERAGE'] = avg_results
    eval_results = eval_results.astype(dtypes)
    eval_results = eval_results.to_dict()

    print_log(f'Evaluation finishes with {(time.time() - t):.2f} s.', logger)

    return eval_results