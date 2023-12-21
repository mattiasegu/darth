"""Utils for formatting and logging evaluation results."""
from collections import defaultdict

import numpy as np


def table(data_frame, global_count) -> str:
    """Convert data model into a table for formatted printing.
    Args:
        include (set[str]): Optional, the metrics to convert
        exclude (set[str]): Optional, the metrics not to convert
    Returns:
        table (str): the exported table string
    """
    summary = data_frame.to_string(float_format=lambda num: f"{num:.1f}")
    summary = summary.replace("NaN", " - ")
    strs = summary.split("\n")
    split_line = "-" * len(strs[0])

    for row_ind in [1, -global_count]:
        strs.insert(row_ind, split_line)
    summary = "".join([f"{s}\n" for s in strs])
    summary = "\n" + summary
    return str(summary)


def accumulate_results(overall_results, keep_metrics):
    results_dict = defaultdict(lambda: {})

    global_count = 0
    for cls in overall_results:
        if cls == "cls_comb_cls_av":
            cls_name = "AVERAGE"
            global_count += 1
        elif cls == "cls_comb_det_av":
            cls_name = "OVERALL"
            global_count += 1
        elif cls == "average":
            cls_name = "AVERAGE"
            global_count += 1
        else:
            cls_name = cls

        def match_key(d):
            for k, v in d.items():
                if isinstance(v, dict):
                    match_key(v)
                else:
                    if k in keep_metrics:
                        #hardcoded
                        if type(v) == np.float64 and k != "Frag":
                            v *= 100
                        elif type(v) == np.ndarray:
                            v = v.mean() * 100
                        else:
                            v = int(v)

                        results_dict[k][cls_name] = v

        match_key(overall_results[cls])

    return results_dict, global_count


def pretty_logging(pd_frame, global_count=0, logger=None):
    """Pretty logs the results as a table."""
    if logger is not None:
        logger.info(table(pd_frame, global_count))
    else:
        print(table(pd_frame, global_count))


def rename_dict(inputs, metric_mapping):
    """Renames dictionary keys."""
    for old, new in metric_mapping.items():
        inputs[new] = inputs.pop(old)


def multiply_dict(results, keys=[], factor=100):
    """Multiplies leaf values of a results dictionary."""
    for key in keys:
        if key in results:
            for cls in results[key]:
                results[key][cls] *= factor
        
