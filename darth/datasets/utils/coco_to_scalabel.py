"""COCO to Scalabel conversion utils."""


def convert_coco_pred_to_scalabel(coco_pred, bdd_cat_id_to_info):
    """
    Convert the single prediction result to label format for bdd
    coco_pred:
        'image_id': 1,
         'bbox': [998.872802734375,
          379.5665283203125,
          35.427490234375,
          59.21759033203125],
         'score': 0.9133418202400208,
         'category_id': 1,
         'video_id': 1,
         'track_id': 16
    - labels [ ]: list of dicts
        - id: string
        - category: string
        - box2d:
           - x1: float
           - y1: float
           - x2: float
           - y2: float
    Args:
        coco_pred: coco_pred dict.
        bdd_cat_id_to_info: bdd category id to category infomation mapping.
    Return:
        the input coco prediction converted to the scalabel format.
    """
    scalabel_pred = {}
    scalabel_pred["id"] = coco_pred["track_id"] if "track_id" in coco_pred else 0
    scalabel_pred["score"] = coco_pred["score"]
    scalabel_pred["category"] = bdd_cat_id_to_info[coco_pred["category_id"]]["name"]
    scalabel_pred["box2d"] = {
        "x1": coco_pred["bbox"][0],
        "y1": coco_pred["bbox"][1],
        "x2": coco_pred["bbox"][0] + coco_pred["bbox"][2] - 1,
        "y2": coco_pred["bbox"][1] + coco_pred["bbox"][3] - 1,
    }
    return scalabel_pred


def convert_coco_results_to_scalabel(
    coco_results,
    bdd_cat_id_to_info,
    image_id_to_info,
    video_id_to_info,
    append_empty_preds=True,
):
    """
    Converts COCO results to the scalabel format.

    Args:
        results: list of coco predictions
        bdd_cat_id_to_info: bdd category id to category information mapping.
        image_id_to_info: image id to image information mapping.
        video_id_to_info: video id to video information mapping.
        append_empty_preds: if True, append scalabel frames for which no
            prediction was provided. Necessary for submitting to the evaluation
            benchmark. Default: True.
    Return:
        A submittable result for the bdd evaluation
    """

    bdd_results = {}
    for result in coco_results:
        id = result["image_id"]
        if id not in bdd_results:
            bdd_result = {}
            bdd_result["name"] = image_id_to_info[id]["file_name"].split("/")[-1]
            bdd_result["videoName"] = video_id_to_info[
                image_id_to_info[id]["video_id"]
            ]["name"]
            bdd_result["frameIndex"] = image_id_to_info[id]["frame_id"]
            bdd_result["labels"] = [
                convert_coco_pred_to_scalabel(result, bdd_cat_id_to_info)
            ]
            bdd_results[id] = bdd_result
        else:
            bdd_results[id]["labels"].append(
                convert_coco_pred_to_scalabel(result, bdd_cat_id_to_info)
            )

    # add entries for images without predictions
    if append_empty_preds:
        for key in image_id_to_info:
            if key not in bdd_results:
                bdd_result = {}
                bdd_result["name"] = image_id_to_info[key]["file_name"]
                bdd_result["videoName"] = video_id_to_info[
                    image_id_to_info[key]["video_id"]
                ]["name"]
                bdd_result["frameIndex"] = image_id_to_info[key]["frame_id"]
                bdd_result["labels"] = []
                bdd_results[key] = bdd_result
    return list(bdd_results.values())
