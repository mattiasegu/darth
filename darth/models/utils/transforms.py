import torch
import numpy as np


def proposals2bboxes(proposals):
    """Process detections to bboxes, labels and confidences."""
    bboxes = proposals[:, :-1]
    confidences = proposals[:, -1]
    
    return bboxes, confidences


def detections2bboxes(detections):
    """Process detections to bboxes, labels and confidences."""
    bboxes = []
    labels = []
    confidences = []
    for cls_dets in detections:
        bbox = []
        label = []
        confidence = []
        for cls, dets in enumerate(cls_dets):
            bbox.append(dets[:, :-1])
            cls_conf = dets[:, -1]
            confidence.append(cls_conf)
            label.append(cls*np.ones_like(cls_conf))
        
        bboxes.append(np.concatenate(bbox))
        labels.append(np.concatenate(label))
        confidences.append(np.concatenate(confidence))
    
    return bboxes, labels, confidences


def filter_bboxes_by_confidence(bboxes, labels, confidences, conf_thr=0.0):
    """Filter bboxes by confidence threshold. Returns the filtered bboxes and
    the corresponding labels and confidences"""
    filtered_bboxes = []
    filtered_labels = []
    filtered_confidences = []
    for bbox, label, conf in zip(bboxes, labels, confidences):
        valid_ids = np.where(conf > conf_thr)
        filtered_bboxes.append(bbox[valid_ids])
        filtered_labels.append(label[valid_ids])
        filtered_confidences.append(conf[valid_ids])

    return filtered_bboxes, filtered_labels, filtered_confidences


def bboxes_to_tensor(bboxes, labels, confidences, device):
    """Filter bboxes by confidence threshold. Returns the filtered bboxes and
    the corresponding labels and confidences"""
    tensor_bboxes = []
    tensor_labels = []
    tensor_confidences = []
    for bbox, label, conf in zip(bboxes, labels, confidences):
        tensor_bbox = torch.from_numpy(bbox).float().to(device)
        tensor_label = torch.from_numpy(label).long().to(device)
        tensor_confidence = torch.from_numpy(conf).float().to(device)
        tensor_bboxes.append(tensor_bbox)
        tensor_labels.append(tensor_label)
        tensor_confidences.append(tensor_confidence)

    return tensor_bboxes, tensor_labels, tensor_confidences