"""Detection adapters for the declared model output contracts."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..models.manifest import ModelManifest
from .contract import (
    BackendOutput,
    Detection,
    DetectionAdapter,
    DetectionBatch,
    DetectionValidationError,
    ImageTransform,
    PreparedInput,
    StillImage,
)
from .preprocessing import enrich_transform, prepare_image


class UltralyticsYoloNmsAdapter:
    """Decode Ultralytics boxes and confidence outputs with deterministic NMS."""

    def preprocess(self, image: StillImage, manifest: ModelManifest) -> PreparedInput:
        """Prepare an image and attach the manifest output contract."""
        return enrich_transform(prepare_image(image, manifest), manifest)

    def decode(self, output: BackendOutput, transform: ImageTransform) -> DetectionBatch:
        """Decode named NMS outputs or the declared raw-output fallback."""
        values = output.values
        output_names = transform.output_name_map
        boxes_name = output_names.get("boxes")
        scores_name = output_names.get("scores")
        classes_name = output_names.get("classes") or transform.class_output_name

        rejected = 0
        if boxes_name is not None and scores_name is not None:
            if boxes_name not in values or scores_name not in values:
                raise DetectionValidationError(
                    "backend output is missing the manifest-declared boxes or scores name"
                )
            boxes = _normalise_boxes(values[boxes_name], transform.box_format)
            scores, inferred_classes = _normalise_scores(values[scores_name], len(boxes))
            classes = inferred_classes
            if classes_name is not None and classes_name in values:
                classes = _normalise_classes(values[classes_name], len(boxes))
        elif transform.raw_output_name is not None:
            if transform.raw_output_name not in values:
                raise DetectionValidationError(
                    "backend output is missing the manifest-declared raw output name"
                )
            boxes, scores, classes = _decode_raw_output(
                values[transform.raw_output_name],
                transform.box_format,
                transform.raw_has_objectness,
            )
        else:
            raise DetectionValidationError("manifest does not declare a usable output contract")

        if len(boxes) != len(scores) or len(boxes) != len(classes):
            raise DetectionValidationError(
                "backend output arrays contain different candidate counts"
            )

        candidates: list[Detection] = []
        for _index, (box, score, class_id) in enumerate(zip(boxes, scores, classes, strict=True)):
            if not _is_finite_box(box) or not np.isfinite(score):
                rejected += 1
                continue
            if not 0 <= float(score) <= 1:
                rejected += 1
                continue
            if not isinstance(class_id, (int, np.integer)) or int(class_id) < 0:
                rejected += 1
                continue
            class_number = int(class_id)
            if class_number >= len(transform.labels):
                rejected += 1
                continue
            source_box = transform.model_to_source_box(box)
            clipped = _clip_box(
                source_box,
                width=transform.source_width,
                height=transform.source_height,
            )
            if clipped is None:
                rejected += 1
                continue
            if float(score) < transform.confidence_threshold:
                rejected += 1
                continue
            candidates.append(
                Detection(
                    box_xyxy=clipped,
                    class_id=class_number,
                    label=transform.labels[class_number],
                    confidence=float(score),
                    model_id=transform.model_id,
                    model_sha256=transform.model_sha256,
                    frame_sequence=transform.frame_sequence,
                    detected_at=transform.captured_at,
                )
            )

        retained, suppressed = _classwise_nms(candidates, transform.iou_threshold)
        return DetectionBatch(
            detections=tuple(retained),
            rejected_count=rejected + suppressed,
        )


def _normalise_boxes(value: Any, box_format: str) -> list[tuple[float, float, float, float]]:
    array = _squeeze_batch(np.asarray(value))
    if array.size == 0:
        return []
    if array.ndim == 1:
        if array.size != 4:
            raise DetectionValidationError("boxes output must contain four coordinates per box")
        array = array.reshape(1, 4)
    elif array.ndim == 2:
        if array.shape[1] == 4:
            pass
        elif array.shape[0] == 4:
            array = array.T
        else:
            raise DetectionValidationError("boxes output must have a coordinate axis of length 4")
    else:
        array = _flatten_coordinate_array(array)
    if array.shape[-1] != 4:
        raise DetectionValidationError("boxes output must have a coordinate axis of length 4")
    rows = [(float(row[0]), float(row[1]), float(row[2]), float(row[3])) for row in array]
    if box_format == "xywh":
        return [_xywh_to_xyxy(row) for row in rows]
    if box_format != "xyxy":
        raise DetectionValidationError(f"unsupported box format: {box_format}")
    return rows


def _normalise_scores(value: Any, candidate_count: int) -> tuple[list[float], list[int]]:
    array = _squeeze_batch(np.asarray(value))
    if array.size == 0 and candidate_count == 0:
        return [], []
    if array.ndim == 0:
        if candidate_count != 1:
            raise DetectionValidationError("scores output has an invalid shape")
        array = array.reshape(1, 1)
    elif array.ndim == 1:
        if array.shape[0] != candidate_count:
            raise DetectionValidationError("scores output count does not match boxes")
        array = array.reshape(candidate_count, 1)
    elif array.ndim == 2:
        if array.shape[0] == candidate_count:
            pass
        elif array.shape[1] == candidate_count:
            array = array.T
        else:
            raise DetectionValidationError("scores output count does not match boxes")
    else:
        raise DetectionValidationError("scores output must be one or two dimensional")

    if array.shape[0] != candidate_count:
        raise DetectionValidationError("scores output count does not match boxes")
    scores: list[float] = []
    classes: list[int] = []
    for row in array:
        if row.shape[0] == 0:
            raise DetectionValidationError("scores output contains an empty class axis")
        if row.shape[0] == 1:
            scores.append(float(row[0]))
            classes.append(0)
            continue
        if not np.isfinite(row).all():
            scores.append(float("nan"))
            classes.append(-1)
            continue
        class_id = int(np.argmax(row))
        scores.append(float(row[class_id]))
        classes.append(class_id)
    return scores, classes


def _normalise_classes(value: Any, candidate_count: int) -> list[int]:
    array = _squeeze_batch(np.asarray(value))
    if array.size != candidate_count:
        raise DetectionValidationError("classes output count does not match boxes")
    classes: list[int] = []
    for item in array.reshape(-1):
        try:
            number = float(item)
        except (TypeError, ValueError) as error:
            raise DetectionValidationError("classes output must contain numeric values") from error
        if not np.isfinite(number) or not number.is_integer():
            classes.append(-1)
        else:
            classes.append(int(number))
    return classes


def _decode_raw_output(
    value: Any,
    box_format: str,
    raw_has_objectness: bool,
) -> tuple[list[tuple[float, float, float, float]], list[float], list[int]]:
    array = _squeeze_batch(np.asarray(value))
    if array.size == 0:
        return [], [], []
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise DetectionValidationError("raw output must be one or two dimensional after batching")
    if array.shape[1] < 5 or (array.shape[0] <= 128 and array.shape[1] > array.shape[0]):
        array = array.T
    if array.shape[1] < 5:
        raise DetectionValidationError("raw output must contain at least five values per candidate")

    boxes = _normalise_boxes(array[:, :4], box_format)
    if not raw_has_objectness:
        class_scores = array[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        scores = [float(class_scores[index, class_id]) for index, class_id in enumerate(class_ids)]
        classes = [int(value) for value in class_ids]
    else:
        if array.shape[1] < 6:
            raise DetectionValidationError(
                "raw output with objectness must contain an objectness and class score"
            )
        objectness = array[:, 4]
        class_scores = array[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        scores = [
            float(objectness[index] * class_scores[index, class_id])
            for index, class_id in enumerate(class_ids)
        ]
        classes = [int(value) for value in class_ids]
    return boxes, scores, classes


def _squeeze_batch(array: np.ndarray) -> np.ndarray:
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    return array


def _flatten_coordinate_array(array: np.ndarray) -> np.ndarray:
    coordinate_axes = [axis for axis, size in enumerate(array.shape) if size == 4]
    if not coordinate_axes:
        raise DetectionValidationError("boxes output must have a coordinate axis of length 4")
    coordinate_axis = coordinate_axes[-1]
    moved = np.moveaxis(array, coordinate_axis, -1)
    return moved.reshape(-1, 4)


def _xywh_to_xyxy(box: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, width, height = (float(value) for value in box)
    return (
        x - width / 2,
        y - height / 2,
        x + width / 2,
        y + height / 2,
    )


def _is_finite_box(box: Sequence[float]) -> bool:
    if len(box) != 4:
        return False
    values = tuple(float(value) for value in box)
    return bool(np.isfinite(values).all() and values[0] < values[2] and values[1] < values[3])


def _clip_box(
    box: Sequence[float],
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    x1, y1, x2, y2 = (float(value) for value in box)
    clipped = (
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    )
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        return None
    return clipped


def _classwise_nms(
    detections: Sequence[Detection],
    iou_threshold: float,
) -> tuple[list[Detection], int]:
    kept: list[Detection] = []
    suppressed = 0
    for class_id in sorted({detection.class_id for detection in detections}):
        class_detections = [
            (index, detection)
            for index, detection in enumerate(detections)
            if detection.class_id == class_id
        ]
        class_detections.sort(key=lambda item: (-item[1].confidence, item[0]))
        class_kept: list[Detection] = []
        for _index, candidate in class_detections:
            if any(
                _iou(candidate.box_xyxy, prior.box_xyxy) > iou_threshold for prior in class_kept
            ):
                suppressed += 1
                continue
            class_kept.append(candidate)
        kept.extend(class_kept)
    kept.sort(key=lambda detection: (-detection.confidence, detection.class_id, detection.box_xyxy))
    return kept, suppressed


def _iou(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    first_x1, first_y1, first_x2, first_y2 = first
    second_x1, second_y1, second_x2, second_y2 = second
    intersection_x1 = max(first_x1, second_x1)
    intersection_y1 = max(first_y1, second_y1)
    intersection_x2 = min(first_x2, second_x2)
    intersection_y2 = min(first_y2, second_y2)
    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first_x2 - first_x1) * max(0.0, first_y2 - first_y1)
    second_area = max(0.0, second_x2 - second_x1) * max(0.0, second_y2 - second_y1)
    union = first_area + second_area - intersection
    return 0.0 if union <= 0 else intersection / union


def adapter_for_manifest(manifest: ModelManifest) -> DetectionAdapter:
    """Return the only adapter allowed by the manifest schema."""
    if manifest.adapter == "ultralytics_yolo_nms":
        return UltralyticsYoloNmsAdapter()
    raise DetectionValidationError(f"unsupported model adapter: {manifest.adapter}")
