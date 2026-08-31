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
            boxes = _normalise_boxes(values[boxes_name])
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
                transform.raw_layout,
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
            if not _has_finite_values(box) or not np.isfinite(score):
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
            try:
                model_box = _to_model_xyxy(box, transform)
            except (TypeError, ValueError):
                rejected += 1
                continue
            if not _is_finite_box(model_box):
                rejected += 1
                continue
            source_box = transform.model_to_source_box(model_box)
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


def _normalise_boxes(value: Any) -> list[tuple[float, float, float, float]]:
    array = _output_array(value, "boxes")
    if array.size == 0:
        return []
    if array.ndim == 1:
        if array.size != 4:
            raise DetectionValidationError("boxes output must contain four coordinates per box")
        array = array.reshape(1, 4)
    elif array.ndim == 2:
        if array.shape[1] != 4:
            raise DetectionValidationError("boxes output must have a coordinate axis of length 4")
    elif array.ndim == 3 and array.shape[0] == 1 and array.shape[2] == 4:
        array = array[0]
    else:
        raise DetectionValidationError(
            "named boxes output must be [N,4] or [1,N,4] with coordinate axis last"
        )
    if array.shape[-1] != 4:
        raise DetectionValidationError("boxes output must have a coordinate axis of length 4")
    try:
        rows = [(float(row[0]), float(row[1]), float(row[2]), float(row[3])) for row in array]
    except (TypeError, ValueError) as error:
        raise DetectionValidationError("boxes output must contain numeric values") from error
    return rows


def _normalise_scores(value: Any, candidate_count: int) -> tuple[list[float], list[int]]:
    array = _output_array(value, "scores")
    if not np.issubdtype(array.dtype, np.number):
        raise DetectionValidationError("scores output must contain numeric values")
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
        else:
            raise DetectionValidationError("scores output count does not match boxes")
    elif array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
        if array.shape[0] != candidate_count:
            raise DetectionValidationError("scores output count does not match boxes")
    else:
        raise DetectionValidationError("scores output must be [N,C] or [1,N,C]")

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
    array = _output_array(value, "classes")
    if array.ndim == 3 and array.shape[0] == 1 and array.shape[2] == 1:
        array = array[0, :, 0]
    elif array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1:
        raise DetectionValidationError("classes output must be [N] or [1,N]")
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
    raw_layout: str | None,
    raw_has_objectness: bool | None,
) -> tuple[list[tuple[float, float, float, float]], list[float], list[int]]:
    array = _output_array(value, "raw")
    if not np.issubdtype(array.dtype, np.number):
        raise DetectionValidationError("raw output must contain numeric values")
    if array.size == 0:
        return [], [], []
    if raw_layout not in {"candidates_first", "channels_first", "channels_last"}:
        raise DetectionValidationError("raw output requires an explicit layout")
    if not isinstance(raw_has_objectness, bool):
        raise DetectionValidationError("raw output requires explicit objectness metadata")
    if raw_layout == "candidates_first":
        if array.ndim != 2:
            raise DetectionValidationError("candidates_first raw output must be [N,A]")
        rows = array
    elif raw_layout == "channels_first":
        if array.ndim != 3 or array.shape[0] != 1:
            raise DetectionValidationError("channels_first raw output must be [1,A,N]")
        rows = array[0].T
    else:
        if array.ndim != 3 or array.shape[0] != 1:
            raise DetectionValidationError("channels_last raw output must be [1,N,A]")
        rows = array[0]
    if rows.shape[1] < (6 if raw_has_objectness else 5):
        raise DetectionValidationError("raw output has too few values per candidate")

    boxes = _normalise_boxes(rows[:, :4])
    if not raw_has_objectness:
        class_scores = rows[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        scores = [float(class_scores[index, class_id]) for index, class_id in enumerate(class_ids)]
        classes = [int(value) for value in class_ids]
    else:
        objectness = rows[:, 4]
        class_scores = rows[:, 5:]
        class_ids = np.argmax(class_scores, axis=1)
        scores = [
            float(objectness[index] * class_scores[index, class_id])
            for index, class_id in enumerate(class_ids)
        ]
        classes = [int(value) for value in class_ids]
    return boxes, scores, classes


def _output_array(value: Any, label: str) -> np.ndarray:
    try:
        return np.asarray(value)
    except (TypeError, ValueError) as error:
        raise DetectionValidationError(f"{label} output is not an array") from error


def _to_model_xyxy(
    box: Sequence[float],
    transform: ImageTransform,
) -> tuple[float, float, float, float]:
    """Convert one explicitly declared box geometry to model-space xyxy."""
    x1, y1, x2, y2 = (float(value) for value in box)
    if transform.coordinate_space == "normalized":
        x1, y1, x2, y2 = (
            x1 * transform.model_width,
            y1 * transform.model_height,
            x2 * transform.model_width,
            y2 * transform.model_height,
        )
    if transform.box_format == "xywh":
        return _xywh_to_xyxy((x1, y1, x2, y2))
    if transform.box_format == "xyxy":
        return x1, y1, x2, y2
    raise DetectionValidationError(f"unsupported box format: {transform.box_format}")


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


def _has_finite_values(values: Sequence[float]) -> bool:
    if len(values) != 4:
        return False
    try:
        return bool(np.isfinite(tuple(float(value) for value in values)).all())
    except (TypeError, ValueError):
        return False


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
