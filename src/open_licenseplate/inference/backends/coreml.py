"""Compatibility exports for the macOS Core ML backend."""

from ..coreml import CoreMLBackend, compare_manifest_to_inspection, coreml_compute_unit

__all__ = ["CoreMLBackend", "compare_manifest_to_inspection", "coreml_compute_unit"]
