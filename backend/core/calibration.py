"""
AttoSense v4 — Confidence Calibration
=====================================
Raw LLM confidence scores are not well-calibrated: a model that says 0.85
may only be correct ~70% of the time in practice. This module:

  1. Stores (raw_confidence, was_correct, modality, intent) samples
     every time a reviewer approves/rejects an inbox item.
  2. Fits a per-modality isotonic regression calibration map from those samples.
  3. Applies the calibration map to raw scores before they are returned to the API.

Before enough data exists (< MIN_SAMPLES per modality), the raw score is
returned unchanged — calibration only activates once it has enough signal.

Per-modality inbox thresholds
------------------------------
Audio and vision are intrinsically noisier than text. A vision result at 0.71
is far less trustworthy than a text result at 0.71. We apply separate thresholds:

  text   ≥ 0.72 → pass  (was 0.70)
  audio  ≥ 0.75 → pass
  vision ≥ 0.78 → pass
"""

from __future__ import annotations

import math
from typing import Optional

from backend.core.logging_config import get_logger

log = get_logger("attosense.calibration")

# ── Per-modality inbox thresholds ──────────────────────────────────────────────

MODALITY_THRESHOLDS: dict[str, float] = {
    "text":   0.72,
    "audio":  0.75,
    "vision": 0.78,
}
DEFAULT_THRESHOLD = 0.72   # fallback if modality unknown


def get_threshold(modality: str) -> float:
    """Return the inbox confidence threshold for this modality."""
    return MODALITY_THRESHOLDS.get(modality, DEFAULT_THRESHOLD)


def is_low_confidence(confidence: float, modality: str) -> bool:
    """True when the calibrated confidence is below the modality threshold."""
    return confidence < get_threshold(modality)


# ── Isotonic calibration map ───────────────────────────────────────────────────

MIN_SAMPLES = 20   # minimum corrections before calibration activates

class _IsotonicCalibrator:
    """
    Simple pool-adjacent-violators isotonic regression.
    Fits a monotone non-decreasing map from raw → calibrated confidence.

    We maintain per-modality calibrators so that audio/vision noise
    doesn't pollute the text calibration curve and vice versa.
    """

    def __init__(self) -> None:
        self._raw:     list[float] = []
        self._correct: list[float] = []   # 1.0 if correct, 0.0 if not
        self._fitted:  bool        = False
        self._breakpoints: list[tuple[float, float]] = []   # (raw, calibrated)

    # ── Data ingestion ─────────────────────────────────────────────────────────

    def record(self, raw_confidence: float, was_correct: bool) -> None:
        """Add one calibration sample. Call after every reviewer decision."""
        self._raw.append(max(0.0, min(1.0, raw_confidence)))
        self._correct.append(1.0 if was_correct else 0.0)
        self._fitted = False   # mark stale

    @property
    def n_samples(self) -> int:
        return len(self._raw)

    # ── Fitting ────────────────────────────────────────────────────────────────

    def fit(self) -> None:
        """Fit the isotonic regression. O(n log n) — fast enough for <10k samples."""
        if self.n_samples < MIN_SAMPLES:
            self._fitted = False
            return

        # Sort by raw score
        pairs = sorted(zip(self._raw, self._correct))
        raw_s   = [p[0] for p in pairs]
        corr_s  = [p[1] for p in pairs]

        # Pool-adjacent-violators (PAV) algorithm
        # Result: monotone non-decreasing sequence of calibrated values
        blocks: list[list[float]] = [[c] for c in corr_s]
        while True:
            merged = False
            i = 0
            new_blocks: list[list[float]] = []
            while i < len(blocks):
                if (i + 1 < len(blocks) and
                        _mean(blocks[i]) > _mean(blocks[i + 1])):
                    new_blocks.append(blocks[i] + blocks[i + 1])
                    i += 2
                    merged = True
                else:
                    new_blocks.append(blocks[i])
                    i += 1
            blocks = new_blocks
            if not merged:
                break

        # Expand blocks back to per-sample calibrated values
        calibrated: list[float] = []
        for block in blocks:
            m = _mean(block)
            calibrated.extend([m] * len(block))

        # Build a lookup table of (raw_score, calibrated_score) breakpoints
        self._breakpoints = list(zip(raw_s, calibrated))
        self._fitted       = True
        log.info("calibration_fitted", extra={"n_samples": self.n_samples})

    # ── Inference ──────────────────────────────────────────────────────────────

    def calibrate(self, raw: float) -> float:
        """Map a raw score to a calibrated score using linear interpolation."""
        if not self._fitted or not self._breakpoints:
            return raw   # not enough data yet — return raw unchanged

        raw = max(0.0, min(1.0, raw))

        # Exact match
        for r, c in self._breakpoints:
            if math.isclose(r, raw, abs_tol=1e-6):
                return c

        # Below lowest breakpoint
        if raw <= self._breakpoints[0][0]:
            return self._breakpoints[0][1]
        # Above highest breakpoint
        if raw >= self._breakpoints[-1][0]:
            return self._breakpoints[-1][1]

        # Linear interpolation between surrounding breakpoints
        for i in range(len(self._breakpoints) - 1):
            r0, c0 = self._breakpoints[i]
            r1, c1 = self._breakpoints[i + 1]
            if r0 <= raw <= r1:
                t = (raw - r0) / (r1 - r0) if r1 > r0 else 0.5
                return c0 + t * (c1 - c0)

        return raw   # fallback


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# ── Module-level per-modality calibrators ─────────────────────────────────────

_calibrators: dict[str, _IsotonicCalibrator] = {
    "text":   _IsotonicCalibrator(),
    "audio":  _IsotonicCalibrator(),
    "vision": _IsotonicCalibrator(),
}


def record_correction(
    raw_confidence: float,
    was_correct: bool,
    modality: str,
) -> None:
    """
    Called by api.py every time a reviewer approves or rejects an inbox item.
    was_correct = True  if reviewer approved without changing the label
                        (model was right)
    was_correct = False if reviewer changed the label or rejected
                        (model was wrong)
    """
    cal = _calibrators.get(modality)
    if cal is None:
        return
    cal.record(raw_confidence, was_correct)
    # Refit after every 5 new samples (cheap enough at typical inbox volumes)
    if cal.n_samples % 5 == 0:
        cal.fit()
    log.debug("calibration_sample_recorded", extra={
        "modality": modality, "raw": raw_confidence,
        "correct": was_correct, "n": cal.n_samples,
    })


def calibrate_confidence(
    raw_confidence: float,
    modality: str,
) -> float:
    """
    Apply the fitted calibration map to a raw LLM confidence score.
    Returns raw score unchanged until MIN_SAMPLES corrections have been recorded.
    """
    cal = _calibrators.get(modality)
    if cal is None:
        return raw_confidence
    calibrated = cal.calibrate(raw_confidence)
    if calibrated != raw_confidence:
        log.debug("calibration_applied", extra={
            "modality": modality, "raw": raw_confidence, "calibrated": calibrated,
        })
    return calibrated


def calibration_status() -> dict:
    """Return the current state of each modality's calibrator (for /health)."""
    return {
        modality: {
            "n_samples": cal.n_samples,
            "fitted":    cal._fitted,
            "active":    cal.n_samples >= MIN_SAMPLES,
            "threshold": get_threshold(modality),
        }
        for modality, cal in _calibrators.items()
    }
