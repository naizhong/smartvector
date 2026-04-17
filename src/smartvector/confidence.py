"""
Confidence Engine — manages the trust lifecycle of a SmartVector.

Implements:
- Exponential decay based on age (Ebbinghaus forgetting curve)
- Positive / negative feedback adjustment (reconsolidation)
- Access-frequency reinforcement (spaced repetition)
- Authority-weighted base confidence

Inspired by:
- Neuroscience: Memory strength = f(recency, frequency, emotional weight)
- ATiSE: Gaussian distributions widening over time
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from smartvector.models import SmartVector


class ConfidenceEngine:
    """Computes and manages confidence scores for SmartVectors.

    Parameters
    ----------
    half_life_days : float
        Number of days for confidence to halve (default 30).
    feedback_positive_boost : float
        Confidence added per positive feedback event.
    feedback_negative_penalty : float
        Confidence removed per negative feedback event.
    access_reinforcement : float
        Base reinforcement per access (diminishing returns via log).
    dormant_threshold : float
        Confidence below which a vector should go dormant.
    min_confidence : float
        Floor — confidence never drops below this.
    """

    def __init__(
        self,
        half_life_days: float = 30.0,
        feedback_positive_boost: float = 0.03,
        feedback_negative_penalty: float = 0.08,
        access_reinforcement: float = 0.01,
        dormant_threshold: float = 0.15,
        min_confidence: float = 0.01,
    ) -> None:
        self.half_life_days = half_life_days
        self.feedback_positive_boost = feedback_positive_boost
        self.feedback_negative_penalty = feedback_negative_penalty
        self.access_reinforcement = access_reinforcement
        self.dormant_threshold = dormant_threshold
        self.min_confidence = min_confidence

    def compute_decay(self, base_confidence: float, age_days: float) -> float:
        """Exponential decay: ``C(t) = C_0 * 2^(-t / half_life)``.

        This mirrors the Ebbinghaus forgetting curve from psychology.
        """
        decayed = base_confidence * (2 ** (-age_days / self.half_life_days))
        return max(self.min_confidence, decayed)

    def apply_feedback(
        self,
        current_confidence: float,
        positive_count: int,
        negative_count: int,
    ) -> float:
        """Adjust confidence based on cumulative user feedback.

        Every positive feedback (user accepted the answer) strengthens the
        memory; every negative feedback (user corrected) weakens it.
        """
        boost = positive_count * self.feedback_positive_boost
        penalty = negative_count * self.feedback_negative_penalty
        adjusted = current_confidence + boost - penalty
        return max(self.min_confidence, min(1.0, adjusted))

    def apply_access_reinforcement(
        self,
        current_confidence: float,
        access_count: int,
    ) -> float:
        """Frequently accessed vectors get a small confidence boost.

        Uses diminishing returns: ``log(access_count + 1) * reinforcement``.
        Analogy: memories you recall often become stronger.
        """
        if access_count <= 0:
            return current_confidence
        reinforcement = math.log(access_count + 1) * self.access_reinforcement
        return min(1.0, current_confidence + reinforcement)

    def compute_current_confidence(
        self,
        vector: SmartVector,
        reference_time: datetime | None = None,
    ) -> float:
        """Full confidence computation combining all factors.

        Pipeline:
        1. Start with ``base_confidence`` (from source authority)
        2. Apply time decay
        3. Apply feedback adjustment
        4. Apply access reinforcement
        """
        if reference_time is None:
            reference_time = datetime.now()

        age_days = (reference_time - vector.created_at).total_seconds() / 86400

        decayed = self.compute_decay(vector.base_confidence, age_days)
        adjusted = self.apply_feedback(
            decayed, vector.positive_feedback, vector.negative_feedback
        )
        reinforced = self.apply_access_reinforcement(adjusted, vector.access_count)
        return round(reinforced, 4)

    def should_go_dormant(self, confidence: float) -> bool:
        """Return ``True`` if the vector should transition to dormant."""
        return confidence < self.dormant_threshold
