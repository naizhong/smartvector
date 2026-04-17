"""Tests for the ConfidenceEngine."""

from datetime import datetime, timedelta

from smartvector.confidence import ConfidenceEngine
from smartvector.models import SmartVector


class TestConfidenceEngine:
    def setup_method(self):
        self.engine = ConfidenceEngine(half_life_days=30.0)

    def test_no_decay_at_time_zero(self):
        result = self.engine.compute_decay(0.85, age_days=0.0)
        assert result == 0.85

    def test_half_life_decay(self):
        result = self.engine.compute_decay(1.0, age_days=30.0)
        assert abs(result - 0.5) < 0.01

    def test_double_half_life(self):
        result = self.engine.compute_decay(1.0, age_days=60.0)
        assert abs(result - 0.25) < 0.01

    def test_min_confidence_floor(self):
        result = self.engine.compute_decay(0.5, age_days=10000)
        assert result >= self.engine.min_confidence

    def test_positive_feedback_boosts(self):
        result = self.engine.apply_feedback(0.5, positive_count=5, negative_count=0)
        assert result > 0.5

    def test_negative_feedback_penalizes(self):
        result = self.engine.apply_feedback(0.5, positive_count=0, negative_count=3)
        assert result < 0.5

    def test_feedback_capped_at_one(self):
        result = self.engine.apply_feedback(0.95, positive_count=100, negative_count=0)
        assert result <= 1.0

    def test_feedback_floored(self):
        result = self.engine.apply_feedback(0.1, positive_count=0, negative_count=100)
        assert result >= self.engine.min_confidence

    def test_access_reinforcement(self):
        result = self.engine.apply_access_reinforcement(0.5, access_count=10)
        assert result > 0.5

    def test_no_reinforcement_for_zero_access(self):
        result = self.engine.apply_access_reinforcement(0.5, access_count=0)
        assert result == 0.5

    def test_compute_current_confidence_fresh(self):
        v = SmartVector(
            content="test",
            base_confidence=0.85,
            created_at=datetime.now(),
        )
        conf = self.engine.compute_current_confidence(v)
        assert 0.84 <= conf <= 0.86

    def test_compute_current_confidence_old(self):
        v = SmartVector(
            content="test",
            base_confidence=0.85,
            created_at=datetime.now() - timedelta(days=60),
        )
        conf = self.engine.compute_current_confidence(v)
        assert conf < 0.5  # Significantly decayed

    def test_compute_with_reference_time(self):
        now = datetime.now()
        v = SmartVector(content="test", base_confidence=0.85, created_at=now)
        future = now + timedelta(days=30)
        conf = self.engine.compute_current_confidence(v, reference_time=future)
        assert abs(conf - 0.425) < 0.05

    def test_should_go_dormant(self):
        assert self.engine.should_go_dormant(0.10) is True
        assert self.engine.should_go_dormant(0.20) is False
        assert self.engine.should_go_dormant(0.14) is True
