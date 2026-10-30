"""Classical LFCC plus GMM baseline."""

from __future__ import annotations

import numpy as np
from sklearn.mixture import GaussianMixture


class LFCCGMM:
    """Fit genuine and synthetic GMMs and score synthetic-minus-genuine log likelihood."""

    def __init__(
        self, n_components: int = 8, covariance_type: str = "diag", random_state: int = 2025
    ) -> None:
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.genuine: GaussianMixture | None = None
        self.synthetic: GaussianMixture | None = None

    def fit(self, features: np.ndarray, labels: np.ndarray) -> LFCCGMM:
        """Fit one GMM per class on 2-D utterance-level features."""
        features, labels = np.asarray(features, dtype=np.float32), np.asarray(labels)
        if features.ndim != 2 or set(np.unique(labels)) != {0, 1}:
            raise ValueError("features must be 2-D and labels must contain both classes 0 and 1")
        kwargs = {
            "n_components": self.n_components,
            "covariance_type": self.covariance_type,
            "random_state": self.random_state,
        }
        self.genuine = GaussianMixture(**kwargs).fit(features[labels == 0])
        self.synthetic = GaussianMixture(**kwargs).fit(features[labels == 1])
        return self

    def score_samples(self, features: np.ndarray) -> np.ndarray:
        """Return log p(synthetic|x) - log p(genuine|x)."""
        if self.genuine is None or self.synthetic is None:
            raise RuntimeError("Call fit before score_samples")
        features = np.asarray(features, dtype=np.float32)
        return self.synthetic.score_samples(features) - self.genuine.score_samples(features)

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Classify positive likelihood-ratio scores as synthetic."""
        return (self.score_samples(features) >= 0).astype(np.int64)
