from semi_seg._utils import FeatureExtractor


class unl_extractor:
    def __init__(self, features: FeatureExtractor, n_uls: int) -> None:
        super().__init__()
        self._features = features
        self._n_uls = n_uls

    def __iter__(self):
        for feature in self._features:
            assert len(feature) > self._n_uls, (len(feature), self._n_uls)
            yield feature[len(feature) - self._n_uls:]
