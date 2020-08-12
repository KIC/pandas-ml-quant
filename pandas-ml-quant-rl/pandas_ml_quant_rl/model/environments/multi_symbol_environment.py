import math
import re
from typing import Union, List, Callable, Tuple, Any, Iterable
import yfinance
import gym
from pandas_ml_common import pd, np, Typing
from pandas_ml_utils import FeaturesAndLabels
from pandas_ml_utils.ml.data.extraction import extract_features


class Cache(object):

    def __init__(self, data_provider: Callable[[str], Typing.PatchedDataFrame] = None):
        self.data_provider = pd.fetch_yahoo if data_provider is None else data_provider

    def get_data_or_fetch(self, symbol) -> pd.DataFrame:
        pass

    def get_feature_frames_or_fetch(self, df, symbol, features_and_labels) -> Tuple[pd.DataFrame, ...]:
        pass


class NoCache(Cache):

    def __init__(self, data_provider: Callable[[str], Typing.PatchedDataFrame] = None):
        super().__init__(data_provider)

    def get_data_or_fetch(self, symbol):
        return self.data_provider(symbol)

    def get_feature_frames_or_fetch(self, df, symbol, features_and_labels):
        (features, _), labels, targets, weights, loss = df._.extract(features_and_labels)
        return features, labels, targets, weights, loss


class FileCache(Cache):

    def __init__(self, file_name: str, data_provider: Callable[[str], Typing.PatchedDataFrame] = None):
        super().__init__(data_provider)
        self.file_cache = pd.HDFStore(file_name, mode='a')

    def get_data_or_fetch(self, symbol):
        if symbol in self.file_cache:
            return self.file_cache[symbol]
        else:
            df = self.data_provider(symbol)
            self.file_cache[symbol] = df
            return df

    def get_feature_frames_or_fetch(self, df, symbol, features_and_labels):
        fkey = f'{symbol}__features'
        lkey = f'{symbol}__labels'
        tkey = f'{symbol}__targets'
        swkey = f'{symbol}__sample_weights'
        glkey = f'{symbol}__gross_loss'

        if fkey in self.file_cache:
            features = self.file_cache[fkey]
            labels = self.file_cache[lkey]
            targets = self.file_cache[tkey] if tkey in self.file_cache else None
            weights = self.file_cache[swkey] if swkey in self.file_cache else None
            loss = self.file_cache[glkey] if glkey in self.file_cache else None
        else:
            (features, _), labels, targets, weights, loss = df._.extract(features_and_labels)
            self.file_cache[fkey] = features
            self.file_cache[lkey] = labels
            if targets is not None: self.file_cache[tkey] = targets
            if weights is not None: self.file_cache[swkey] = weights
            if loss is not None: self.file_cache[glkey] = loss

        return features, labels, targets, weights, loss


class RandomAssetEnv(gym.Env):

    def __init__(self,
                 features_and_labels: FeaturesAndLabels,
                 symbols: Union[str, List[str]],
                 action_space: gym.Space,
                 reward_provider: Callable[[Any, np.ndarray, np.ndarray, np.ndarray], Tuple[float, bool]] = None,
                 pct_train_data: float = 0.8,
                 max_steps: int = None,
                 use_cache: Cache = NoCache(lambda symbol: yfinance.download(symbol))):
        super().__init__()
        self.max_steps = math.inf if max_steps is None else max_steps
        self.features_and_labels = features_and_labels
        self.reward_provider = reward_provider
        self.pct_train_data = pct_train_data

        # define spaces
        self.action_space = action_space
        self.observation_space = None

        # define execution mode
        self.cache = use_cache
        self.mode = 'train'
        self.done = True

        # load symbols available to randomly draw from
        if isinstance(symbols, str):
            with open(symbols) as f:
                self.symbols = np.array(re.split("[\r\n]|[\n]|[\r]", f.read()))
        else:
            self.symbols = symbols

        # finally make a dummy initialisation
        self._init()

    def as_train(self):
        self.mode = 'train'
        return self

    def as_test(self):
        self.mode = 'test'
        return self

    def as_predict(self):
        self.mode = 'predict'
        return self

    def step(self, action):
        reward, game_over = self.reward_provider(
            action,
            self._labels.iloc[[self._state_idx]]._.values,
            self._sample_weights.iloc[[self._state_idx]]._.values if self._sample_weights is not None else None,
            self._gross_loss.iloc[[self._state_idx]]._.values if self._gross_loss is not None else None
        )

        self._state_idx += 1
        self.done = game_over \
                    or self._state_idx >= self._last_index \
                    or self._state_idx > (self._start_idx + self.max_steps)

        return self._current_state(), reward, self.done, {}

    def render(self, mode='human'):
        # eventually implement me
        pass

    def reset(self):
        return self._init()

    def _init(self):
        self._symbol = np.random.choice(self.symbols, 1).item()
        self._df = self.cache.get_data_or_fetch(self._symbol)

        if self.mode in ['train', 'test']:
            self._features, self._labels, self._targets, self._sample_weights, self._gross_loss = \
                self.cache.get_feature_frames_or_fetch(self._df, self._symbol, self.features_and_labels)

            if self.mode == 'train':
                self._last_index = int(len(self._features) * 0.8)
                # allow at least one step
                self._state_idx = np.random.randint(1, self._last_index - 1, 1).item()
            if self.mode == 'test':
                self._last_index = len(self._features)
                # allow at least one step
                self._state_idx = np.random.randint(int(len(self._features) * 0.8), len(self._features) - 1, 1).item()

        else:
            # only extract features!
            _, self._features, self._targets = extract_features(self._df, self.features_and_labels)
            self._last_index = len(self._features)
            self._labels = pd.DataFrame({})
            self._sample_weights = pd.DataFrame({})
            self._gross_loss = pd.DataFrame({})
            self._state_idx = 0

        if self.observation_space is None:
            self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=self._features.iloc[[-1]]._.values.shape[1:])

        self._start_idx = self._state_idx
        self.done = self._state_idx >= self._last_index
        return self._current_state()

    def _current_state(self):
        # make sure to return it as a batch of size one therefore use list of state_idx
        return self._features.iloc[[self._state_idx]]

# ...
# TODO later allow features to be a list of feature sets echa witha possible different shape
