import math
import re
from typing import Union, List, Callable, Tuple, Any, Iterable
import yfinance
import gym
from pandas_ml_common import pd, np, Typing
from pandas_ml_utils import FeaturesAndLabels
from pandas_ml_utils.ml.data.extraction import extract_features


class RandomAssetEnv(gym.Env):

    # TODO der reward ist ein callable (state, action, label)
    #
    # die actions muss man als gym space definieren. bei take action rufen wir die reward Funktion auf und gehen einen Schritt weiter im Feature und label array.
    #
    # wenn man state braucht z.b für Kauf Verkauf usw. muss das der action handler können. um die Position bewerten zu können muss dann eine explizite hold Aktion ausgeführt werden.
    #
    # das env muss ein train / test / predict modus haben entsprechend werden die Daten und Parameter im reset() gesetzt. der Agent muss dann entsprechend ausserhalb der train loop laufen. 2 Varianten sollten möglich sein:  führe actions zufällig gemäss prob distribution aus oder führe immer besste action aus.

    def __init__(self,
                 features_and_labels: FeaturesAndLabels,
                 symbols: Union[str, List[str]],
                 action_space: gym.Space,
                 data_provider: Callable[[str], Typing.PatchedDataFrame] = lambda symbol: yfinance.download(symbol),
                 reward_provider: Callable[[Any, np.ndarray, np.ndarray, np.ndarray], Tuple[float, bool]] = None,
                 pct_train_data: float = 0.8,
                 max_steps: int = None,
                 use_file_cache: str = None):
        super().__init__()
        self.max_steps = math.inf if max_steps is None else max_steps
        self.features_and_labels = features_and_labels
        self.reward_provider = reward_provider
        self.pct_train_data = pct_train_data
        self.data_provider = data_provider

        # define spaces
        self.action_space = action_space
        self.observation_space = None

        # define execution mode
        self.file_cache = pd.HDFStore(use_file_cache, mode='a') if use_file_cache is not None else None
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
        self._df = self._from_cache_or_create(self._symbol, lambda: self.data_provider(self._symbol))

        if self.mode in ['train', 'test']:
            self._features, self._labels, self._targets, self._sample_weights, self._gross_loss = \
                [*self._from_cache_or_create(
                    [f'{self._symbol}__features', f'{self._symbol}__labels', f'{self._symbol}__targets', f'{self._symbol}__sample_weights', f'{self._symbol}__gross_loss'],
                    lambda: [f[0] if isinstance(f, Tuple) else f for f in self._df._.extract(self.features_and_labels)]
                 ),
                 *[None] * 5
                ][:5]

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

    def _from_cache_or_create(self, keys, creator):
        if self.file_cache is None:
            return creator()

        # make keys a list
        if isinstance(keys, str): keys = [keys]

        frames = []
        for key in keys:
            if key in self.file_cache:
                frames.append(self.file_cache[key])

        if len(frames) <= 0:
            frames = creator()
            if isinstance(frames, pd.DataFrame): frames = [frames]
            for i in range(len(frames)):
                if frames[i] is not None:
                    self.file_cache[keys[i]] = frames[i]

        return frames[0] if len(frames) == 1 else frames

    def _current_state(self):
        # make sure to return it as a batch of size one therefore use list of state_idx
        return self._features.iloc[[self._state_idx]]

# ...
# TODO later allow features to be a list of feature sets echa witha possible different shape
