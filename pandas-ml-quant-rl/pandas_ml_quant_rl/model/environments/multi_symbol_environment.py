import math
import re
from typing import Union, List, Callable, Tuple, Any
import gym
from pandas_ml_common import pd, np
from pandas_ml_quant_rl.cache.abstract_cache import Cache
from pandas_ml_quant_rl.cache import NoCache
from pandas_ml_quant_rl.renderer.abstract_renderer import Renderer
from pandas_ml_utils import FeaturesAndLabels
from pandas_ml_utils.ml.data.extraction import extract_features
from ..strategies.abstract_startegy import Strategy


class RandomAssetEnv(gym.Env):

    def __init__(self,
                 features_and_labels: FeaturesAndLabels,
                 symbols: Union[str, List[str]],
                 strategy: Strategy,
                 pct_train_data: float = 0.8,
                 max_steps: int = None,
                 min_training_samples: float = np.inf,
                 use_cache: Cache = NoCache(),
                 renderer: Renderer = Renderer()
                 ):
        super().__init__()
        self.max_steps = math.inf if max_steps is None else max_steps
        self.min_training_samples = min_training_samples
        self.features_and_labels = features_and_labels
        self.pct_train_data = pct_train_data
        self.strategy = strategy
        self.renderer = renderer

        # define spaces
        self.observation_space = None
        self.action_space = strategy.action_space

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

    def as_train(self) -> Tuple['RandomAssetEnv', pd.Series] :
        # note that shallow copy is shuffling due to the _init call
        copy = self._shallow_copy()
        copy.mode = 'train'
        return copy, self._current_state()

    def as_test(self, renderer=None) -> Tuple['RandomAssetEnv', pd.Series] :
        # note that shallow copy is shuffling due to the _init call
        copy = self._shallow_copy()
        copy.mode = 'test'
        if renderer is not None: copy.renderer = renderer
        return copy, self._current_state()

    def as_predict(self, renderer=None) -> Tuple['RandomAssetEnv', pd.Series] :
        # note that shallow copy is shuffling due to the _init call
        copy = self._shallow_copy()
        copy.mode = 'predict'
        if renderer is not None: copy.renderer = renderer
        return copy, self._current_state()

    def _shallow_copy(self):
        # note that shallow copy is shuffling due to the _init call
        return RandomAssetEnv(
            self.features_and_labels,
            self.symbols,
            self.strategy,
            self.pct_train_data,
            self.max_steps,
            self.min_training_samples,
            self.cache,
            self.renderer
        )

    def sample_action(self, probs=None):
        # FIXME allow a way to query possible action i.e. if you already bought you can only sell or hold
        #  then sample as long as needed until we get a possible action
        action = np.random.choice(len(probs), p=probs) if probs is not None else self.strategy.action_space.sample()
        return action

    def step(self, action):
        reward, game_over = self.strategy.trade_reward(
            action,
            self._labels.iloc[self._state_idx],
            self._sample_weights.iloc[self._state_idx] if self._sample_weights is not None else None,
            self._gross_loss.iloc[self._state_idx] if self._gross_loss is not None else None
        )

        old_price_frame = self._current_price_frame()
        self._state_idx += 1
        self.done = game_over \
                    or self._state_idx >= self._last_index \
                    or self._state_idx > (self._start_idx + self.max_steps)

        # push rendering information
        new_price_frame = self._current_price_frame()
        self.renderer.plot(old_price_frame, action, new_price_frame, reward, self.done)

        # return new state reward and done flag
        return self._current_state(), reward, self.done, {}

    def render(self, mode='human'):
        self.renderer.render(mode)

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
                # if min samples is infinity then we start from the first index of the test data
                test_start_idx = int(len(self._features) * 0.8)
                test_end_idx = min(len(self._features) - self.min_training_samples, test_start_idx + 1)
                self._state_idx = np.random.randint(test_start_idx, test_end_idx, 1).item()
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

    def _current_price_frame(self):
        timestep = self._features.index[self._state_idx]
        return self._df.loc[[timestep]]


# ...
# TODO later allow features to be a list of feature sets echa witha possible different shape

# Also later we want to implement an envoronment where the agent has access to all assets and he needs to pick a combination
# of assets like in a portfolio construction manner
