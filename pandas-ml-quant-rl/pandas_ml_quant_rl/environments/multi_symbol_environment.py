import math
import re
from typing import Union, List, Tuple, Dict

import gym

from pandas_ml_common import pd, np
from pandas_ml_quant_rl.cache import NoCache
from pandas_ml_quant_rl.cache.abstract_cache import Cache
from pandas_ml_utils import FeaturesAndLabels
from pandas_ml_quant_rl.model.environment import Environment, Strategy
from .utils import  SpaceUtils


class RandomAssetEnv(Environment):

    def __init__(self,
                 features_and_labels: FeaturesAndLabels,
                 symbols: Union[str, List[str]],
                 strategy: Strategy,
                 pct_train_data: float = 0.8,
                 max_steps: int = None,
                 min_training_samples: float = np.inf,
                 use_cache: Cache = NoCache(),
                 ):
        super().__init__(strategy)
        self.max_steps = math.inf if max_steps is None else max_steps
        self.min_training_samples = min_training_samples
        self.features_and_labels = features_and_labels
        self.pct_train_data = pct_train_data

        # define spaces
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

    def as_train(self) -> Tuple['RandomAssetEnv', Tuple[pd.DataFrame, np.ndarray]]:
        # note that shallow copy is shuffling due to the _init call
        copy = self._shallow_copy()
        copy.mode = 'train'
        return copy, self._current_state()

    def as_test(self) -> Tuple['RandomAssetEnv', Tuple[pd.DataFrame, np.ndarray]]:
        # note that shallow copy is shuffling due to the _init call
        copy = self._shallow_copy()
        copy.mode = 'test'
        return copy, self._current_state()

    def as_predict(self) -> Tuple['RandomAssetEnv', Tuple[pd.DataFrame, np.ndarray]]:
        # note that shallow copy is shuffling due to the _init call
        copy = self._shallow_copy()
        copy.mode = 'predict'
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
            self.cache
        )

    def sample_action(self, probs=None):
        return self.strategy.sample_action(probs)

    def step(self, action) -> Tuple[Tuple[np.ndarray, np.ndarray], float, bool, Dict]:
        old_price_frame = self._current_price_frame()

        # regardless what action we take we will end up in the next time step
        # and this is the first possible time where we actually can do something
        self._state_idx += 1
        new_price_frame = self._current_price_frame()

        # let the agent execute an action according to our strategy
        strategy_state, reward, game_over = self.strategy.trade_reward(
            old_price_frame,
            action,
            new_price_frame
        )

        # check if the epoch is over
        self.done = game_over \
                    or self._state_idx >= self._last_index \
                    or self._state_idx > (self._start_idx + self.max_steps)

        # return new state reward and done flag
        return self._current_state(strategy_state), reward, self.done, {}

    def render(self, mode='human'):
        return self.strategy.render(mode)

    def reset(self) -> Tuple[np.ndarray, np.ndarray]:
        self.strategy.reset()
        return self._init()

    def _init(self) -> Tuple[np.ndarray, np.ndarray]:
        # this is reinforcement learning, we can only extract features (no labels)!
        self._symbol = np.random.choice(self.symbols, 1).item()
        self._df = self.cache.get_data_or_fetch(self._symbol)
        self._features, self._features_index = self.cache.get_feature_frames_or_fetch(self._df, self._symbol, self.features_and_labels)
        self._is_feature_tuple = isinstance(self._features, tuple)
        nr_of_samples = len(self._features[0]) if self._is_feature_tuple else len(self._features)

        if self.mode in ['train', 'test']:
            if self.mode == 'train':
                self._last_index = int(nr_of_samples * 0.8)
                # allow at least one step
                self._state_idx = np.random.randint(1, self._last_index - 1, 1).item()
            if self.mode == 'test':
                self._last_index = nr_of_samples
                # if min samples is infinity then we start from the first index of the test data
                test_start_idx = int(nr_of_samples * 0.8)
                test_end_idx = min(nr_of_samples - self.min_training_samples, test_start_idx + 1)
                self._state_idx = np.random.randint(test_start_idx, test_end_idx, 1).item()
        else:
            self._last_index = nr_of_samples
            self._labels = pd.DataFrame({})
            self._sample_weights = pd.DataFrame({})
            self._gross_loss = pd.DataFrame({})
            self._state_idx = 0

        if self.observation_space is None:
            self.observation_space = gym.spaces.Tuple((
                SpaceUtils.unbounded_tuple_boxes(*[f.shape[1:] for f in self._features]) if self._is_feature_tuple else SpaceUtils.unbounded_box(self._features.shape[1:]),
                SpaceUtils.unbounded_box(self.strategy.current_state().shape)
            ))

        self._start_idx = self._state_idx
        self.done = self._state_idx >= self._last_index
        return self._current_state()

    def _current_state(self, strategy_state=None) -> Tuple[Union[np.ndarray, Tuple[np.ndarray]], np.ndarray]:
        # make sure to return it as a batch of size one therefore use list of state_idx
        return (
            tuple([f[[self._state_idx]] for f in self._features]) if self._is_feature_tuple else self._features[[self._state_idx]],
            strategy_state if strategy_state is not None else self.strategy.current_state()
        )

    def _current_price_frame(self) -> pd.Series:
        ts = self._features_index[self._state_idx]
        return self._df.loc[ts]


# ...
# TODO later allow features to be a list of feature sets echa witha possible different shape

# Also later we want to implement an envoronment where the agent has access to all assets and he needs to pick a combination
# of assets like in a portfolio construction manner
