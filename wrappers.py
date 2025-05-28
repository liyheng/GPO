from typing import ClassVar, Dict, Optional, Tuple
from brax.envs.base import Env, State, Wrapper, PipelineEnv
import jax
from jax import numpy as jp
from typing import Optional, Type
import numpy as np
import gym
from gym import spaces
from brax.io import torch
from gym.vector import utils
from envs import ant, half_cheetah, humanoid, humanoidstandup, walker2d, swimmer, inverted_double_pendulum

_envs = {
    'StatelessAnt': ant.StatelessAnt,
    'NoisyStatelessAntEasy': ant.NoisyStatelessAntEasy,
    'NoisyStatelessAntMedium': ant.NoisyStatelessAntMedium,
    'NoisyStatelessAntHard': ant.NoisyStatelessAntHard,
    'StatelessHalfcheetah': half_cheetah.StatelessHalfcheetah,
    'NoisyStatelessHalfcheetahEasy': half_cheetah.NoisyStatelessHalfcheetahEasy,
    'NoisyStatelessHalfcheetahMedium': half_cheetah.NoisyStatelessHalfcheetahMedium,
    'NoisyStatelessHalfcheetahHard': half_cheetah.NoisyStatelessHalfcheetahHard,
    'StatelessHumanoid': humanoid.StatelessHumanoid,
    'NoisyStatelessHumanoidEasy': humanoid.NoisyStatelessHumanoidEasy,
    'NoisyStatelessHumanoidMedium': humanoid.NoisyStatelessHumanoidMedium,
    'NoisyStatelessHumanoidHard': humanoid.NoisyStatelessHumanoidHard,
    'StatelessHumanoidStandup': humanoidstandup.StatelessHumanoidStandup,
    'NoisyStatelessHumanoidStandupEasy': humanoidstandup.NoisyStatelessHumanoidStandupEasy,
    'NoisyStatelessHumanoidStandupMedium': humanoidstandup.NoisyStatelessHumanoidStandupMedium,
    'NoisyStatelessHumanoidStandupHard': humanoidstandup.NoisyStatelessHumanoidStandupHard,
    'StatelessWalker2d': walker2d.StatelessWalker2d,
    'NoisyStatelessWalker2dEasy': walker2d.NoisyStatelessWalker2dEasy,
    'NoisyStatelessWalker2dMedium': walker2d.NoisyStatelessWalker2dMedium,
    'NoisyStatelessWalker2dHard': walker2d.NoisyStatelessWalker2dHard,
    'StatelessSwimmer': swimmer.StatelessSwimmer,
    'NoisyStatelessSwimmerEasy': swimmer.NoisyStatelessSwimmerEasy,
    'NoisyStatelessSwimmerMedium': swimmer.NoisyStatelessSwimmerMedium,
    'NoisyStatelessSwimmerHard': swimmer.NoisyStatelessSwimmerHard,
    'StatelessInvertedDoublePendulum': inverted_double_pendulum.StatelessInvertedDoublePendulum,
    'NoisyStatelessInvertedDoublePendulumEasy': inverted_double_pendulum.NoisyStatelessInvertedDoublePendulumEasy,
    'NoisyStatelessInvertedDoublePendulumMedium': inverted_double_pendulum.NoisyStatelessInvertedDoublePendulumMedium,
    'NoisyStatelessInvertedDoublePendulumHard': inverted_double_pendulum.NoisyStatelessInvertedDoublePendulumHard,
}

def create(
    env_name: str,
    episode_length: int = 1000,
    action_repeat: int = 1,
    auto_reset: bool = True,
    batch_size: Optional[int] = None,
    seed: int = 0,
    device: str = 'cuda',
    **kwargs,
) -> Env:
  """Creates an environment from the registry.

  Args:
    env_name: environment name string
    episode_length: length of episode
    action_repeat: how many repeated actions to take per environment step
    auto_reset: whether to auto reset the environment after an episode is done
    batch_size: the number of environments to batch together
    **kwargs: keyword argments that get passed to the Env class constructor

  Returns:
    env: an environment
  """
  env = _envs[env_name](**kwargs)

  if episode_length is not None:
    env = EpisodeWrapper(env, episode_length, action_repeat)
  if batch_size:
    env = VmapWrapper(env, batch_size)
  if auto_reset:
    env = AutoResetWrapper(env)
  env = VectorGymWrapper(env, seed)
  env = TorchWrapper(env, device=device)
  return env

class EpisodeWrapper(Wrapper):
  """Maintains episode step count and sets done at episode end."""

  def __init__(self, env: Env, episode_length: int, action_repeat: int):
    super().__init__(env)
    self.episode_length = episode_length
    self.action_repeat = action_repeat

  def reset(self, rng: jax.Array) -> State:
    state = self.env.reset(rng)
    state.info['steps'] = jp.zeros(rng.shape[:-1])
    state.info['truncation'] = jp.zeros(rng.shape[:-1])
    # Keep separate record of episode done as state.info['done'] can be erased
    # by AutoResetWrapper
    state.info['episode_done'] = jp.zeros(rng.shape[:-1])
    episode_metrics = dict()
    episode_metrics['sum_reward'] = jp.zeros(rng.shape[:-1])
    episode_metrics['length'] = jp.zeros(rng.shape[:-1])
    for metric_name in state.metrics.keys():
      episode_metrics[metric_name] = jp.zeros(rng.shape[:-1])
    state.info['episode_metrics'] = episode_metrics
    return state

  def step(self, rng: jax.Array, state: State, action: jax.Array) -> State:
    def f(state, _):
      nstate = self.env.step(rng, state, action)
      return nstate, nstate.reward

    state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
    state = state.replace(reward=jp.sum(rewards, axis=0))
    steps = state.info['steps'] + self.action_repeat
    one = jp.ones_like(state.done)
    zero = jp.zeros_like(state.done)
    episode_length = jp.array(self.episode_length, dtype=jp.int32)
    done = jp.where(steps >= episode_length, one, state.done)
    state.info['truncation'] = jp.where(
        steps >= episode_length, 1 - state.done, zero
    )
    state.info['steps'] = steps

    # Aggregate state metrics into episode metrics
    prev_done = state.info['episode_done']
    state.info['episode_metrics']['sum_reward'] += jp.sum(rewards, axis=0)
    state.info['episode_metrics']['sum_reward'] *= (1 - prev_done)
    state.info['episode_metrics']['length'] += self.action_repeat
    state.info['episode_metrics']['length'] *= (1 - prev_done)
    for metric_name in state.metrics.keys():
      if metric_name != 'reward':
        state.info['episode_metrics'][metric_name] += state.metrics[metric_name]
        state.info['episode_metrics'][metric_name] *= (1 - prev_done)
    state.info['episode_done'] = done
    return state.replace(done=done)

class VmapWrapper(Wrapper):
  """Vectorizes Brax env."""

  def __init__(self, env: Env, batch_size: Optional[int] = None):
    super().__init__(env)
    self.batch_size = batch_size

  def reset(self, rng: jax.Array) -> State:
    if self.batch_size is not None:
      rng = jax.random.split(rng, self.batch_size)
    return jax.vmap(self.env.reset)(rng)

  def step(self, rng: jax.Array, state: State, action: jax.Array) -> State:
    if self.batch_size is not None:
      rng = jax.random.split(rng, self.batch_size)
    return jax.vmap(self.env.step)(rng, state, action)
   
class AutoResetWrapper(Wrapper):
  """Automatically resets Brax envs that are done."""

  def reset(self, rng: jax.Array) -> State:
    state = self.env.reset(rng)
    state.info['first_pipeline_state'] = state.pipeline_state
    state.info['first_obs'] = state.obs
    state.info['first_pobs'] = state.pobs
    return state

  def step(self, rng: jax.Array, state: State, action: jax.Array) -> State:
    if 'steps' in state.info:
      steps = state.info['steps']
      steps = jp.where(state.done, jp.zeros_like(steps), steps)
      state.info.update(steps=steps)
    state = state.replace(done=jp.zeros_like(state.done))
    state = self.env.step(rng, state, action)

    def where_done(x, y):
      done = state.done
      if done.shape:
        done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
      return jp.where(done, x, y)

    pipeline_state = jax.tree_map(
        where_done, state.info['first_pipeline_state'], state.pipeline_state
    )
    obs = jax.tree_map(where_done, state.info['first_obs'], state.obs)
    pobs = jax.tree_map(where_done, state.info['first_pobs'], state.pobs)
    return state.replace(pipeline_state=pipeline_state, obs=obs, pobs=pobs)

class VectorGymWrapper(gym.vector.VectorEnv):
  """A wrapper that converts batched Brax Env to one that follows Gym VectorEnv API."""

  # Flag that prevents `gym.register` from misinterpreting the `_step` and
  # `_reset` as signs of a deprecated gym Env API.
  _gym_disable_underscore_compat: ClassVar[bool] = True

  def __init__(
      self, env: PipelineEnv, seed: int = 0, backend: Optional[str] = None
  ):
    self._env = env
    self.metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 1 / self._env.dt,
    }
    if not hasattr(self._env, 'batch_size'):
      raise ValueError('underlying env must be batched')

    self.num_envs = self._env.batch_size
    self.seed(seed)
    self.backend = backend
    self._state = None

    obs = np.inf * np.ones(self._env.observation_size, dtype='float32')
    obs_space = spaces.Box(-obs, obs, dtype='float32')
    self.observation_space = utils.batch_space(obs_space, self.num_envs)

    action = jax.tree_map(np.array, self._env.sys.actuator.ctrl_range)
    action_space = spaces.Box(action[:, 0], action[:, 1], dtype='float32')
    self.action_space = utils.batch_space(action_space, self.num_envs)

    def reset(key):
      key1, key2 = jax.random.split(key)
      state = self._env.reset(key2)
      return state, state.obs, state.pobs, key1

    self._reset = jax.jit(reset, backend=self.backend)

    def step(key, state, action):
      key1, key2 = jax.random.split(key)
      state = self._env.step(key2, state, action)
      info = {**state.metrics, **state.info}
      return state, state.obs, state.pobs, state.reward, state.done, info, key1

    self._step = jax.jit(step, backend=self.backend)

  def reset(self):
    self._state, obs, pobs, self._key1 = self._reset(self._key1)
    return obs, pobs

  def step(self, action):
    self._state, obs, pobs, reward, done, info, self._key2 = self._step(self._key2, self._state, action)
    return obs, pobs, reward, done, info

  def seed(self, seed: int = 0):
    self._key1, self._key2 = jax.random.split(jax.random.PRNGKey(seed))

class TorchWrapper(gym.Wrapper):
  """Wrapper that converts Jax tensors to PyTorch tensors."""

  def __init__(self, env: gym.Env, device: Optional[str] = None):
    """Creates a gym Env to one that outputs PyTorch tensors."""
    super().__init__(env)
    self.device = device

  def reset(self):
    obs, pobs = super().reset()
    return torch.jax_to_torch(obs, device=self.device), torch.jax_to_torch(pobs, device=self.device)

  def step(self, action):
    action = torch.torch_to_jax(action)
    obs, pobs, reward, done, info = super().step(action)
    obs = torch.jax_to_torch(obs, device=self.device)
    pobs = torch.jax_to_torch(pobs, device=self.device)
    reward = torch.jax_to_torch(reward, device=self.device)
    done = torch.jax_to_torch(done, device=self.device)
    info = torch.jax_to_torch(info, device=self.device)
    return obs, pobs, reward, done, info
