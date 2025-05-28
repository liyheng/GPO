import collections
import math
import os
from typing import Any, Callable, Dict, Optional, Sequence
from wrappers import create
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import sys
from pathlib import Path
from config import get_config
from torch.utils.tensorboard import SummaryWriter

class Agent(nn.Module):
  def __init__(self,
               policy_layers: Sequence[int],
               value_layers: Sequence[int],
               entropy_cost: float,
               discounting: float,
               reward_scaling: float,
               eps: float,
               alpha: float,
               use_clip: bool,
               device: str):
    super(Agent, self).__init__()

    policy = []
    for w1, w2 in zip(policy_layers, policy_layers[1:]):
      policy.append(nn.Linear(w1, w2))
      policy.append(nn.SiLU())
    policy.pop()  # drop the final activation
    self.policy = nn.Sequential(*policy)

    value = []
    for w1, w2 in zip(value_layers, value_layers[1:]):
      value.append(nn.Linear(w1, w2))
      value.append(nn.SiLU())
    value.pop()  # drop the final activation
    self.value = nn.Sequential(*value)

    self.num_steps = torch.zeros((), device=device)
    self.running_mean = torch.zeros(policy_layers[0], device=device)
    self.running_variance = torch.zeros(policy_layers[0], device=device)

    self.entropy_cost = entropy_cost
    self.discounting = discounting
    self.reward_scaling = reward_scaling
    self.lambda_ = 0.95
    self.epsilon = 0.3
    self.eps = eps
    self.alpha = alpha
    self.use_clip = use_clip
    self.device = device

  @torch.jit.export
  def dist_create(self, logits):
    """Normal followed by tanh.

    torch.distribution doesn't work with torch.jit, so we roll our own."""
    loc, scale = torch.split(logits, logits.shape[-1] // 2, dim=-1)
    scale = F.softplus(scale) + .001
    return loc, scale

  @torch.jit.export
  def dist_sample_no_postprocess(self, loc, scale):
    return torch.normal(loc, scale)

  @classmethod
  def dist_postprocess(cls, x):
    return torch.tanh(x)

  @torch.jit.export
  def dist_entropy(self, loc, scale):
    log_normalized = 0.5 * math.log(2 * math.pi) + torch.log(scale)
    entropy = 0.5 + log_normalized
    entropy = entropy * torch.ones_like(loc)
    dist = torch.normal(loc, scale)
    log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
    entropy = entropy + log_det_jacobian
    return entropy.sum(dim=-1)

  @torch.jit.export
  def dist_log_prob(self, loc, scale, dist):
    log_unnormalized = -0.5 * ((dist - loc) / scale).square()
    log_normalized = 0.5 * math.log(2 * math.pi) + torch.log(scale)
    log_det_jacobian = 2 * (math.log(2) - dist - F.softplus(-2 * dist))
    log_prob = log_unnormalized - log_normalized - log_det_jacobian
    return log_prob.sum(dim=-1)

  @torch.jit.export
  def update_normalization(self, observation):
    self.num_steps += observation.shape[0] * observation.shape[1]
    input_to_old_mean = observation - self.running_mean
    mean_diff = torch.sum(input_to_old_mean / self.num_steps, dim=(0, 1))
    self.running_mean = self.running_mean + mean_diff
    input_to_new_mean = observation - self.running_mean
    var_diff = torch.sum(input_to_new_mean * input_to_old_mean, dim=(0, 1))
    self.running_variance = self.running_variance + var_diff

  @torch.jit.export
  def normalize(self, observation):
    variance = self.running_variance / (self.num_steps + 1.0)
    variance = torch.clip(variance, 1e-6, 1e6)
    return ((observation - self.running_mean) / variance.sqrt()).clip(-5, 5)
  
  @torch.jit.export
  def get_logits_action(self, observation):
    observation = self.normalize(observation)
    logits = self.policy(observation)
    loc, scale = self.dist_create(logits)
    action = self.dist_sample_no_postprocess(loc, scale)
    return logits, action

  @torch.jit.export
  def compute_gae(self, truncation, termination, reward, values,
                  bootstrap_value):
    truncation_mask = 1 - truncation
    # Append bootstrapped value to get [v1, ..., v_t+1]
    values_t_plus_1 = torch.cat(
        [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0)
    deltas = reward + self.discounting * (
        1 - termination) * values_t_plus_1 - values
    deltas *= truncation_mask

    acc = torch.zeros_like(bootstrap_value)
    vs_minus_v_xs = torch.zeros_like(truncation_mask)

    for ti in range(truncation_mask.shape[0]):
      ti = truncation_mask.shape[0] - ti - 1
      acc = deltas[ti] + self.discounting * (
          1 - termination[ti]) * truncation_mask[ti] * self.lambda_ * acc
      vs_minus_v_xs[ti] = acc

    # Add V(x_s) to get v_s.
    vs = vs_minus_v_xs + values
    vs_t_plus_1 = torch.cat([vs[1:], torch.unsqueeze(bootstrap_value, 0)], 0)
    advantages = (reward + self.discounting *
                  (1 - termination) * vs_t_plus_1 - values) * truncation_mask
    return vs, advantages
  
  @torch.jit.export
  def kl_divergence(self, p_loc, p_scale, q_loc, q_scale):
    """Calculate KL divergence between two normal distributions."""
    var_ratio = (p_scale / q_scale).pow(2)
    t1 = ((p_loc - q_loc) / q_scale).pow(2)
    kl = 0.5 * (var_ratio + t1 - 1 - var_ratio.log())
    return kl.mean(dim=-1)

  @torch.jit.export
  def stable_exp(self, x):
    """Clamped exp function"""
    return torch.exp(torch.clamp(x, max=10))
  
  @torch.jit.export
  def loss(self, td: Dict[str, torch.Tensor], kl_coef: float):
    # Normalize two observations together
    observation = self.normalize(td['observation'])
    pobservation = self.normalize(td['pobservation'])
    guider_policy_logits = self.policy(observation[:-1])
    learner_policy_logits = self.policy(pobservation[:-1])
    baseline = self.value(observation)
    baseline = torch.squeeze(baseline, dim=-1)
    # Use last baseline value (from the value function) to bootstrap.
    bootstrap_value = baseline[-1]
    baseline = baseline[:-1]
    reward = td['reward'] * self.reward_scaling
    termination = td['done'] * (1 - td['truncation'])

    loc, scale = self.dist_create(td['logits'])
    behaviour_action_log_probs = self.dist_log_prob(loc, scale, td['action'])
    loc_guider, scale_guider = self.dist_create(guider_policy_logits)
    guider_action_log_probs = self.dist_log_prob(loc_guider, scale_guider, td['action'])
    loc_learner, scale_learner = self.dist_create(learner_policy_logits)
    learner_action_log_probs = self.dist_log_prob(loc_learner, scale_learner, td['action'])
    
    # KL loss for guider and learner
    kl_loss_learner = self.kl_divergence(loc_guider.detach(), scale_guider.detach(), loc_learner, scale_learner)
    kl_loss_learner = torch.mean(kl_loss_learner)
    kl_loss_guider = self.kl_divergence(loc_guider, scale_guider, loc_learner.detach(), scale_learner.detach())
    if self.use_clip:
      m = torch.where(guider_action_log_probs - learner_action_log_probs > torch.log(1 + self.eps), 1., 
                      torch.where(guider_action_log_probs - learner_action_log_probs<torch.log(1 - self.eps), 1., 0.))
      kl_loss_guider = torch.mean(kl_loss_guider * m)
    else:
      kl_loss_guider = torch.mean(kl_loss_guider) * kl_coef

    with torch.no_grad():
      vs, advantages = self.compute_gae(
          truncation=td['truncation'],
          termination=termination,
          reward=reward,
          values=baseline,
          bootstrap_value=bootstrap_value)
      advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
    
    # Guider RL loss
    rho = torch.exp(guider_action_log_probs - behaviour_action_log_probs)
    if self.use_clip:
      rho_clip = torch.exp((guider_action_log_probs - learner_action_log_probs.detach()).clip(
                        torch.log(1 - self.eps), 
                        torch.log(1 + self.eps)) \
                       + learner_action_log_probs.detach() - behaviour_action_log_probs)
      rho_clip = rho_clip.clip(1 - self.epsilon,1 + self.epsilon)
    else:
      rho_clip = rho.clip(1 - self.epsilon,1 + self.epsilon)

    surrogate_loss1 = rho * advantages
    surrogate_loss2 = rho_clip * advantages
    guider_policy_loss = -torch.mean(torch.minimum(surrogate_loss1, surrogate_loss2))

    # Learner RL loss
    # Avoid numerical instability when learner is not able to follow behaviour policy
    rho = self.stable_exp(learner_action_log_probs - behaviour_action_log_probs)
    rho_clip = rho.clip(1 - self.epsilon, 1 + self.epsilon)
    surrogate_loss1 = rho * advantages
    surrogate_loss2 = rho_clip * advantages
    learner_policy_loss = -torch.mean(torch.minimum(surrogate_loss1, surrogate_loss2))

    # Value function loss
    v_error = vs - baseline
    v_loss = torch.mean(v_error * v_error) * 0.5 * 0.5

    # Entropy reward
    entropy = torch.mean(self.dist_entropy(loc, scale))
    entropy_loss = self.entropy_cost * -entropy

    # Total loss
    if self.use_clip:
      learner_loss = kl_loss_learner + learner_policy_loss * self.alpha 
    else:
      learner_loss = kl_loss_learner + learner_policy_loss * kl_coef
    guider_loss = kl_loss_guider + guider_policy_loss
    total_loss = learner_loss + guider_loss + v_loss + entropy_loss
    
    return  total_loss, guider_loss, learner_loss, kl_loss_learner
  
StepData = collections.namedtuple(
    'StepData',
    ('observation','pobservation', 'logits', 'action', 'reward', 'done', 'truncation'))


def sd_map(f: Callable[..., torch.Tensor], *sds) -> StepData:
  """Map a function over each field in StepData."""
  items = {}
  keys = sds[0]._asdict().keys()
  for k in keys:
    items[k] = f(*[sd._asdict()[k] for sd in sds])
  return StepData(**items)

def eval_unroll(agent, env, length):
  """Return number of episodes and average reward for a single unroll."""
  _, pobservation = env.reset()
  episodes = torch.zeros((), device=agent.device)
  episode_reward = torch.zeros((), device=agent.device)
  for _ in range(length):
    # make sure only partial observations are used
    _, action = agent.get_logits_action(pobservation)
    _, pobservation, reward, done, info = env.step(Agent.dist_postprocess(action))
    episodes += torch.sum(done)
    episode_reward += torch.sum(reward)
  return episodes, episode_reward / episodes

def train_unroll(agent, env, observation, pobservation, num_unrolls, unroll_length):
  """Return step data over multple unrolls."""
  sd = StepData([], [], [], [], [], [], [])
  for _ in range(num_unrolls):
    one_unroll = StepData([observation], [pobservation], [], [], [], [], [])
    for _ in range(unroll_length):
      logits, action = agent.get_logits_action(observation)
      observation, pobservation, reward, done, info = env.step(Agent.dist_postprocess(action))
      one_unroll.observation.append(observation)
      one_unroll.pobservation.append(pobservation)
      one_unroll.logits.append(logits)
      one_unroll.action.append(action)
      one_unroll.reward.append(reward)
      one_unroll.done.append(done)
      one_unroll.truncation.append(info['truncation'])
    one_unroll = sd_map(torch.stack, one_unroll)
    sd = sd_map(lambda x, y: x + [y], sd, one_unroll)
  td = sd_map(torch.stack, sd)
  return observation, pobservation, td

def train(
    env_name: str = 'ant',
    num_envs: int = 2048,
    episode_length: int = 1000,
    device: str = 'cuda',
    num_timesteps: int = 30_000_000,
    eval_frequency: int = 10,
    unroll_length: int = 5,
    batch_size: int = 512,
    num_minibatches: int = 4,
    num_update_epochs: int = 4,
    reward_scaling: float = 1.,
    entropy_cost: float = 1e-3,
    discounting: float = .99,
    learning_rate: float = 3e-4,
    seed: int = 0,
    target_kl: float = 0.01,
    eps: float = 0.3,
    alpha: float = 2.,
    use_clip: bool = True,
):
  """Trains a policy via PPO."""
  env = create(env_name, batch_size=num_envs, episode_length=episode_length, seed=seed, device=device)
  folder_path = Path(env_name)
  folder_path.mkdir(parents=True, exist_ok=True)
  filename = env_name + '/' + str(seed)
  writer = SummaryWriter(filename)

  # env warmup
  env.reset()
  action = torch.zeros(env.action_space.shape).to(device)
  env.step(action)
  hidden = 128
  kl_coef = 0
  policy_layers = [
      env.observation_space.shape[-1], hidden, hidden, env.action_space.shape[-1] * 2
  ]
  value_layers = [env.observation_space.shape[-1], hidden, hidden, 1]
  agent = Agent(policy_layers, value_layers, entropy_cost, discounting,
                reward_scaling, eps, alpha, use_clip, device)
  agent = torch.jit.script(agent.to(device))
  optimizer = optim.Adam(agent.parameters(), lr=learning_rate)

  total_steps = 0
  for eval_i in range(eval_frequency + 1):

    with torch.no_grad():
      episode_count, episode_reward = eval_unroll(agent, env, episode_length)
      writer.add_scalar("charts/episodic_return", episode_reward, total_steps)
      print(episode_reward)
    if eval_i == eval_frequency:
      break

    observation, pobservation = env.reset()
    num_steps = batch_size * num_minibatches * unroll_length
    num_epochs = num_timesteps // (num_steps * eval_frequency)
    num_unrolls = batch_size * num_minibatches // env.num_envs
    for epoch in range(num_epochs):
      observation, pobservation, td = train_unroll(agent, env, observation, pobservation, num_unrolls,
                                     unroll_length)

      # make unroll first
      def unroll_first(data):
        data = data.swapaxes(0, 1)
        return data.reshape([data.shape[0], -1] + list(data.shape[3:]))
      td = sd_map(unroll_first, td)

      # update normalization statistics
      # NOTE: We choose to normalize observations and partial observations together, but they can also be done separately
      agent.update_normalization(td.observation)
      agent.update_normalization(td.pobservation)
      for _ in range(num_update_epochs):
        kl_epochs = 0
        loss_epochs = 0
        guider_loss_epochs = 0
        learner_loss_epochs = 0
        # shuffle and batch the data
        with torch.no_grad():
          permutation = torch.randperm(td.observation.shape[1], device=device)
          def shuffle_batch(data):
            data = data[:, permutation]
            data = data.reshape([data.shape[0], num_minibatches, -1] +
                                list(data.shape[2:]))
            return data.swapaxes(0, 1)
          epoch_td = sd_map(shuffle_batch, td)

        for minibatch_i in range(num_minibatches):
          td_minibatch = sd_map(lambda d: d[minibatch_i], epoch_td)
          loss, guider_loss, learner_loss, kl = agent.loss(td_minibatch._asdict(), kl_coef)
          kl_epochs += kl
          loss_epochs += loss
          guider_loss_epochs += guider_loss
          learner_loss_epochs += learner_loss
          optimizer.zero_grad()
          loss.backward()
          nn.utils.clip_grad_norm_(agent.parameters(), 1)
          optimizer.step()
        d = kl_epochs/num_minibatches
        if not use_clip:
          if kl_coef < 1e-5 and d > target_kl:
            kl_coef = 0.1
          if d > target_kl*1.2: kl_coef = kl_coef * 1.2
          if d < target_kl/1.2: kl_coef = kl_coef / 1.2
          kl_coef = min(kl_coef, 10)

        writer.add_scalar("losses/loss", (loss_epochs/num_minibatches).item(), total_steps + epoch * num_steps)
        writer.add_scalar("losses/guider_loss", (guider_loss_epochs/num_minibatches).item(), total_steps + epoch * num_steps)
        writer.add_scalar("losses/learner_loss", (learner_loss_epochs/num_minibatches).item(), total_steps + epoch * num_steps)
        writer.add_scalar("losses/kl", d.item(), total_steps + epoch * num_steps)
      
    total_steps += num_epochs * num_steps


if __name__ == "__main__":
  os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
  args = sys.argv[1:]
  parser = get_config()
  args = parser.parse_known_args(args)[0]

  train(env_name = args.env_name,
        num_envs = args.num_envs,
        episode_length = args.episode_length,
        device = args.device,
        num_timesteps = args.num_timesteps,
        eval_frequency = args.eval_frequency,
        unroll_length = args.unroll_length,
        batch_size = args.batch_size,
        num_minibatches = args.num_minibatches,
        num_update_epochs = args.num_update_epochs,
        reward_scaling = args.reward_scaling,
        entropy_cost = args.entropy_cost,
        discounting = args.discounting,
        learning_rate = args.learning_rate,
        seed = args.seed,
        target_kl = args.target_kl,
        eps = args.eps,
        alpha = args.alpha,
        use_clip = args.use_clip)
