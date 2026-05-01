import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from typing import Sequence, Dict
import numpy as np

def soft_update_params(live_params, target_params, tau):
    return jax.tree.map(lambda l, t: tau * l + (1.0 - tau) * t, live_params, target_params)

class InfluenceEstimator(nn.Module):
    obs_dim: int
    action_dim: int
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, obs_k, action_k, action_i):
        x = jnp.concatenate([obs_k, action_k, action_i], axis=-1)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        next_obs_pred = nn.Dense(self.obs_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return next_obs_pred

class BaselinePredictor(nn.Module):
    obs_dim: int
    action_dim: int
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, obs_k, action_k):
        x = jnp.concatenate([obs_k, action_k], axis=-1)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        next_obs_pred = nn.Dense(self.obs_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return next_obs_pred

def create_influence_estimator_state(rng, obs_dim: int, action_dim: int, hidden_dim: int=64, learning_rate: float=0.001):
    (rng, rng_inf, rng_base) = jax.random.split(rng, 3)
    influence_net = InfluenceEstimator(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    dummy_obs = jnp.zeros((1, obs_dim))
    dummy_action = jnp.zeros((1, action_dim))
    influence_params = influence_net.init(rng_inf, dummy_obs, dummy_action, dummy_action)
    influence_tx = optax.adam(learning_rate)
    influence_state = TrainState.create(apply_fn=influence_net.apply, params=influence_params, tx=influence_tx)
    baseline_net = BaselinePredictor(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim)
    baseline_params = baseline_net.init(rng_base, dummy_obs, dummy_action)
    baseline_tx = optax.adam(learning_rate)
    baseline_state = TrainState.create(apply_fn=baseline_net.apply, params=baseline_params, tx=baseline_tx)
    inf_target_params = jax.tree.map(jnp.copy, influence_state.params)
    base_target_params = jax.tree.map(jnp.copy, baseline_state.params)
    return (influence_state, baseline_state, influence_net, baseline_net, inf_target_params, base_target_params)

def compute_influence_reward(influence_state: TrainState, baseline_state: TrainState, obs_all: jnp.ndarray, next_obs_all: jnp.ndarray, actions_all: jnp.ndarray, num_agents: int, action_dim: int, beta: float=1.0, sigma: float=1.0, influence_target_params=None, baseline_target_params=None):
    influence_apply = influence_state.apply_fn
    baseline_apply = baseline_state.apply_fn
    influence_params = influence_target_params if influence_target_params is not None else influence_state.params
    baseline_params = baseline_target_params if baseline_target_params is not None else baseline_state.params

    def compute_influence_for_agent_i(agent_i):
        action_i = actions_all[agent_i]

        def compute_influence_on_k(agent_k):
            is_self = agent_i == agent_k
            obs_k = obs_all[agent_k]
            next_obs_k = next_obs_all[agent_k]
            action_k = actions_all[agent_k]
            pred_with_influence = influence_apply(influence_params, obs_k, action_k, action_i)
            pred_baseline = baseline_apply(baseline_params, obs_k, action_k)
            error_with_influence = jnp.sum((next_obs_k - pred_with_influence) ** 2, axis=-1)
            error_baseline = jnp.sum((next_obs_k - pred_baseline) ** 2, axis=-1)
            influence_diff = (error_baseline - error_with_influence) / (2 * sigma ** 2)
            influence_diff = jnp.where(is_self, 0.0, influence_diff)
            return influence_diff
        agent_indices = jnp.arange(num_agents)
        influence_on_all = jax.vmap(compute_influence_on_k)(agent_indices)
        total_influence = jnp.sum(influence_on_all, axis=0)
        return beta * total_influence
    agent_indices = jnp.arange(num_agents)
    influence_rewards = jax.vmap(compute_influence_for_agent_i)(agent_indices)
    return influence_rewards

def update_influence_networks(influence_state: TrainState, baseline_state: TrainState, obs_all: jnp.ndarray, next_obs_all: jnp.ndarray, actions_all: jnp.ndarray, num_agents: int):
    batch_size = obs_all.shape[0]

    def influence_loss_fn(influence_params):
        total_loss = 0.0
        count = 0
        for i in range(num_agents):
            for k in range(num_agents):
                if i == k:
                    continue
                obs_k = obs_all[:, k]
                next_obs_k = next_obs_all[:, k]
                action_k = actions_all[:, k]
                action_i = actions_all[:, i]
                pred = influence_state.apply_fn(influence_params, obs_k, action_k, action_i)
                loss = jnp.mean((pred - next_obs_k) ** 2)
                total_loss += loss
                count += 1
        return total_loss / max(count, 1)

    def baseline_loss_fn(baseline_params):
        total_loss = 0.0
        for k in range(num_agents):
            obs_k = obs_all[:, k]
            next_obs_k = next_obs_all[:, k]
            action_k = actions_all[:, k]
            pred = baseline_state.apply_fn(baseline_params, obs_k, action_k)
            loss = jnp.mean((pred - next_obs_k) ** 2)
            total_loss += loss
        return total_loss / num_agents
    (influence_loss, influence_grads) = jax.value_and_grad(influence_loss_fn)(influence_state.params)
    new_influence_state = influence_state.apply_gradients(grads=influence_grads)
    (baseline_loss, baseline_grads) = jax.value_and_grad(baseline_loss_fn)(baseline_state.params)
    new_baseline_state = baseline_state.apply_gradients(grads=baseline_grads)
    losses = {'influence_loss': influence_loss, 'baseline_loss': baseline_loss}
    return (new_influence_state, new_baseline_state, losses)
