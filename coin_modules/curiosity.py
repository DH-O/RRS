import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from typing import Sequence
import numpy as np
from coin_modules.influence import soft_update_params

class CuriosityPredictor(nn.Module):
    output_dim: int
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, obs, action):
        x = jnp.concatenate([obs, action], axis=-1)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        pred = nn.Dense(self.output_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return pred

class JointCuriosityPredictor(nn.Module):
    state_dim: int
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, obs_all_flat, actions_all_flat):
        x = jnp.concatenate([obs_all_flat, actions_all_flat], axis=-1)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        pred = nn.Dense(self.state_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return pred

def create_curiosity_predictor_state(rng, obs_dim: int, action_dim: int, num_agents: int, state_dim: int=None, hidden_dim: int=64, learning_rate: float=0.001):
    if state_dim is None:
        state_dim = obs_dim * num_agents
    (rng, rng_obs, rng_state) = jax.random.split(rng, 3)
    obs_predictor = CuriosityPredictor(output_dim=obs_dim, hidden_dim=hidden_dim)
    dummy_obs = jnp.zeros((1, obs_dim))
    dummy_action = jnp.zeros((1, action_dim))
    obs_params = obs_predictor.init(rng_obs, dummy_obs, dummy_action)
    obs_tx = optax.adam(learning_rate)
    obs_predictor_state = TrainState.create(apply_fn=obs_predictor.apply, params=obs_params, tx=obs_tx)
    state_predictor = JointCuriosityPredictor(state_dim=state_dim, hidden_dim=hidden_dim * 2)
    dummy_obs_flat = jnp.zeros((1, obs_dim * num_agents))
    dummy_action_flat = jnp.zeros((1, action_dim * num_agents))
    state_params = state_predictor.init(rng_state, dummy_obs_flat, dummy_action_flat)
    state_tx = optax.adam(learning_rate)
    state_predictor_state = TrainState.create(apply_fn=state_predictor.apply, params=state_params, tx=state_tx)
    obs_target_params = jax.tree.map(jnp.copy, obs_predictor_state.params)
    state_target_params = jax.tree.map(jnp.copy, state_predictor_state.params)
    return (obs_predictor_state, state_predictor_state, obs_predictor, state_predictor, obs_target_params, state_target_params)

def compute_curiosity_reward(obs_predictor_state: TrainState, state_predictor_state: TrainState, obs_all: jnp.ndarray, next_obs_all: jnp.ndarray, actions_all: jnp.ndarray, world_state: jnp.ndarray=None, next_world_state: jnp.ndarray=None, num_agents: int=5, obs_weight: float=1.0, state_weight: float=1.0, obs_target_params=None, state_target_params=None):
    obs_apply = obs_predictor_state.apply_fn
    obs_params = obs_target_params if obs_target_params is not None else obs_predictor_state.params

    def compute_obs_error(obs, action, next_obs):
        pred = obs_apply(obs_params, obs, action)
        error = jnp.sum((next_obs - pred) ** 2, axis=-1)
        return error
    obs_errors = jax.vmap(compute_obs_error)(obs_all, actions_all, next_obs_all)
    mean_obs_error = jnp.mean(obs_errors, axis=0)
    if world_state is not None and next_world_state is not None:
        state_apply = state_predictor_state.apply_fn
        state_params = state_target_params if state_target_params is not None else state_predictor_state.params
        obs_flat = obs_all.reshape(-1, *obs_all.shape[2:])
        obs_flat = jnp.concatenate([obs_all[i] for i in range(num_agents)], axis=-1)
        actions_flat = jnp.concatenate([actions_all[i] for i in range(num_agents)], axis=-1)
        pred_state = state_apply(state_params, obs_flat, actions_flat)
        state_error = jnp.sum((next_world_state - pred_state) ** 2, axis=-1)
        curiosity_reward = obs_weight * mean_obs_error + state_weight * state_error
    else:
        curiosity_reward = obs_weight * mean_obs_error
    return curiosity_reward

def update_curiosity_networks(obs_predictor_state: TrainState, state_predictor_state: TrainState, obs_all: jnp.ndarray, next_obs_all: jnp.ndarray, actions_all: jnp.ndarray, world_state: jnp.ndarray=None, next_world_state: jnp.ndarray=None, num_agents: int=5):
    batch_size = obs_all.shape[0]

    def obs_loss_fn(params):
        total_loss = 0.0
        for i in range(num_agents):
            obs_i = obs_all[:, i]
            next_obs_i = next_obs_all[:, i]
            action_i = actions_all[:, i]
            pred = obs_predictor_state.apply_fn(params, obs_i, action_i)
            loss = jnp.mean((pred - next_obs_i) ** 2)
            total_loss += loss
        return total_loss / num_agents
    (obs_loss, obs_grads) = jax.value_and_grad(obs_loss_fn)(obs_predictor_state.params)
    new_obs_state = obs_predictor_state.apply_gradients(grads=obs_grads)
    state_loss = jnp.array(0.0)
    new_state_state = state_predictor_state
    if world_state is not None and next_world_state is not None:

        def state_loss_fn(params):
            obs_flat = jnp.concatenate([obs_all[:, i] for i in range(num_agents)], axis=-1)
            actions_flat = jnp.concatenate([actions_all[:, i] for i in range(num_agents)], axis=-1)
            pred = state_predictor_state.apply_fn(params, obs_flat, actions_flat)
            return jnp.mean((pred - next_world_state) ** 2)
        (state_loss, state_grads) = jax.value_and_grad(state_loss_fn)(state_predictor_state.params)
        new_state_state = state_predictor_state.apply_gradients(grads=state_grads)
    losses = {'obs_predictor_loss': obs_loss, 'state_predictor_loss': state_loss}
    return (new_obs_state, new_state_state, losses)
