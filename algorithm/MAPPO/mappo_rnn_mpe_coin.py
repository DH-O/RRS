import os
import sys
import setup_jax
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Dict
from flax.training.train_state import TrainState
import distrax
import hydra
from omegaconf import OmegaConf
from functools import partial
import jaxmarl
from jaxmarl.wrappers.baselines import MPELogWrapper, LogWrapper, JaxMARLWrapper
import wandb
import yaml
from safetensors.flax import save_file
from flax.traverse_util import flatten_dict
from time import time
from core import SMAXWorldStateWrapper
from coin_modules import create_influence_estimator_state, compute_influence_reward, update_influence_networks, create_curiosity_predictor_state, compute_curiosity_reward, update_curiosity_networks, soft_update_params

class MPEWorldStateWrapper(JaxMARLWrapper):

    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        (obs, env_state) = self._env.reset(key)
        obs['world_state'] = self.world_state(obs)
        return (obs, env_state)

    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, action):
        (obs, env_state, reward, done, info) = self._env.step(key, state, action)
        obs['world_state'] = self.world_state(obs)
        return (obs, env_state, reward, done, info)

    @partial(jax.jit, static_argnums=0)
    def world_state(self, obs):

        @partial(jax.vmap, in_axes=(0, None))
        def _roll_obs(aidx, all_obs):
            robs = jnp.roll(all_obs, -aidx, axis=0)
            robs = robs.flatten()
            return robs
        all_obs = jnp.array([obs[agent] for agent in self._env.agents]).flatten()
        all_obs = jnp.expand_dims(all_obs, axis=0).repeat(len(self._env.agents), axis=0)
        return all_obs

    def world_state_size(self):
        spaces = [self._env.observation_space(agent) for agent in self._env.agents]
        return sum([space.shape[-1] for space in spaces])

class ScannedRNN(nn.Module):

    @partial(nn.scan, variable_broadcast='params', in_axes=0, out_axes=0, split_rngs={'params': False})
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        (ins, resets) = x
        rnn_state = jnp.where(resets[:, np.newaxis], self.initialize_carry(rnn_state.shape[0], rnn_state.shape[1]), rnn_state)
        (new_rnn_state, y) = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return (new_rnn_state, y)

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))

class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        (obs, dones) = x
        embedding = nn.Dense(self.config['FC_DIM_SIZE'], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        embedding = nn.relu(embedding)
        rnn_in = (embedding, dones)
        (hidden, embedding) = ScannedRNN()(hidden, rnn_in)
        actor_mean = nn.Dense(self.config['GRU_HIDDEN_DIM'], kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        actor_mean = nn.relu(actor_mean)
        action_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        pi = distrax.Categorical(logits=action_logits)
        return (hidden, pi, action_logits)

class CriticRNN(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        (world_state, dones) = x
        embedding = nn.Dense(self.config['FC_DIM_SIZE'], kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(world_state)
        embedding = nn.relu(embedding)
        rnn_in = (embedding, dones)
        (hidden, embedding) = ScannedRNN()(hidden, rnn_in)
        critic = nn.Dense(self.config['GRU_HIDDEN_DIM'], kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return (hidden, jnp.squeeze(critic, axis=-1))

class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    info: jnp.ndarray
    action_onehot: jnp.ndarray

def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for (i, a) in enumerate(agent_list)}

def make_train(config):
    env_kwargs = config.get('ENV_KWARGS', {})
    env_name = config['ENV_NAME']
    sparse_reward = config.get('SPARSE_REWARD', False)
    if 'corridor' in env_name.lower():
        env_kwargs['sparse_reward'] = sparse_reward
    env = jaxmarl.make(env_name, **env_kwargs)
    config['NUM_ACTORS'] = len(env.agents) * config['NUM_ENVS']
    config['NUM_UPDATES'] = config['TOTAL_TIMESTEPS'] // config['NUM_STEPS'] // config['NUM_ENVS']
    config['MINIBATCH_SIZE'] = config['NUM_ACTORS'] * config['NUM_STEPS'] // config['NUM_MINIBATCHES']
    config['CLIP_EPS'] = config['CLIP_EPS'] / len(env.agents) if config['SCALE_CLIP_EPS'] else config['CLIP_EPS']
    num_envs = config['NUM_ENVS']
    num_agents = len(env.agents)
    action_dim = env.action_space(env.agents[0]).n
    obs_dim = env.observation_space(env.agents[0]).shape[0]
    coin_config = config.get('COIN', {})
    influence_hidden_dim = coin_config.get('INFLUENCE_HIDDEN_DIM', 64)
    influence_lr = coin_config.get('INFLUENCE_LR', 0.001)
    influence_beta = coin_config.get('INFLUENCE_BETA', 1.0)
    influence_sigma = coin_config.get('INFLUENCE_SIGMA', 1.0)
    curiosity_hidden_dim = coin_config.get('CURIOSITY_HIDDEN_DIM', 64)
    curiosity_lr = coin_config.get('CURIOSITY_LR', 0.001)
    obs_weight = coin_config.get('OBS_WEIGHT', 1.0)
    state_weight = coin_config.get('STATE_WEIGHT', 0.5)
    influence_scale = coin_config.get('INFLUENCE_SCALE', 1.0)
    curiosity_scale = coin_config.get('CURIOSITY_SCALE', 0.5)
    warmup_rollouts = coin_config.get('WARMUP_ROLLOUTS', 5)
    target_tau = coin_config.get('TARGET_TAU', 0.005)
    intrinsic_scale = config.get('INTRINSIC_REWARD_SCALE', 1.0)
    is_smax = 'SMAX' in env_name or 'smax' in env_name
    if is_smax:
        env = SMAXWorldStateWrapper(env)
        env = LogWrapper(env)
    else:
        env = MPEWorldStateWrapper(env)
        env = MPELogWrapper(env)

    def linear_schedule(count):
        frac = 1.0 - count // (config['NUM_MINIBATCHES'] * config['UPDATE_EPOCHS']) / config['NUM_UPDATES']
        return config['LR'] * frac

    def train(rng):
        actor_network = ActorRNN(action_dim, config=config)
        critic_network = CriticRNN(config=config)
        (rng, _rng_actor, _rng_critic, _rng_coin) = jax.random.split(rng, 4)
        ac_init_x = (jnp.zeros((1, config['NUM_ENVS'], obs_dim)), jnp.zeros((1, config['NUM_ENVS'])))
        ac_init_hstate = ScannedRNN.initialize_carry(config['NUM_ENVS'], config['GRU_HIDDEN_DIM'])
        actor_network_params = actor_network.init(_rng_actor, ac_init_hstate, ac_init_x)
        cr_init_x = (jnp.zeros((1, config['NUM_ENVS'], env.world_state_size())), jnp.zeros((1, config['NUM_ENVS'])))
        cr_init_hstate = ScannedRNN.initialize_carry(config['NUM_ENVS'], config['GRU_HIDDEN_DIM'])
        critic_network_params = critic_network.init(_rng_critic, cr_init_hstate, cr_init_x)
        tx = optax.chain(optax.clip_by_global_norm(config['MAX_GRAD_NORM']), optax.adam(config['LR'], eps=1e-05))
        actor_train_state = TrainState.create(apply_fn=actor_network.apply, params=actor_network_params, tx=tx)
        critic_train_state = TrainState.create(apply_fn=critic_network.apply, params=critic_network_params, tx=tx)
        (rng_inf, rng_cur) = jax.random.split(_rng_coin, 2)
        (influence_state, baseline_state, _, _, inf_target_params, base_target_params) = create_influence_estimator_state(rng=rng_inf, obs_dim=obs_dim, action_dim=action_dim, hidden_dim=influence_hidden_dim, learning_rate=influence_lr)
        (obs_pred_state, state_pred_state, _, _, obs_target_params, state_target_params) = create_curiosity_predictor_state(rng=rng_cur, obs_dim=obs_dim, action_dim=action_dim, num_agents=num_agents, hidden_dim=curiosity_hidden_dim, learning_rate=curiosity_lr)
        (rng, _rng) = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config['NUM_ENVS'])
        (obsv, env_state) = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        ac_init_hstate = ScannedRNN.initialize_carry(config['NUM_ACTORS'], config['GRU_HIDDEN_DIM'])
        cr_init_hstate = ScannedRNN.initialize_carry(config['NUM_ACTORS'], config['GRU_HIDDEN_DIM'])
        coin_states = (influence_state, baseline_state, obs_pred_state, state_pred_state, inf_target_params, base_target_params, obs_target_params, state_target_params)

        def _update_step(update_runner_state, unused):
            (runner_state, update_steps, coin_states) = update_runner_state
            (influence_state, baseline_state, obs_pred_state, state_pred_state, inf_target_params, base_target_params, obs_target_params, state_target_params) = coin_states

            def _env_step(runner_state, unused):
                (train_states, env_state, last_obs, last_done, hstates, rng) = runner_state
                (rng, _rng) = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, config['NUM_ACTORS'])
                ac_in = (obs_batch[np.newaxis, :], last_done[np.newaxis, :])
                (ac_hstate, pi, action_logits) = actor_network.apply(train_states[0].params, hstates[0], ac_in)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                action_onehot = jax.nn.one_hot(action.squeeze(), action_dim)
                env_act = unbatchify(action, env.agents, config['NUM_ENVS'], num_agents)
                if is_smax:
                    env_act = {a: v.squeeze(-1) for (a, v) in env_act.items()}
                world_state = last_obs['world_state'].swapaxes(0, 1)
                world_state = world_state.reshape((config['NUM_ACTORS'], -1))
                cr_in = (world_state[None, :], last_done[np.newaxis, :])
                (cr_hstate, value) = critic_network.apply(train_states[1].params, hstates[1], cr_in)
                (rng, _rng) = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config['NUM_ENVS'])
                (obsv, env_state, reward, done, info) = jax.vmap(env.step, in_axes=(0, 0, 0))(rng_step, env_state, env_act)
                info = jax.tree.map(lambda x: x.reshape(config['NUM_ACTORS']), info)
                done_batch = batchify(done, env.agents, config['NUM_ACTORS']).squeeze()
                transition = Transition(jnp.tile(done['__all__'], num_agents), last_done, action.squeeze(), value.squeeze(), batchify(reward, env.agents, config['NUM_ACTORS']).squeeze(), log_prob.squeeze(), obs_batch, world_state, info, action_onehot)
                runner_state = (train_states, env_state, obsv, done_batch, (ac_hstate, cr_hstate), rng)
                return (runner_state, transition)
            initial_hstates = runner_state[-2]
            (runner_state, traj_batch) = jax.lax.scan(_env_step, runner_state, None, config['NUM_STEPS'])
            (train_states, env_state, last_obs, last_done, hstates, rng) = runner_state
            obs_reshaped = traj_batch.obs.reshape(config['NUM_STEPS'], num_agents, num_envs, obs_dim)
            actions_reshaped = traj_batch.action_onehot.reshape(config['NUM_STEPS'], num_agents, num_envs, action_dim)
            next_obs_batch = batchify(last_obs, env.agents, config['NUM_ACTORS'])
            next_obs_all = jnp.concatenate([traj_batch.obs[1:], next_obs_batch[None, :]], axis=0)
            next_obs_reshaped = next_obs_all.reshape(config['NUM_STEPS'], num_agents, num_envs, obs_dim)

            def compute_influence_per_step(t):
                obs_t = obs_reshaped[t]
                next_obs_t = next_obs_reshaped[t]
                actions_t = actions_reshaped[t]
                influence_rewards = compute_influence_reward(influence_state=influence_state, baseline_state=baseline_state, obs_all=obs_t, next_obs_all=next_obs_t, actions_all=actions_t, num_agents=num_agents, action_dim=action_dim, beta=influence_beta, sigma=influence_sigma, influence_target_params=inf_target_params, baseline_target_params=base_target_params)
                return influence_rewards

            def compute_curiosity_per_step(t):
                obs_t = obs_reshaped[t]
                next_obs_t = next_obs_reshaped[t]
                actions_t = actions_reshaped[t]
                curiosity_reward = compute_curiosity_reward(obs_predictor_state=obs_pred_state, state_predictor_state=state_pred_state, obs_all=obs_t, next_obs_all=next_obs_t, actions_all=actions_t, num_agents=num_agents, obs_weight=obs_weight, state_weight=0.0, obs_target_params=obs_target_params, state_target_params=state_target_params)
                return curiosity_reward
            timesteps = jnp.arange(config['NUM_STEPS'])
            influence_rewards = jax.vmap(compute_influence_per_step)(timesteps)
            curiosity_rewards = jax.vmap(compute_curiosity_per_step)(timesteps)
            influence_flat = influence_rewards.reshape(config['NUM_STEPS'], config['NUM_ACTORS'])
            curiosity_broadcast = jnp.broadcast_to(curiosity_rewards[:, None, :], (config['NUM_STEPS'], num_agents, num_envs)).reshape(config['NUM_STEPS'], config['NUM_ACTORS'])
            intrinsic_reward = influence_scale * influence_flat + curiosity_scale * curiosity_broadcast
            warmup_mask = (update_steps >= warmup_rollouts).astype(jnp.float32)
            intrinsic_reward = intrinsic_reward * warmup_mask
            intrinsic_reward = jnp.clip(intrinsic_reward, -10.0, 10.0)
            total_reward = traj_batch.reward + intrinsic_scale * intrinsic_reward
            last_world_state = last_obs['world_state'].swapaxes(0, 1)
            last_world_state = last_world_state.reshape((config['NUM_ACTORS'], -1))
            cr_in = (last_world_state[None, :], last_done[np.newaxis, :])
            (_, last_val) = critic_network.apply(train_states[1].params, hstates[1], cr_in)
            last_val = last_val.squeeze()
            traj_batch_with_intrinsic = Transition(global_done=traj_batch.global_done, done=traj_batch.done, action=traj_batch.action, value=traj_batch.value, reward=total_reward, log_prob=traj_batch.log_prob, obs=traj_batch.obs, world_state=traj_batch.world_state, info=traj_batch.info, action_onehot=traj_batch.action_onehot)

            def _calculate_gae(traj_batch, last_val):

                def _get_advantages(gae_and_next_value, transition):
                    (gae, next_value) = gae_and_next_value
                    (done, value, reward) = (transition.global_done, transition.value, transition.reward)
                    delta = reward + config['GAMMA'] * next_value * (1 - done) - value
                    gae = delta + config['GAMMA'] * config['GAE_LAMBDA'] * (1 - done) * gae
                    return ((gae, value), gae)
                (_, advantages) = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val), traj_batch, reverse=True, unroll=16)
                return (advantages, advantages + traj_batch.value)
            (advantages, targets) = _calculate_gae(traj_batch_with_intrinsic, last_val)

            def _update_epoch(update_state, unused):

                def _update_minbatch(train_states, batch_info):
                    (actor_train_state, critic_train_state) = train_states
                    (ac_init_hstate, cr_init_hstate, traj_batch, advantages, targets) = batch_info

                    def _actor_loss_fn(actor_params, init_hstate, traj_batch, gae):
                        (_, pi, _) = actor_network.apply(actor_params, init_hstate.squeeze(), (traj_batch.obs, traj_batch.done))
                        log_prob = pi.log_prob(traj_batch.action)
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-08)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, 1.0 - config['CLIP_EPS'], 1.0 + config['CLIP_EPS']) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()
                        approx_kl = (ratio - 1 - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config['CLIP_EPS'])
                        actor_loss = loss_actor - config['ENT_COEF'] * entropy
                        return (actor_loss, (loss_actor, entropy, ratio, approx_kl, clip_frac))

                    def _critic_loss_fn(critic_params, init_hstate, traj_batch, targets):
                        (_, value) = critic_network.apply(critic_params, init_hstate.squeeze(), (traj_batch.world_state, traj_batch.done))
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(-config['CLIP_EPS'], config['CLIP_EPS'])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        critic_loss = config['VF_COEF'] * value_loss
                        return (critic_loss, value_loss)
                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    (actor_loss, actor_grads) = actor_grad_fn(actor_train_state.params, ac_init_hstate, traj_batch, advantages)
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    (critic_loss, critic_grads) = critic_grad_fn(critic_train_state.params, cr_init_hstate, traj_batch, targets)
                    actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
                    critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)
                    total_loss = actor_loss[0] + critic_loss[0]
                    loss_info = {'total_loss': total_loss, 'actor_loss': actor_loss[0], 'value_loss': critic_loss[0], 'entropy': actor_loss[1][1], 'ratio': actor_loss[1][2], 'approx_kl': actor_loss[1][3], 'clip_frac': actor_loss[1][4]}
                    return ((actor_train_state, critic_train_state), loss_info)
                (train_states, init_hstates, traj_batch, advantages, targets, rng) = update_state
                (rng, _rng) = jax.random.split(rng)
                init_hstates = jax.tree.map(lambda x: jnp.reshape(x, (1, config['NUM_ACTORS'], -1)), init_hstates)
                batch = (init_hstates[0], init_hstates[1], traj_batch, advantages, targets)
                permutation = jax.random.permutation(_rng, config['NUM_ACTORS'])
                shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)
                minibatches = jax.tree.map(lambda x: jnp.swapaxes(jnp.reshape(x, (x.shape[0], config['NUM_MINIBATCHES'], -1) + x.shape[2:]), 0, 1), shuffled_batch)
                (train_states, loss_info) = jax.lax.scan(_update_minbatch, train_states, minibatches)
                update_state = (train_states, jax.tree.map(lambda x: x.squeeze(), init_hstates), traj_batch, advantages, targets, rng)
                return (update_state, loss_info)
            update_state = (train_states, initial_hstates, traj_batch_with_intrinsic, advantages, targets, rng)
            (update_state, loss_info) = jax.lax.scan(_update_epoch, update_state, None, config['UPDATE_EPOCHS'])
            train_states = update_state[0]
            obs_for_training = obs_reshaped.transpose(0, 2, 1, 3).reshape(-1, num_agents, obs_dim)
            next_obs_for_training = next_obs_reshaped.transpose(0, 2, 1, 3).reshape(-1, num_agents, obs_dim)
            actions_for_training = actions_reshaped.transpose(0, 2, 1, 3).reshape(-1, num_agents, action_dim)
            (new_influence_state, new_baseline_state, inf_losses) = update_influence_networks(influence_state=influence_state, baseline_state=baseline_state, obs_all=obs_for_training, next_obs_all=next_obs_for_training, actions_all=actions_for_training, num_agents=num_agents)
            (new_obs_pred_state, new_state_pred_state, cur_losses) = update_curiosity_networks(obs_predictor_state=obs_pred_state, state_predictor_state=state_pred_state, obs_all=obs_for_training, next_obs_all=next_obs_for_training, actions_all=actions_for_training, world_state=None, next_world_state=None, num_agents=num_agents)
            new_inf_target_params = soft_update_params(new_influence_state.params, inf_target_params, target_tau)
            new_base_target_params = soft_update_params(new_baseline_state.params, base_target_params, target_tau)
            new_obs_target_params = soft_update_params(new_obs_pred_state.params, obs_target_params, target_tau)
            new_state_target_params = soft_update_params(new_state_pred_state.params, state_target_params, target_tau)
            new_coin_states = (new_influence_state, new_baseline_state, new_obs_pred_state, new_state_pred_state, new_inf_target_params, new_base_target_params, new_obs_target_params, new_state_target_params)
            mean_influence = jnp.mean(influence_flat)
            mean_curiosity = jnp.mean(curiosity_broadcast)
            mean_extrinsic = jnp.mean(traj_batch.reward)
            metric = {'returns': traj_batch.info['returned_episode_returns'][-1, :].mean(), 'env_step': update_steps * config['NUM_ENVS'] * config['NUM_STEPS'], **jax.tree.map(lambda x: x.mean(), loss_info), 'reward/extrinsic': mean_extrinsic, 'reward/influence': mean_influence, 'reward/curiosity': mean_curiosity, 'reward/intrinsic_scaled': (mean_influence * influence_scale + mean_curiosity * curiosity_scale) * intrinsic_scale, 'coin/influence_loss': inf_losses['influence_loss'], 'coin/baseline_loss': inf_losses['baseline_loss'], 'coin/obs_predictor_loss': cur_losses['obs_predictor_loss']}

            def _print_progress(m):
                step = int(m['env_step'])
                returns = float(m['returns'])
                r_ext = float(m['reward/extrinsic'])
                r_inf = float(m['reward/influence'])
                r_cur = float(m['reward/curiosity'])
                loss = float(m['total_loss'])
                update = step // (config['NUM_ENVS'] * config['NUM_STEPS'])
                total_updates = int(config['NUM_UPDATES'])
                if update % 10 == 0 or update == total_updates:
                    print(f'[{update:4d}/{total_updates}] step={step:8d} | ret={returns:7.2f} | r_ext={r_ext:6.3f} r_inf={r_inf:6.3f} r_cur={r_cur:6.3f} | loss={loss:6.3f}')
            jax.debug.callback(_print_progress, metric)
            if config.get('WANDB_MODE') != 'disabled':
                jax.debug.callback(lambda m: wandb.log(m), metric)
            runner_state = (train_states, env_state, last_obs, last_done, hstates, rng)
            return ((runner_state, update_steps + 1, new_coin_states), metric)
        (rng, _rng) = jax.random.split(rng)
        runner_state = ((actor_train_state, critic_train_state), env_state, obsv, jnp.zeros(config['NUM_ACTORS'], dtype=bool), (ac_init_hstate, cr_init_hstate), _rng)
        (runner_state, metrics) = jax.lax.scan(_update_step, (runner_state, 0, coin_states), None, config['NUM_UPDATES'])
        return {'runner_state': runner_state, 'metrics': metrics}
    return train

def convert_wandb_args_to_hydra():
    new_argv = [sys.argv[0]]
    for arg in sys.argv[1:]:
        if arg.startswith('--') and '=' in arg:
            new_arg = arg[2:]
            new_argv.append(new_arg)
        else:
            new_argv.append(arg)
    sys.argv = new_argv
convert_wandb_args_to_hydra()

@hydra.main(version_base=None, config_path='config', config_name='mappo_coin_corridor')
def main(config):
    config = OmegaConf.to_container(config)
    print('=' * 80)
    print('COIN (CuriOsity + INfluence) Training')
    print('=' * 80)
    print(f"Environment: {config['ENV_NAME']}")
    print(f"Training Steps: {config['TOTAL_TIMESTEPS']:.0f}")
    coin_config = config.get('COIN', {})
    print(f"Influence Scale: {coin_config.get('INFLUENCE_SCALE', 1.0)}")
    print(f"Curiosity Scale: {coin_config.get('CURIOSITY_SCALE', 0.5)}")
    print(f"Intrinsic Reward Scale: {config.get('INTRINSIC_REWARD_SCALE', 1.0)}")
    print('=' * 80)
    print()
    wandb_project = config.get('PROJECT', 'JaxMARL')
    env_display_name = config['ENV_NAME'].replace('/', '-').replace('_', '-')
    run_name = config.get('RUN_NAME') or f'COIN-{env_display_name}'
    run = wandb.init(project=wandb_project, name=run_name, tags=['COIN', 'Influence', 'Curiosity', 'NeurIPS2023', config['ENV_NAME']], config=config, mode=config['WANDB_MODE'])
    rng = jax.random.PRNGKey(config['SEED'])
    print('Starting training...')
    try:
        with jax.disable_jit(False):
            train_jit = jax.jit(make_train(config))
            start_time = time()
            out = train_jit(rng)
            elapsed_time = time() - start_time
            print()
            print('=' * 80)
            print(f'Training completed in {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)')
            print('=' * 80)
    except KeyboardInterrupt:
        print('\n> Training interrupted by user')
        raise
    except Exception as e:
        print(f'\n> Training failed with error: {e}')
        import traceback
        traceback.print_exc()
        raise
    train_states = out['runner_state'][0][0]
    actor_params = train_states[0].params
    critic_params = train_states[1].params
    if config.get('WANDB_MODE') != 'disabled' and hasattr(run, 'dir'):
        save_dir = os.path.join(run.dir, 'params')
        os.makedirs(save_dir, exist_ok=True)
        save_file(flatten_dict(actor_params, sep=','), os.path.join(save_dir, 'actor_params.safetensors'))
        save_file(flatten_dict(critic_params, sep=','), os.path.join(save_dir, 'critic_params.safetensors'))
        config_path = os.path.join(save_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f'Checkpoints saved to {save_dir}')
if __name__ == '__main__':
    main()
