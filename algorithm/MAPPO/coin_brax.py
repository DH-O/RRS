import os
import sys
os.environ['JAX_DETERMINISTIC'] = '0'
import setup_jax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import distrax
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import hydra
from omegaconf import OmegaConf
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper
import wandb
from time import time
from typing import NamedTuple, Dict
from core import make_linear_schedule, stack_params, save_params, vectorized_apply_gradients
from coin_modules.influence import soft_update_params

class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: dict

class ActorContinuousFF(nn.Module):
    action_dim: int
    activation: str = 'tanh'
    fc_dim: int = 64

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == 'relu' else nn.tanh
        x = nn.Dense(self.fc_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Dense(self.fc_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        actor_logstd = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        actor_logstd = jnp.clip(actor_logstd, -20.0, 2.0)
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logstd))
        return pi

class CriticFF(nn.Module):
    activation: str = 'tanh'
    fc_dim: int = 64

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == 'relu' else nn.tanh
        x = nn.Dense(self.fc_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        x = nn.Dense(self.fc_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = activation(x)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        return jnp.squeeze(value, axis=-1)

class InfluenceEstimator(nn.Module):
    obs_dim: int
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return nn.Dense(self.obs_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)

class BaselinePredictor(nn.Module):
    obs_dim: int
    hidden_dim: int = 64

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        return nn.Dense(self.obs_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)

def batchify(x: dict, agent_list, num_actors):
    max_dim = max([x[a].shape[-1] for a in agent_list])

    def pad(z):
        pad_size = max_dim - z.shape[-1]
        if pad_size > 0:
            return jnp.concatenate([z, jnp.zeros(z.shape[:-1] + (pad_size,))], -1)
        return z
    x = jnp.stack([x[a] if x[a].shape[-1] == max_dim else pad(x[a]) for a in agent_list])
    return x.reshape((num_actors, -1))

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_agents, action_dims=None):
    x = x.reshape((num_agents, num_envs, -1))
    if action_dims is not None:
        return {a: x[i, :, :action_dims[i]] for (i, a) in enumerate(agent_list)}
    return {a: x[i] for (i, a) in enumerate(agent_list)}

def make_train(config, rng_init):
    env = jaxmarl.make(config['ENV_NAME'], **config.get('ENV_KWARGS', {}))
    num_agents = len(env.agents)
    config['NUM_ACTORS'] = num_agents * config['NUM_ENVS']
    config['NUM_UPDATES'] = int(config['TOTAL_TIMESTEPS']) // config['NUM_STEPS'] // config['NUM_ENVS']
    num_envs = config['NUM_ENVS']
    num_steps = config['NUM_STEPS']
    num_actors = config['NUM_ACTORS']
    num_updates = config['NUM_UPDATES']
    gamma = config['GAMMA']
    gae_lambda = config['GAE_LAMBDA']
    clip_eps = config['CLIP_EPS']
    ent_coef = config['ENT_COEF']
    anneal_ent = config.get('ANNEAL_ENT', False)
    ent_coef_end = config.get('ENT_COEF_END', ent_coef)
    vf_coef = config['VF_COEF']
    max_grad_norm = config['MAX_GRAD_NORM']
    update_epochs = config['UPDATE_EPOCHS']
    num_minibatches = config.get('NUM_MINIBATCHES', 1)
    fc_dim = config.get('FC_DIM_SIZE', 64)
    activation = config.get('ACTIVATION', 'tanh')
    coin_config = config.get('COIN', {})
    influence_hidden_dim = coin_config.get('INFLUENCE_HIDDEN_DIM', 64)
    influence_lr = coin_config.get('INFLUENCE_LR', 0.001)
    influence_scale = coin_config.get('INFLUENCE_SCALE', 1.0)
    influence_sigma = coin_config.get('INFLUENCE_SIGMA', 1.0)
    warmup_rollouts = coin_config.get('WARMUP_ROLLOUTS', 5)
    target_tau = coin_config.get('TARGET_TAU', 0.005)
    env = LogWrapper(env, replace_info=True)
    lr = config.get('LR', 0.001)

    def make_tx():
        return optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate=lr, eps=1e-05))
    per_agent_obs_dims = [env.observation_space(a).shape[0] for a in env.agents]
    per_agent_action_dims = [env.action_space(a).shape[0] for a in env.agents]
    obs_dim = max(per_agent_obs_dims)
    action_dim = max(per_agent_action_dims)
    (rng_init, _rng_actor, _rng_critic, _rng_coin) = jax.random.split(rng_init, 4)
    actor_network = ActorContinuousFF(action_dim=action_dim, activation=activation, fc_dim=fc_dim)
    init_x = jnp.zeros((obs_dim,))
    actor_rngs = jax.random.split(_rng_actor, num_agents)
    actor_train_states = []
    for i in range(num_agents):
        params = actor_network.init(actor_rngs[i], init_x)
        state = TrainState.create(apply_fn=actor_network.apply, params=params, tx=make_tx())
        actor_train_states.append(state)
    actor_train_states = tuple(actor_train_states)
    critic_network = CriticFF(activation=activation, fc_dim=fc_dim)
    critic_rngs = jax.random.split(_rng_critic, num_agents)
    critic_train_states = []
    for i in range(num_agents):
        params = critic_network.init(critic_rngs[i], init_x)
        state = TrainState.create(apply_fn=critic_network.apply, params=params, tx=make_tx())
        critic_train_states.append(state)
    critic_train_states = tuple(critic_train_states)
    (rng_inf, rng_base) = jax.random.split(_rng_coin, 2)
    influence_net = InfluenceEstimator(obs_dim=obs_dim, hidden_dim=influence_hidden_dim)
    influence_input_dim = obs_dim + action_dim + action_dim
    influence_params = influence_net.init(rng_inf, jnp.zeros((1, influence_input_dim)))
    influence_tx = optax.adam(influence_lr)
    influence_state = TrainState.create(apply_fn=influence_net.apply, params=influence_params, tx=influence_tx)
    baseline_net = BaselinePredictor(obs_dim=obs_dim, hidden_dim=influence_hidden_dim)
    baseline_input_dim = obs_dim + action_dim
    baseline_params = baseline_net.init(rng_base, jnp.zeros((1, baseline_input_dim)))
    baseline_tx = optax.adam(influence_lr)
    baseline_state = TrainState.create(apply_fn=baseline_net.apply, params=baseline_params, tx=baseline_tx)
    inf_target_params = jax.tree.map(jnp.copy, influence_state.params)
    base_target_params = jax.tree.map(jnp.copy, baseline_state.params)
    print(f'[COIN-BRAX] {num_agents} agents, obs={obs_dim}, act={action_dim}')
    print(f'[COIN-BRAX] Influence input: {influence_input_dim}, Baseline input: {baseline_input_dim}')
    print(f'[COIN-BRAX] influence_scale={influence_scale}, sigma={influence_sigma}')
    print(f'[COIN-BRAX] LR={lr}, updates={num_updates}')

    def train(rng):
        (rng, _rng) = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, num_envs)
        (obsv, env_state) = jax.vmap(env.reset)(reset_rng)

        def _wandb_log_callback(metrics):

            def safe_float(x):
                if hasattr(x, 'ndim'):
                    return float(x.item()) if x.ndim == 0 else float(x.mean())
                return float(x)
            log_dict = {}
            for (k, v) in metrics.items():
                if k in ('_timesteps', '_update_step'):
                    continue
                try:
                    log_dict[k] = safe_float(v)
                except Exception:
                    pass
            timesteps = int(safe_float(metrics['_timesteps']))
            update_step = int(safe_float(metrics['_update_step']))
            log_dict['timesteps'] = timesteps
            log_dict['update_step'] = update_step
            if 'env/returned_episode_returns' in log_dict:
                log_dict['returned_episode_returns'] = log_dict['env/returned_episode_returns']
            wandb.log(log_dict, step=update_step)
            if timesteps % (num_envs * num_steps * 100) < num_envs * num_steps + 1:
                ep_ret = log_dict.get('returned_episode_returns', log_dict.get('env/returned_episode_returns', 0.0))
                inf_r = log_dict.get('coin/influence_reward', 0.0)
                print(f'[Update] step={update_step:6d} ts={timesteps:10,d} | ep_return={ep_ret:.1f} | inf_reward={inf_r:.4f}')

        def _update_step(update_runner_state, unused):
            (runner_state, update_steps, inf_state, base_state, inf_tgt_params, base_tgt_params) = update_runner_state
            current_ent_coef = jax.lax.cond(anneal_ent, lambda : ent_coef - (ent_coef - ent_coef_end) * (update_steps / num_updates), lambda : jnp.float32(ent_coef))

            def _env_step(runner_state, unused):
                (train_states, env_state, last_obs, last_done, rng) = runner_state
                (actor_ts, critic_ts) = train_states
                (rng, _rng) = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, num_actors)
                obs_reshaped = obs_batch.reshape(num_agents, num_envs, -1)
                stacked_actor_params = stack_params([s.params for s in actor_ts])

                def _forward_actor(params, obs):
                    pi = actor_network.apply(params, obs)
                    return (pi.loc, pi.scale_diag)
                (action_mean, action_std) = jax.vmap(_forward_actor)(stacked_actor_params, obs_reshaped)
                action_mean_flat = action_mean.reshape(num_actors, -1)
                action_std_flat = action_std.reshape(num_actors, -1)
                pi = distrax.MultivariateNormalDiag(action_mean_flat, action_std_flat)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                stacked_critic_params = stack_params([s.params for s in critic_ts])

                def _forward_critic(params, obs):
                    return critic_network.apply(params, obs)
                value_per_agent = jax.vmap(_forward_critic)(stacked_critic_params, obs_reshaped)
                value = value_per_agent.reshape(num_actors)
                action_clipped = jnp.clip(action, -1.0, 1.0)
                env_act = unbatchify(action_clipped, env.agents, num_envs, num_agents, action_dims=per_agent_action_dims)
                (rng, _rng) = jax.random.split(rng)
                rng_step = jax.random.split(_rng, num_envs)
                (obsv, env_state, reward, done, info) = jax.vmap(env.step)(rng_step, env_state, env_act)
                info = jax.tree.map(lambda x: x.reshape(num_actors), info)
                done_batch = batchify(done, env.agents, num_actors).squeeze()
                reward_batch = batchify(reward, env.agents, num_actors).squeeze()
                transition = Transition(global_done=jnp.tile(done['__all__'], num_agents), done=last_done, action=action, value=value, reward=reward_batch, log_prob=log_prob, obs=obs_batch, info=info)
                runner_state = (train_states, env_state, obsv, done_batch, rng)
                return (runner_state, transition)
            (runner_state, traj_batch) = jax.lax.scan(_env_step, runner_state, None, num_steps)
            (train_states, env_state, last_obs, last_done, rng) = runner_state
            (actor_ts, critic_ts) = train_states
            last_obs_batch = batchify(last_obs, env.agents, num_actors)
            last_obs_reshaped = last_obs_batch.reshape(num_agents, num_envs, -1)
            stacked_critic_params = stack_params([s.params for s in critic_ts])

            def _get_last_val(params, obs):
                return critic_network.apply(params, obs)
            last_val_per_agent = jax.vmap(_get_last_val)(stacked_critic_params, last_obs_reshaped)
            last_val = last_val_per_agent.reshape(num_actors)
            obs_pa = traj_batch.obs.reshape(num_steps, num_agents, num_envs, obs_dim)
            act_pa = traj_batch.action.reshape(num_steps, num_agents, num_envs, action_dim)
            next_obs_batch = jnp.concatenate([traj_batch.obs[1:], last_obs_batch[None, :, :]], axis=0)
            next_obs_pa = next_obs_batch.reshape(num_steps, num_agents, num_envs, obs_dim)

            def _compute_influence_all_steps(obs_all, next_obs_all, act_all):

                def _influence_single_step(obs_step, next_obs_step, act_step):

                    def _influence_for_agent_i(i):
                        action_i = act_step[i]

                        def _influence_on_k(k):
                            obs_k = obs_step[k]
                            next_obs_k = next_obs_step[k]
                            action_k = act_step[k]
                            inf_input = jnp.concatenate([obs_k, action_k, action_i], axis=-1)
                            pred_inf = influence_net.apply(inf_tgt_params, inf_input)
                            error_inf = jnp.sum((next_obs_k - pred_inf) ** 2, axis=-1)
                            base_input = jnp.concatenate([obs_k, action_k], axis=-1)
                            pred_base = baseline_net.apply(base_tgt_params, base_input)
                            error_base = jnp.sum((next_obs_k - pred_base) ** 2, axis=-1)
                            diff = (error_base - error_inf) / (2 * influence_sigma ** 2)
                            is_self = i == k
                            return jnp.where(is_self, 0.0, diff)
                        k_indices = jnp.arange(num_agents)
                        influences = jax.vmap(_influence_on_k)(k_indices)
                        return jnp.sum(influences, axis=0)
                    i_indices = jnp.arange(num_agents)
                    return jax.vmap(_influence_for_agent_i)(i_indices)
                return jax.vmap(_influence_single_step)(obs_all, next_obs_all, act_all)
            influence_rewards = _compute_influence_all_steps(obs_pa, next_obs_pa, act_pa)
            influence_flat = influence_rewards.reshape(num_steps, num_actors)
            obs_flat = obs_pa.transpose(0, 2, 1, 3).reshape(-1, num_agents, obs_dim)
            next_obs_flat = next_obs_pa.transpose(0, 2, 1, 3).reshape(-1, num_agents, obs_dim)
            act_flat = act_pa.transpose(0, 2, 1, 3).reshape(-1, num_agents, action_dim)
            n_samples = obs_flat.shape[0]

            def _influence_loss_fn(inf_params):
                total_loss = jnp.float32(0.0)

                def _pair_loss(i, k):
                    obs_k = obs_flat[:, k]
                    next_obs_k = next_obs_flat[:, k]
                    action_k = act_flat[:, k]
                    action_i = act_flat[:, i]
                    inf_input = jnp.concatenate([obs_k, action_k, action_i], axis=-1)
                    pred = influence_net.apply(inf_params, inf_input)
                    return jnp.mean((pred - next_obs_k) ** 2)

                def _agent_i_loss(i):

                    def _agent_k_loss(k):
                        loss = _pair_loss(i, k)
                        mask = (i != k).astype(jnp.float32)
                        return loss * mask
                    k_losses = jax.vmap(_agent_k_loss)(jnp.arange(num_agents))
                    return jnp.sum(k_losses) / (num_agents - 1)
                losses = jax.vmap(_agent_i_loss)(jnp.arange(num_agents))
                return jnp.mean(losses)

            def _baseline_loss_fn(base_params):

                def _agent_k_loss(k):
                    obs_k = obs_flat[:, k]
                    next_obs_k = next_obs_flat[:, k]
                    action_k = act_flat[:, k]
                    base_input = jnp.concatenate([obs_k, action_k], axis=-1)
                    pred = baseline_net.apply(base_params, base_input)
                    return jnp.mean((pred - next_obs_k) ** 2)
                losses = jax.vmap(_agent_k_loss)(jnp.arange(num_agents))
                return jnp.mean(losses)
            (inf_loss, inf_grads) = jax.value_and_grad(_influence_loss_fn)(inf_state.params)
            new_inf_state = inf_state.apply_gradients(grads=inf_grads)
            (base_loss, base_grads) = jax.value_and_grad(_baseline_loss_fn)(base_state.params)
            new_base_state = base_state.apply_gradients(grads=base_grads)
            use_intrinsic = update_steps >= warmup_rollouts
            r_ext = traj_batch.reward.astype(jnp.float32)
            r_combined = jax.lax.cond(use_intrinsic, lambda : r_ext + influence_scale * influence_flat, lambda : r_ext)

            def _calculate_gae(traj, last_v, rewards):

                def _get_advantages(gae_and_nv, inputs):
                    (gae, nv) = gae_and_nv
                    (done, value, reward) = inputs
                    delta = reward + gamma * nv * (1 - done) - value
                    gae = delta + gamma * gae_lambda * (1 - done) * gae
                    return ((gae, value), gae)
                (_, advantages) = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_v), last_v), (traj.global_done, traj.value, rewards), reverse=True, unroll=16)
                return (advantages, advantages + traj.value)
            (advantages, targets) = _calculate_gae(traj_batch, last_val, r_combined)

            def _update_epoch(update_state, unused):
                (train_states_upd, traj_batch_upd, advantages_upd, targets_upd, rng_upd) = update_state
                (rng_upd, _rng) = jax.random.split(rng_upd)
                (actor_ts_upd, critic_ts_upd) = train_states_upd

                def to_per_agent_3d(x):
                    return x.reshape(num_steps, num_agents, num_envs, -1).transpose(1, 0, 2, 3)

                def to_per_agent_2d(x):
                    return x.reshape(num_steps, num_agents, num_envs).transpose(1, 0, 2)
                obs_pa = to_per_agent_3d(traj_batch_upd.obs)
                action_pa = to_per_agent_3d(traj_batch_upd.action)
                log_prob_pa = to_per_agent_2d(traj_batch_upd.log_prob)
                adv_pa = to_per_agent_2d(advantages_upd)
                value_pa = to_per_agent_2d(traj_batch_upd.value)
                targets_pa = to_per_agent_2d(targets_upd)
                batch_size_pa = num_steps * num_envs
                obs_flat = obs_pa.reshape(num_agents, batch_size_pa, -1)
                action_flat = action_pa.reshape(num_agents, batch_size_pa, -1)
                log_prob_flat = log_prob_pa.reshape(num_agents, batch_size_pa)
                adv_flat = adv_pa.reshape(num_agents, batch_size_pa)
                value_flat = value_pa.reshape(num_agents, batch_size_pa)
                targets_flat = targets_pa.reshape(num_agents, batch_size_pa)
                perm = jax.random.permutation(_rng, batch_size_pa)

                def shuffle_split(x):
                    x_shuf = jnp.take(x, perm, axis=1)
                    return x_shuf.reshape(num_agents, num_minibatches, -1, *x.shape[2:])
                obs_mb = shuffle_split(obs_flat).transpose(1, 0, 2, 3)
                action_mb = shuffle_split(action_flat).transpose(1, 0, 2, 3)
                log_prob_mb = shuffle_split(log_prob_flat).transpose(1, 0, 2)
                adv_mb = shuffle_split(adv_flat).transpose(1, 0, 2)
                value_mb = shuffle_split(value_flat).transpose(1, 0, 2)
                targets_mb = shuffle_split(targets_flat).transpose(1, 0, 2)
                minibatches = (obs_mb, action_mb, log_prob_mb, adv_mb, value_mb, targets_mb)

                def _update_minibatch(carry, mb_data):
                    (actor_st_mb, critic_st_mb) = carry
                    (obs_m, action_m, log_prob_m, adv_m, value_m, targets_m) = mb_data
                    stacked_actor_p = stack_params([s.params for s in actor_st_mb])
                    stacked_critic_p = stack_params([s.params for s in critic_st_mb])

                    def _actor_loss(actor_p, obs, action, log_prob_old, gae):
                        pi = actor_network.apply(actor_p, obs)
                        new_log_prob = pi.log_prob(action)
                        entropy = pi.entropy()
                        logratio = new_log_prob - log_prob_old
                        ratio = jnp.exp(logratio)
                        gae_norm = (gae - gae.mean()) / (gae.std() + 1e-08)
                        loss1 = ratio * gae_norm
                        loss2 = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * gae_norm
                        loss_actor = -jnp.minimum(loss1, loss2).mean()
                        entropy_mean = entropy.mean()
                        return (loss_actor - current_ent_coef * entropy_mean, (loss_actor, entropy_mean))

                    def _compute_actor_grad(p, o, a, lp, g):
                        ((loss, aux), grad) = jax.value_and_grad(_actor_loss, has_aux=True)(p, o, a, lp, g)
                        return (loss, aux, grad)
                    (a_losses, a_aux, a_grads) = jax.vmap(_compute_actor_grad)(stacked_actor_p, obs_m, action_m, log_prob_m, adv_m)
                    new_actors = vectorized_apply_gradients(list(actor_st_mb), a_grads)

                    def _critic_loss(critic_p, obs, value_old, tgt):
                        value = critic_network.apply(critic_p, obs)
                        v_clipped = value_old + (value - value_old).clip(-clip_eps, clip_eps)
                        v_losses = jnp.square(value - tgt)
                        v_losses_clipped = jnp.square(v_clipped - tgt)
                        v_loss = 0.5 * jnp.maximum(v_losses, v_losses_clipped).mean()
                        return (vf_coef * v_loss, v_loss)

                    def _compute_critic_grad(p, o, v_old, tgt):
                        ((loss, raw), grad) = jax.value_and_grad(_critic_loss, has_aux=True)(p, o, v_old, tgt)
                        return (loss, raw, grad)
                    (c_losses, c_raw, c_grads) = jax.vmap(_compute_critic_grad)(stacked_critic_p, obs_m, value_m, targets_m)
                    new_critics = vectorized_apply_gradients(list(critic_st_mb), c_grads)
                    mb_metrics = {'loss/actor': a_aux[0].mean(), 'loss/critic': c_raw.mean(), 'loss/entropy': a_aux[1].mean()}
                    return ((tuple(new_actors), tuple(new_critics)), mb_metrics)
                init_carry = (actor_ts_upd, critic_ts_upd)
                ((new_act, new_crit), mb_metrics) = jax.lax.scan(_update_minibatch, init_carry, minibatches)
                update_metrics = jax.tree.map(lambda x: x.mean(), mb_metrics)
                train_states_out = (new_act, new_crit)
                return ((train_states_out, traj_batch_upd, advantages_upd, targets_upd, rng_upd), update_metrics)
            (rng, _rng_update) = jax.random.split(rng)
            upd_train_states = (actor_ts, critic_ts)
            init_state = (upd_train_states, traj_batch, advantages, targets, _rng_update)
            (update_state_out, update_metrics) = jax.lax.scan(_update_epoch, init_state, None, update_epochs)
            ((actor_ts, critic_ts), _, _, _, _) = update_state_out
            env_info = jax.tree.map(lambda x: x.mean(), traj_batch.info)
            update_metrics_agg = jax.tree.map(lambda x: x.mean(), update_metrics)
            metrics = {**update_metrics_agg, 'env/reward_mean': traj_batch.reward.mean(), 'env/reward_std': traj_batch.reward.std(), 'coin/influence_reward': jnp.mean(influence_flat), 'coin/influence_loss': inf_loss, 'coin/baseline_loss': base_loss}
            for (k, v) in env_info.items():
                metrics[f'env/{k}'] = v
            timesteps = (update_steps + 1) * num_envs * num_steps
            callback_metrics = {k: v for (k, v) in metrics.items()}
            callback_metrics['_timesteps'] = timesteps
            callback_metrics['_update_step'] = update_steps + 1
            jax.experimental.io_callback(_wandb_log_callback, None, callback_metrics, ordered=True)
            new_inf_tgt_params = soft_update_params(new_inf_state.params, inf_tgt_params, target_tau)
            new_base_tgt_params = soft_update_params(new_base_state.params, base_tgt_params, target_tau)
            train_states = (actor_ts, critic_ts)
            runner_state = (train_states, env_state, last_obs, last_done, rng)
            return ((runner_state, update_steps + 1, new_inf_state, new_base_state, new_inf_tgt_params, new_base_tgt_params), metrics)
        init_done = jnp.zeros((num_actors,), dtype=bool)
        train_states = (actor_train_states, critic_train_states)
        runner_state = (train_states, env_state, obsv, init_done, rng)
        update_runner_state = (runner_state, jnp.array(0), influence_state, baseline_state, inf_target_params, base_target_params)
        (update_runner_state, metrics_all) = jax.lax.scan(_update_step, update_runner_state, None, num_updates)
        return {'metrics': metrics_all}
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

@hydra.main(version_base=None, config_path='config', config_name='coin_brax_ant')
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    coin_config = config.get('COIN', {})
    print('\n' + '=' * 80)
    print('COIN IPPO for MABrax')
    print('=' * 80)
    print(f"Environment: {config['ENV_NAME']}")
    print(f"Seed: {config['SEED']}")
    print(f"Total Timesteps: {config['TOTAL_TIMESTEPS']:,.0f}")
    print(f"Influence Scale: {coin_config.get('INFLUENCE_SCALE', 1.0)}")
    print('=' * 80 + '\n')
    rng = jax.random.PRNGKey(config['SEED'])
    wandb.init(project=config.get('PROJECT', 'coin-mabrax'), name=config.get('RUN_NAME', f"coin-{config['ENV_NAME']}"), tags=['coin', 'mabrax', config['ENV_NAME']], config=config, mode=config.get('WANDB_MODE', 'online'))
    (rng, _rng) = jax.random.split(rng)
    train_fn = make_train(config, _rng)
    if not config.get('DISABLE_JIT', False):
        device_idx = config.get('DEVICE', 0)
        devices = jax.devices()
        device = devices[device_idx] if device_idx < len(devices) else devices[0]
        train_fn = jax.jit(train_fn, device=device)
    t_start = time()
    result = train_fn(rng)
    jax.block_until_ready(result)
    t_end = time()
    print(f'\n[COIN-BRAX] Training completed in {t_end - t_start:.1f}s')
    final_return = float(result['metrics']['env/returned_episode_returns'][-1])
    print(f'[COIN-BRAX] Final return: {final_return:.3f}')
    wandb.finish()
if __name__ == '__main__':
    main()
