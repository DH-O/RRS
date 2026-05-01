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
from core import create_tdd_train_states, make_linear_schedule, discounted_sampling, stack_params, save_params, compute_intrinsic_reward_sd, mrn_distance, vectorized_apply_gradients

class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: dict
    intrinsic_reward: jnp.ndarray = None
    agent_features: jnp.ndarray = None
    agent_idx: jnp.ndarray = None

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

class TrainerConfig(NamedTuple):
    num_envs: int
    num_steps: int
    num_agents: int
    num_actors: int
    gamma: float
    gae_lambda: float
    clip_eps: float
    ent_coef: float
    anneal_ent: bool
    ent_coef_end: float
    vf_coef: float
    max_grad_norm: float
    num_updates: int
    update_epochs: int
    num_minibatches: int
    seed: int
    fc_dim: int
    activation: str
    tdd_input_dim: int
    tdd_obs_slice_start: int
    tdd_obs_slice_end: int
    tdd_latent_dim: int
    tdd_output_dim: int
    tdd_lr: float
    tdd_batch_size: int
    tdd_update_epochs: int
    tdd_discount: float
    intrinsic_reward_scale: float
    warmup_rollouts: int
    rcb_beta_max: float
    rcb_beta_min: float
    rcb_kappa: float
    rcb_target: float
    rcb_ema_alpha: float
    rsq_lambda: float
    rsq_ref: float
    rsq_h_min: float
    rsq_h_max: float
    rsq_ema_alpha: float
    rsq_use_advantage: bool = False
    tdd_augment: str = 'none'
    use_lagrangian: bool = False
    lagrangian_lr: float = 0.005
    lagrangian_beta_init: float = 0.3

def create_trainer_config(config: Dict, num_agents: int) -> TrainerConfig:
    tdd_obs_slice_end = config.get('TDD_OBS_SLICE_END')
    if tdd_obs_slice_end is None:
        tdd_obs_slice_end = 0
    return TrainerConfig(num_envs=config['NUM_ENVS'], num_steps=config['NUM_STEPS'], num_agents=num_agents, num_actors=config['NUM_ACTORS'], gamma=config['GAMMA'], gae_lambda=config['GAE_LAMBDA'], clip_eps=config['CLIP_EPS'], ent_coef=config['ENT_COEF'], anneal_ent=config.get('ANNEAL_ENT', False), ent_coef_end=config.get('ENT_COEF_END', config['ENT_COEF']), vf_coef=config['VF_COEF'], max_grad_norm=config['MAX_GRAD_NORM'], num_updates=config['NUM_UPDATES'], update_epochs=config['UPDATE_EPOCHS'], num_minibatches=config.get('NUM_MINIBATCHES', 1), seed=config['SEED'], fc_dim=config.get('FC_DIM_SIZE', 64), activation=config.get('ACTIVATION', 'tanh'), tdd_input_dim=config.get('TDD_INPUT_DIM', 2), tdd_obs_slice_start=config.get('TDD_OBS_SLICE_START', -2), tdd_obs_slice_end=tdd_obs_slice_end, tdd_latent_dim=config.get('TDD_LATENT_DIM', 64), tdd_output_dim=config.get('TDD_OUTPUT_DIM', 32), tdd_lr=config.get('TDD_LR', 0.001), tdd_batch_size=config.get('TDD_BATCH_SIZE', 512), tdd_update_epochs=config.get('TDD_UPDATE_EPOCHS', 25), tdd_discount=config.get('TDD_DISCOUNT', 0.99), intrinsic_reward_scale=config.get('INTRINSIC_REWARD_SCALE', 1.0), warmup_rollouts=config.get('WARMUP_ROLLOUTS', 0), rcb_beta_max=config.get('RCB_BETA_MAX', 0.5), rcb_beta_min=config.get('RCB_BETA_MIN', 0.05), rcb_kappa=config.get('RCB_KAPPA', 0.015), rcb_target=config.get('RCB_TARGET', 400.0), rcb_ema_alpha=config.get('RCB_EMA_ALPHA', 0.03), rsq_lambda=config.get('RSQ_LAMBDA', 2.0), rsq_ref=config.get('RSQ_REF', 0.5), rsq_h_min=config.get('RSQ_H_MIN', 0.1), rsq_h_max=config.get('RSQ_H_MAX', 2.0), rsq_ema_alpha=config.get('RSQ_EMA_ALPHA', 0.1), rsq_use_advantage=config.get('RSQ_USE_ADVANTAGE', False), tdd_augment=config.get('TDD_AUGMENT', 'none'), use_lagrangian=config.get('USE_LAGRANGIAN', False), lagrangian_lr=config.get('LAGRANGIAN_LR', 0.005), lagrangian_beta_init=config.get('LAGRANGIAN_BETA_INIT', 0.3))

def print_training_header(config: Dict):
    if config.get('USE_LAGRANGIAN', False):
        mode = 'Lagrangian'
    else:
        mode = 'RCB-RSQ'
    print('\n' + '=' * 80)
    print(f'{mode} IPPO for MABrax')
    print('=' * 80)
    print(f"Environment: {config['ENV_NAME']}")
    print(f"Seed: {config['SEED']}")
    print(f"Total Timesteps: {config['TOTAL_TIMESTEPS']:,.0f}")
    print(f"RCB: beta_max={config.get('RCB_BETA_MAX')}, beta_min={config.get('RCB_BETA_MIN')}, kappa={config.get('RCB_KAPPA')}, target={config.get('RCB_TARGET')}")
    rsq_src = 'advantage' if config.get('RSQ_USE_ADVANTAGE', False) else 'intrinsic'
    print(f"RSQ: lambda={config.get('RSQ_LAMBDA')}, ref={config.get('RSQ_REF')}, h=[{config.get('RSQ_H_MIN')}, {config.get('RSQ_H_MAX')}], source={rsq_src}")
    print('=' * 80 + '\n')

def make_train(config, rng_init):
    env = jaxmarl.make(config['ENV_NAME'], **config.get('ENV_KWARGS', {}))
    num_agents = len(env.agents)
    config['NUM_ACTORS'] = num_agents * config['NUM_ENVS']
    config['NUM_UPDATES'] = int(config['TOTAL_TIMESTEPS']) // config['NUM_STEPS'] // config['NUM_ENVS']
    tc = create_trainer_config(config, num_agents)
    env = LogWrapper(env, replace_info=True)
    lr = config.get('LR', 0.001)

    def make_tx():
        return optax.chain(optax.clip_by_global_norm(tc.max_grad_norm), optax.adam(learning_rate=lr, eps=1e-05))
    per_agent_obs_dims = [env.observation_space(a).shape[0] for a in env.agents]
    per_agent_action_dims = [env.action_space(a).shape[0] for a in env.agents]
    obs_dim = max(per_agent_obs_dims)
    action_dim = max(per_agent_action_dims)
    (rng_init, _rng_actor, _rng_critic, _rng_tdd) = jax.random.split(rng_init, 4)
    actor_network = ActorContinuousFF(action_dim=action_dim, activation=tc.activation, fc_dim=tc.fc_dim)
    init_x = jnp.zeros((obs_dim,))
    actor_rngs = jax.random.split(_rng_actor, tc.num_agents)
    actor_train_states = []
    for i in range(tc.num_agents):
        params = actor_network.init(actor_rngs[i], init_x)
        state = TrainState.create(apply_fn=actor_network.apply, params=params, tx=make_tx())
        actor_train_states.append(state)
    actor_train_states = tuple(actor_train_states)
    critic_network = CriticFF(activation=tc.activation, fc_dim=tc.fc_dim)
    critic_rngs = jax.random.split(_rng_critic, tc.num_agents)
    critic_train_states = []
    for i in range(tc.num_agents):
        params = critic_network.init(critic_rngs[i], init_x)
        state = TrainState.create(apply_fn=critic_network.apply, params=params, tx=make_tx())
        critic_train_states.append(state)
    critic_train_states = tuple(critic_train_states)
    effective_tdd_input_dim = tc.tdd_input_dim
    if tc.tdd_augment == 'action':
        effective_tdd_input_dim = tc.tdd_input_dim + action_dim
    elif tc.tdd_augment == 'agent_id':
        effective_tdd_input_dim = tc.tdd_input_dim + num_agents
    elif tc.tdd_augment == 'action_id':
        effective_tdd_input_dim = tc.tdd_input_dim + action_dim + num_agents
    tc = tc._replace(tdd_input_dim=effective_tdd_input_dim)
    (tdd_potential_states, tdd_s_encoder_states) = create_tdd_train_states(rng=_rng_tdd, config=config, num_agents=tc.num_agents, input_dim=tc.tdd_input_dim)
    tdd_potential_states = tuple(tdd_potential_states)
    tdd_s_encoder_states = tuple(tdd_s_encoder_states)
    print(f'[RCB-RSQ-BRAX] {tc.num_agents} agents, obs={per_agent_obs_dims} (pad→{obs_dim}), act={per_agent_action_dims} (pad→{action_dim})')
    print(f'[RCB-RSQ-BRAX] RCB: beta=[{tc.rcb_beta_min}, {tc.rcb_beta_max}], kappa={tc.rcb_kappa}, target={tc.rcb_target}')
    print(f'[RCB-RSQ-BRAX] RSQ: lambda={tc.rsq_lambda}, ref={tc.rsq_ref}, h=[{tc.rsq_h_min}, {tc.rsq_h_max}], use_adv={tc.rsq_use_advantage}')
    print(f'[RCB-RSQ-BRAX] TDD: input_dim={tc.tdd_input_dim}, augment={tc.tdd_augment}, obs_slice=[{tc.tdd_obs_slice_start}:{tc.tdd_obs_slice_end}]')
    print(f'[RCB-RSQ-BRAX] LR={lr}, updates={tc.num_updates}')

    def train(rng):
        (rng, _rng) = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, tc.num_envs)
        (obsv, env_state) = jax.vmap(env.reset)(reset_rng)
        agent_idx_array = jnp.repeat(jnp.arange(tc.num_agents), tc.num_envs)

        def extract_tdd_features(obs_batch, action_batch=None):
            if tc.tdd_obs_slice_end == 0:
                feats = obs_batch[:, tc.tdd_obs_slice_start:]
            else:
                feats = obs_batch[:, tc.tdd_obs_slice_start:tc.tdd_obs_slice_end]
            aug = tc.tdd_augment
            if aug == 'action' and action_batch is not None:
                feats = jnp.concatenate([feats, action_batch], axis=-1)
            elif aug == 'agent_id':
                agent_onehot = jax.nn.one_hot(agent_idx_array, tc.num_agents)
                feats = jnp.concatenate([feats, agent_onehot], axis=-1)
            elif aug == 'action_id' and action_batch is not None:
                agent_onehot = jax.nn.one_hot(agent_idx_array, tc.num_agents)
                feats = jnp.concatenate([feats, action_batch, agent_onehot], axis=-1)
            return feats
        ir_mu_init = jnp.zeros(tc.num_agents)
        ir_var_init = jnp.ones(tc.num_agents)
        adv_mu_init = jnp.zeros(tc.num_agents)
        adv_var_init = jnp.ones(tc.num_agents)
        lagrangian_beta_init = jnp.float32(tc.lagrangian_beta_init)

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
            if timesteps % (tc.num_envs * tc.num_steps * 100) < tc.num_envs * tc.num_steps + 1:
                ep_ret = log_dict.get('returned_episode_returns', log_dict.get('env/returned_episode_returns', 0.0))
                beta = log_dict.get('rcb/adaptive_beta', 0.0)
                rsq = log_dict.get('rsq/rsq_mean', 0.0)
                tdd = log_dict.get('tdd/loss', 0.0)
                x_pos = log_dict.get('env/x_position', 0.0)
                x_vel = log_dict.get('env/x_velocity', 0.0)
                print(f'[Update] step={update_step:6d} ts={timesteps:10,d} | ep_return={ep_ret:.1f} | beta={beta:.3f} | rsq={rsq:.3f} | tdd={tdd:.4f} | x_pos={x_pos:.3f} x_vel={x_vel:.3f}')

        def _update_step(update_runner_state, unused):
            (runner_state, update_steps, return_ema, ir_mu, ir_var, adv_mu, adv_var, lagrangian_beta) = update_runner_state
            current_ent_coef = jax.lax.cond(tc.anneal_ent, lambda : tc.ent_coef - (tc.ent_coef - tc.ent_coef_end) * (update_steps / tc.num_updates), lambda : jnp.float32(tc.ent_coef))

            def _env_step(runner_state, unused):
                (train_states, env_state, last_obs, last_done, rng) = runner_state
                (actor_ts, critic_ts, _, _) = train_states
                (rng, _rng) = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, tc.num_actors)
                obs_reshaped = obs_batch.reshape(tc.num_agents, tc.num_envs, -1)
                stacked_actor_params = stack_params([s.params for s in actor_ts])

                def _forward_actor(params, obs):
                    pi = actor_network.apply(params, obs)
                    return (pi.loc, pi.scale_diag)
                (action_mean, action_std) = jax.vmap(_forward_actor)(stacked_actor_params, obs_reshaped)
                action_mean = action_mean.reshape(tc.num_actors, -1)
                action_std = action_std.reshape(tc.num_actors, -1)
                pi = distrax.MultivariateNormalDiag(action_mean, action_std)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                stacked_critic_params = stack_params([s.params for s in critic_ts])

                def _forward_critic(params, obs):
                    return critic_network.apply(params, obs)
                value_per_agent = jax.vmap(_forward_critic)(stacked_critic_params, obs_reshaped)
                value = value_per_agent.reshape(tc.num_actors)
                action_clipped = jnp.clip(action, -1.0, 1.0)
                env_act = unbatchify(action_clipped, env.agents, tc.num_envs, tc.num_agents, action_dims=per_agent_action_dims)
                (rng, _rng) = jax.random.split(rng)
                rng_step = jax.random.split(_rng, tc.num_envs)
                (obsv, env_state, reward, done, info) = jax.vmap(env.step)(rng_step, env_state, env_act)
                info = jax.tree.map(lambda x: x.reshape(tc.num_actors), info)
                done_batch = batchify(done, env.agents, tc.num_actors).squeeze()
                agent_features = extract_tdd_features(obs_batch, action_clipped)
                reward_batch = batchify(reward, env.agents, tc.num_actors).squeeze()
                transition = Transition(global_done=jnp.tile(done['__all__'], tc.num_agents), done=last_done, action=action, value=value, reward=reward_batch, log_prob=log_prob, obs=obs_batch, info=info, intrinsic_reward=None, agent_features=agent_features, agent_idx=agent_idx_array)
                runner_state = (train_states, env_state, obsv, done_batch, rng)
                return (runner_state, transition)
            (runner_state, traj_batch) = jax.lax.scan(_env_step, runner_state, None, tc.num_steps)
            (train_states, env_state, last_obs, last_done, rng) = runner_state
            (actor_train_states_cur, critic_train_states_cur, tdd_pot_states, tdd_enc_states) = train_states
            last_obs_batch = batchify(last_obs, env.agents, tc.num_actors)
            last_obs_reshaped = last_obs_batch.reshape(tc.num_agents, tc.num_envs, -1)
            stacked_critic_params = stack_params([s.params for s in critic_train_states_cur])

            def _get_last_val(params, obs):
                return critic_network.apply(params, obs)
            last_val_per_agent = jax.vmap(_get_last_val)(stacked_critic_params, last_obs_reshaped)
            last_val = last_val_per_agent.reshape(tc.num_actors)
            agent_features_all = traj_batch.agent_features.reshape(tc.num_steps, tc.num_agents, tc.num_envs, tc.tdd_input_dim)
            agent_features_curr_all = agent_features_all.transpose(1, 0, 2, 3)
            agent_features_next_all = jnp.concatenate([agent_features_curr_all[:, 1:, :, :], agent_features_curr_all[:, -1:, :, :]], axis=1)
            (rng, _rng_tdd) = jax.random.split(rng)

            def _tdd_update_step(tdd_states_arg, rng_arg):
                (potential_states_inner, s_encoder_states_inner) = tdd_states_arg
                batch_size = tc.tdd_batch_size
                I = jnp.eye(batch_size)
                agent_rngs = jax.random.split(rng_arg, tc.num_agents)
                potential_apply_fn = potential_states_inner[0].apply_fn
                s_encoder_apply_fn = s_encoder_states_inner[0].apply_fn
                stacked_pot_params = stack_params([s.params for s in potential_states_inner])
                stacked_enc_params = stack_params([s.params for s in s_encoder_states_inner])

                def _tdd_loss_single_agent(pot_params, enc_params, agent_features, agent_rng):
                    (rng_t, rng_env, rng_goal) = jax.random.split(agent_rng, 3)
                    t_indices = jax.random.randint(rng_t, (batch_size,), 0, tc.num_steps - 1)
                    env_indices = jax.random.randint(rng_env, (batch_size,), 0, tc.num_envs)
                    obs = agent_features[t_indices, env_indices, :]
                    ranges = (tc.num_steps - 1 - t_indices).astype(jnp.int32)
                    ranges = jnp.maximum(ranges, 1)
                    intervals = discounted_sampling(ranges, tc.tdd_discount, rng_goal)
                    goal_t = t_indices + intervals + 1
                    goal_t = jnp.clip(goal_t, 0, tc.num_steps - 1)
                    goal = agent_features[goal_t, env_indices, :]
                    valid_mask = goal_t > t_indices
                    valid_mask_float = valid_mask.astype(jnp.float32)
                    num_valid = jnp.sum(valid_mask_float)
                    c_g = potential_apply_fn(pot_params, goal)
                    phi_s = s_encoder_apply_fn(enc_params, obs)
                    phi_g = s_encoder_apply_fn(enc_params, goal)
                    mrn_dists = mrn_distance(phi_s[:, None, :], phi_g[None, :, :])
                    logits = c_g.squeeze(-1)[None, :] - mrn_dists
                    log_sm = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
                    loss1 = jnp.sum(-(log_sm * I).sum(axis=-1) * valid_mask_float) / jnp.maximum(num_valid, 1)
                    log_sm_T = logits.T - jax.scipy.special.logsumexp(logits.T, axis=-1, keepdims=True)
                    loss2 = jnp.sum(-(log_sm_T * I).sum(axis=-1) * valid_mask_float) / jnp.maximum(num_valid, 1)
                    return (loss1 + loss2) / 2

                def _compute_loss_and_grad(pp, ep, feat, rng):
                    (loss, (pg, eg)) = jax.value_and_grad(_tdd_loss_single_agent, argnums=(0, 1))(pp, ep, feat, rng)
                    return (loss, pg, eg)
                (losses, pot_grads, enc_grads) = jax.vmap(_compute_loss_and_grad)(stacked_pot_params, stacked_enc_params, agent_features_curr_all, agent_rngs)
                new_pot = vectorized_apply_gradients(list(potential_states_inner), pot_grads)
                new_enc = vectorized_apply_gradients(list(s_encoder_states_inner), enc_grads)
                return ((tuple(new_pot), tuple(new_enc)), losses.mean())

            def _tdd_update_epoch(carry, unused):
                (tdd_st, rng_inner) = carry
                (rng_inner, _rng_step) = jax.random.split(rng_inner)
                (new_tdd_st, loss) = _tdd_update_step(tdd_st, _rng_step)
                return ((new_tdd_st, rng_inner), loss)
            ((new_tdd_states, _), tdd_losses) = jax.lax.scan(_tdd_update_epoch, ((tdd_pot_states, tdd_enc_states), _rng_tdd), None, tc.tdd_update_epochs)
            tdd_loss = tdd_losses.mean()
            tdd_pot_states = tuple(new_tdd_states[0])
            tdd_enc_states = tuple(new_tdd_states[1])
            intrinsic_reward_per_agent = compute_intrinsic_reward_sd(agent_features_curr_all, agent_features_next_all, tdd_enc_states, tdd_shared=False, num_agents=tc.num_agents, num_envs=tc.num_envs, num_steps=tc.num_steps)
            intrinsic_reward_all = intrinsic_reward_per_agent.transpose(1, 0, 2).reshape(tc.num_steps, tc.num_actors)
            intrinsic_reward_raw = intrinsic_reward_all
            r_int_scaled = intrinsic_reward_all * tc.intrinsic_reward_scale
            ep_returns = traj_batch.info['returned_episode_returns']
            ep_dones = traj_batch.info['returned_episode']
            mean_return = jnp.where(ep_dones.any(), jnp.sum(ep_returns * ep_dones) / jnp.maximum(ep_dones.sum(), 1), return_ema)
            delta_R = mean_return - return_ema
            new_return_ema = tc.rcb_ema_alpha * mean_return + (1 - tc.rcb_ema_alpha) * return_ema
            r_ext = traj_batch.reward.astype(jnp.float32)
            use_intrinsic = update_steps >= tc.warmup_rollouts
            if tc.use_lagrangian:
                lagrangian_beta_new = jnp.clip(lagrangian_beta + tc.lagrangian_lr * (tc.rcb_target - new_return_ema), 0.0, tc.rcb_beta_max)
                adaptive_beta = lagrangian_beta_new
                sigmoid_val = jnp.float32(0.0)
            else:
                sigmoid_val = jax.nn.sigmoid(tc.rcb_kappa * (tc.rcb_target - new_return_ema))
                adaptive_beta = tc.rcb_beta_min + (tc.rcb_beta_max - tc.rcb_beta_min) * sigmoid_val
                lagrangian_beta_new = lagrangian_beta
            batch_mean = jnp.mean(intrinsic_reward_per_agent, axis=(1, 2))
            batch_var = jnp.var(intrinsic_reward_per_agent, axis=(1, 2))
            ir_mu_new = tc.rsq_ema_alpha * batch_mean + (1 - tc.rsq_ema_alpha) * ir_mu
            ir_var_new = tc.rsq_ema_alpha * batch_var + (1 - tc.rsq_ema_alpha) * ir_var
            mu_sq = ir_mu_new ** 2
            rsq_ir = mu_sq / (mu_sq + ir_var_new + 1e-08)
            adv_mu_sq = adv_mu ** 2
            rsq_adv = adv_mu_sq / (adv_mu_sq + adv_var + 1e-08)
            rsq = jax.lax.cond(tc.rsq_use_advantage, lambda : rsq_adv, lambda : rsq_ir)
            g = jnp.clip(1.0 + tc.rsq_lambda * (rsq - tc.rsq_ref), tc.rsq_h_min, tc.rsq_h_max)
            g_broadcast = jnp.repeat(g, tc.num_envs)
            g_broadcast = jnp.broadcast_to(g_broadcast, (tc.num_steps, tc.num_actors))
            r_int_modulated = r_int_scaled * g_broadcast
            r_combined = jax.lax.cond(use_intrinsic, lambda : r_ext + adaptive_beta * r_int_modulated, lambda : r_ext)
            traj_batch = traj_batch._replace(intrinsic_reward=r_int_scaled)

            def _calculate_gae(traj, last_v, combined_r):

                def _get_advantages(gae_and_nv, inputs):
                    (gae, nv) = gae_and_nv
                    (done, value, reward) = inputs
                    delta = reward + tc.gamma * nv * (1 - done) - value
                    gae = delta + tc.gamma * tc.gae_lambda * (1 - done) * gae
                    return ((gae, value), gae)
                (_, advantages) = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_v), last_v), (traj.global_done, traj.value, combined_r), reverse=True, unroll=16)
                return (advantages, advantages + traj.value)
            (advantages, targets) = _calculate_gae(traj_batch, last_val, r_combined)
            adv_per_agent = advantages.reshape(tc.num_steps, tc.num_agents, tc.num_envs).transpose(1, 0, 2)
            adv_batch_mean = jnp.mean(adv_per_agent, axis=(1, 2))
            adv_batch_var = jnp.var(adv_per_agent, axis=(1, 2))
            adv_mu_new = tc.rsq_ema_alpha * adv_batch_mean + (1 - tc.rsq_ema_alpha) * adv_mu
            adv_var_new = tc.rsq_ema_alpha * adv_batch_var + (1 - tc.rsq_ema_alpha) * adv_var

            def _update_epoch(update_state, unused):
                (train_states_upd, traj_batch_upd, advantages_upd, targets_upd, rng_upd) = update_state
                (rng_upd, _rng) = jax.random.split(rng_upd)
                (actor_ts_upd, critic_ts_upd) = train_states_upd

                def to_per_agent_3d(x):
                    return x.reshape(tc.num_steps, tc.num_agents, tc.num_envs, -1).transpose(1, 0, 2, 3)

                def to_per_agent_2d(x):
                    return x.reshape(tc.num_steps, tc.num_agents, tc.num_envs).transpose(1, 0, 2)
                obs_pa = to_per_agent_3d(traj_batch_upd.obs)
                action_pa = to_per_agent_3d(traj_batch_upd.action)
                log_prob_pa = to_per_agent_2d(traj_batch_upd.log_prob)
                adv_pa = to_per_agent_2d(advantages_upd)
                value_pa = to_per_agent_2d(traj_batch_upd.value)
                targets_pa = to_per_agent_2d(targets_upd)
                batch_size_pa = tc.num_steps * tc.num_envs
                obs_flat = obs_pa.reshape(tc.num_agents, batch_size_pa, -1)
                action_flat = action_pa.reshape(tc.num_agents, batch_size_pa, -1)
                log_prob_flat = log_prob_pa.reshape(tc.num_agents, batch_size_pa)
                adv_flat = adv_pa.reshape(tc.num_agents, batch_size_pa)
                value_flat = value_pa.reshape(tc.num_agents, batch_size_pa)
                targets_flat = targets_pa.reshape(tc.num_agents, batch_size_pa)
                perm = jax.random.permutation(_rng, batch_size_pa)

                def shuffle_split(x):
                    x_shuf = jnp.take(x, perm, axis=1)
                    return x_shuf.reshape(tc.num_agents, tc.num_minibatches, -1, *x.shape[2:])
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
                        loss2 = jnp.clip(ratio, 1 - tc.clip_eps, 1 + tc.clip_eps) * gae_norm
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
                        v_clipped = value_old + (value - value_old).clip(-tc.clip_eps, tc.clip_eps)
                        v_losses = jnp.square(value - tgt)
                        v_losses_clipped = jnp.square(v_clipped - tgt)
                        v_loss = 0.5 * jnp.maximum(v_losses, v_losses_clipped).mean()
                        return (tc.vf_coef * v_loss, v_loss)

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
            upd_train_states = (actor_train_states_cur, critic_train_states_cur)
            init_state = (upd_train_states, traj_batch, advantages, targets, _rng_update)
            (update_state_out, update_metrics) = jax.lax.scan(_update_epoch, init_state, None, tc.update_epochs)
            ((actor_train_states_cur, critic_train_states_cur), _, _, _, _) = update_state_out
            env_info = jax.tree.map(lambda x: x.mean(), traj_batch.info)
            update_metrics_agg = jax.tree.map(lambda x: x.mean(), update_metrics)
            metrics = {**update_metrics_agg, 'env/reward_mean': traj_batch.reward.mean(), 'env/reward_std': traj_batch.reward.std(), 'intrinsic/r_int_raw_mean': intrinsic_reward_raw.mean(), 'intrinsic/r_int_scaled_mean': r_int_scaled.mean(), 'tdd/loss': tdd_loss, 'rcb/adaptive_beta': adaptive_beta, 'rcb/return_ema': new_return_ema, 'rcb/sigmoid_val': sigmoid_val, 'rsq/rsq_mean': jnp.mean(rsq), 'rsq/rsq_min': jnp.min(rsq), 'rsq/rsq_max': jnp.max(rsq), 'rsq/rsq_ir_range': jnp.max(rsq_ir) - jnp.min(rsq_ir), 'rsq/rsq_adv_range': jnp.max(rsq_adv) - jnp.min(rsq_adv), 'rsq/g_mean': jnp.mean(g), 'rsq/g_min': jnp.min(g), 'rsq/g_max': jnp.max(g), 'rewards/extrinsic': jnp.mean(r_ext), 'rewards/combined': jnp.mean(r_combined)}
            for (k, v) in env_info.items():
                metrics[f'env/{k}'] = v
            timesteps = (update_steps + 1) * tc.num_envs * tc.num_steps
            callback_metrics = {k: v for (k, v) in metrics.items()}
            callback_metrics['_timesteps'] = timesteps
            callback_metrics['_update_step'] = update_steps + 1
            jax.experimental.io_callback(_wandb_log_callback, None, callback_metrics, ordered=True)
            train_states = (actor_train_states_cur, critic_train_states_cur, tdd_pot_states, tdd_enc_states)
            runner_state = (train_states, env_state, last_obs, last_done, rng)
            return ((runner_state, update_steps + 1, new_return_ema, ir_mu_new, ir_var_new, adv_mu_new, adv_var_new, lagrangian_beta_new), metrics)
        init_done = jnp.zeros((tc.num_actors,), dtype=bool)
        train_states = (actor_train_states, critic_train_states, tdd_potential_states, tdd_s_encoder_states)
        runner_state = (train_states, env_state, obsv, init_done, rng)
        update_runner_state = (runner_state, jnp.array(0), jnp.float32(0.0), ir_mu_init, ir_var_init, adv_mu_init, adv_var_init, lagrangian_beta_init)
        (update_runner_state, metrics_all) = jax.lax.scan(_update_step, update_runner_state, None, tc.num_updates)
        return {'metrics': metrics_all, 'runner_state': update_runner_state}
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

@hydra.main(version_base=None, config_path='config', config_name='rcb_rsq_ant')
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    print_training_header(config)
    rng = jax.random.PRNGKey(config['SEED'])
    wandb.init(project=config.get('PROJECT', 'rcb-rsq-mabrax'), name=config.get('RUN_NAME', f"rcb-rsq-{config['ENV_NAME']}"), tags=['rcb-rsq', 'mabrax', config['ENV_NAME']], config=config, mode=config.get('WANDB_MODE', 'online'))
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
    print(f'\n[RCB-RSQ-BRAX] Training completed in {t_end - t_start:.1f}s')
    final_return = float(result['metrics']['env/returned_episode_returns'][-1])
    print(f'[RCB-RSQ-BRAX] Final return: {final_return:.3f}')
    run_name = config.get('RUN_NAME', None)
    if run_name and 'runner_state' in result:
        import pickle
        ckpt_dir = os.path.join('checkpoints', config['ENV_NAME'], run_name)
        os.makedirs(ckpt_dir, exist_ok=True)
        try:
            update_runner_state = result['runner_state']
            runner_state_inner = update_runner_state[0]
            train_states = runner_state_inner[0]
            actor_ts_tuple = train_states[0]
            num_agents = len(actor_ts_tuple)
            for i in range(num_agents):
                agent_params = jax.device_get(actor_ts_tuple[i].params)
                pkl_path = os.path.join(ckpt_dir, f'actor_{i}_final.pkl')
                with open(pkl_path, 'wb') as f:
                    pickle.dump(agent_params, f)
            print(f'[RCB-RSQ-BRAX] Saved {num_agents} actor checkpoints to {ckpt_dir}')
        except Exception as e:
            print(f'[RCB-RSQ-BRAX] Warning: Could not save checkpoints: {e}')
    wandb.finish()
if __name__ == '__main__':
    main()
