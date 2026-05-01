import os
import sys
import setup_jax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import distrax
import flax.linen as nn
import hydra
from omegaconf import OmegaConf
import jaxmarl
from jaxmarl.wrappers.baselines import MPELogWrapper, LogWrapper
import wandb
from time import time
from typing import NamedTuple, Dict, Sequence
from core import MPEWorldStateWrapper, SMAXWorldStateWrapper, ScannedRNN, create_actor_train_state, create_critic_train_state, Transition, batchify, unbatchify, make_linear_schedule, compute_video_steps, stack_params, save_params, CallbackHandler, vectorized_apply_gradients, ActorRNN
from flax.training.train_state import TrainState
from flax.linen.initializers import orthogonal, constant

class MAVENConfig(NamedTuple):
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
    max_grad_norm: float
    num_updates: int
    update_epochs: int
    seed: int
    noise_dim: int
    mi_loss_weight: float
    discrim_hidden_dim: int
    discrim_lr: float
    hier_lr: float
    hier_hidden_dim: int

def create_maven_config(config: Dict, num_agents: int) -> MAVENConfig:
    return MAVENConfig(num_envs=config['NUM_ENVS'], num_steps=config['NUM_STEPS'], num_agents=num_agents, num_actors=config['NUM_ACTORS'], gamma=config['GAMMA'], gae_lambda=config['GAE_LAMBDA'], clip_eps=config['CLIP_EPS'], ent_coef=config['ENT_COEF'], anneal_ent=config.get('ANNEAL_ENT', False), ent_coef_end=config.get('ENT_COEF_END', config['ENT_COEF']), max_grad_norm=config['MAX_GRAD_NORM'], num_updates=config['NUM_UPDATES'], update_epochs=config['UPDATE_EPOCHS'], seed=config['SEED'], noise_dim=config.get('NOISE_DIM', 8), mi_loss_weight=config.get('MI_LOSS_WEIGHT', 1.0), discrim_hidden_dim=config.get('DISCRIM_HIDDEN_DIM', 128), discrim_lr=config.get('DISCRIM_LR', 0.001), hier_lr=config.get('HIER_LR', 0.001), hier_hidden_dim=config.get('HIER_HIDDEN_DIM', 64))

class Discriminator(nn.Module):
    hidden_dim: int
    noise_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim // 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        logits = nn.Dense(self.noise_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        return logits

class HierarchicalPolicy(nn.Module):
    hidden_dim: int
    noise_dim: int

    @nn.compact
    def __call__(self, s0):
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(s0)
        x = nn.relu(x)
        x = nn.Dense(self.noise_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        return x

def make_train(config):
    env_kwargs = config.get('ENV_KWARGS', {})
    env_name = config['ENV_NAME']
    if 'corridor' in env_name.lower():
        env_kwargs['sparse_reward'] = config.get('SPARSE_REWARD', False)
    env = jaxmarl.make(env_name, **env_kwargs)
    num_learning_agents = len(env.agents)
    config['NUM_ACTORS'] = num_learning_agents * config['NUM_ENVS']
    config['NUM_UPDATES'] = config['TOTAL_TIMESTEPS'] // config['NUM_STEPS'] // config['NUM_ENVS']
    tc = create_maven_config(config, num_learning_agents)
    is_smax = 'SMAX' in env_name or 'smax' in env_name
    if is_smax:
        env = SMAXWorldStateWrapper(env)
        env = LogWrapper(env)
    else:
        env = MPEWorldStateWrapper(env)
        env = MPELogWrapper(env)
    _obs_dim_static = env.observation_space(env.agents[0]).shape[0]
    if is_smax:
        sd_pos_start = config.get('SD_POS_START', _obs_dim_static - 9)
        sd_pos_end = config.get('SD_POS_END', _obs_dim_static - 7)
    else:
        sd_pos_start = config.get('SD_POS_START', 2)
        sd_pos_end = config.get('SD_POS_END', 4)
    actor_lr = config.get('ACTOR_LR') or config.get('LR', 0.001)
    critic_lr = config.get('CRITIC_LR') or config.get('LR', 0.001)
    actor_lr_schedule = None
    critic_lr_schedule = None

    def train(rng):
        (rng, _rng_actor, _rng_critic, _rng_discrim, _rng_hier) = jax.random.split(rng, 5)
        obs_dim = env.observation_space(env.agents[0]).shape[0]
        action_dim = env.action_space(env.agents[0]).n
        actor_input_dim = obs_dim + tc.noise_dim
        (_, _, actor_networks, actor_train_states) = create_actor_train_state(rng=_rng_actor, config=config, action_dim=action_dim, obs_dim=actor_input_dim, num_agents=tc.num_agents, lr_schedule=actor_lr_schedule, shared_params=False)
        (critic_network, critic_train_state) = create_critic_train_state(rng=_rng_critic, config=config, world_state_size=env.world_state_size(), lr_schedule=critic_lr_schedule)
        discrim_net = Discriminator(hidden_dim=tc.discrim_hidden_dim, noise_dim=tc.noise_dim)
        discrim_input_dim = tc.num_agents * action_dim
        discrim_params = discrim_net.init(_rng_discrim, jnp.zeros((1, discrim_input_dim)))
        discrim_tx = optax.chain(optax.clip_by_global_norm(tc.max_grad_norm), optax.adam(learning_rate=tc.discrim_lr, eps=1e-05))
        discrim_state = TrainState.create(apply_fn=discrim_net.apply, params=discrim_params, tx=discrim_tx)
        hier_net = HierarchicalPolicy(hidden_dim=tc.hier_hidden_dim, noise_dim=tc.noise_dim)
        world_state_size = env.world_state_size()
        hier_params = hier_net.init(_rng_hier, jnp.zeros((1, world_state_size)))
        hier_tx = optax.chain(optax.clip_by_global_norm(tc.max_grad_norm), optax.adam(learning_rate=tc.hier_lr, eps=1e-05))
        hier_state = TrainState.create(apply_fn=hier_net.apply, params=hier_params, tx=hier_tx)
        print(f'[MAVEN] noise_dim={tc.noise_dim}, mi_loss_weight={tc.mi_loss_weight}')
        print(f'[MAVEN] Actor input: obs({obs_dim}) + z({tc.noise_dim}) = {actor_input_dim}')
        print(f'[MAVEN] Discriminator input: {discrim_input_dim}')
        print(f'[MAVEN] Hierarchical policy: world_state({world_state_size}) -> z({tc.noise_dim}), lr={tc.hier_lr}')
        (rng, _rng) = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, tc.num_envs)
        (obsv, env_state) = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        ac_init_hstate = ScannedRNN.initialize_carry(tc.num_actors, config['GRU_HIDDEN_DIM'])
        cr_init_hstate = ScannedRNN.initialize_carry(tc.num_actors, config['GRU_HIDDEN_DIM'])
        raw_env = env
        while hasattr(raw_env, '_env'):
            raw_env = raw_env._env
        video_steps = compute_video_steps(tc.num_updates, 0, config.get('NUM_INTERMEDIATE_VIDEOS', 6))
        callback_handler = CallbackHandler(config=config, env=raw_env, actor_network=actor_networks, critic_network=critic_network, shared_actor_params=False, num_agents=tc.num_agents, video_steps=video_steps)
        agent_idx_array = jnp.repeat(jnp.arange(tc.num_agents), tc.num_envs)

        def _update_step(update_runner_state, unused):
            (runner_state, update_steps) = update_runner_state
            current_ent_coef = jax.lax.cond(tc.anneal_ent, lambda : tc.ent_coef - (tc.ent_coef - tc.ent_coef_end) * (update_steps / tc.num_updates), lambda : jnp.float32(tc.ent_coef))
            (train_states, env_state, last_obs, last_done, hstates, rng) = runner_state
            (actor_train_states_local, critic_train_state_local, discrim_state_local, hier_state_local) = train_states
            (rng, _rng_z) = jax.random.split(rng)
            s0 = last_obs['world_state'][:, 0, :]
            hier_logits = hier_net.apply(hier_state_local.params, s0)
            z_indices = jax.random.categorical(_rng_z, hier_logits)
            hier_log_prob = jax.nn.log_softmax(hier_logits)[jnp.arange(tc.num_envs), z_indices]
            z_onehot = jax.nn.one_hot(z_indices, tc.noise_dim)
            z_per_actor = jnp.tile(z_onehot, (tc.num_agents, 1))
            runner_state = (train_states, env_state, last_obs, last_done, hstates, rng)

            def _env_step(runner_state, unused):
                (train_states, env_state, last_obs, last_done, hstates, rng) = runner_state
                (actor_train_states_local, critic_train_state_local, discrim_state_local, hier_state_local) = train_states
                (rng, _rng) = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, tc.num_actors)
                obs_with_z = jnp.concatenate([obs_batch, z_per_actor], axis=-1)
                actor_template = actor_networks[0]
                stacked_actor_params = stack_params([s.params for s in actor_train_states_local])
                obs_z_reshaped = obs_with_z.reshape(tc.num_agents, tc.num_envs, -1)
                done_reshaped = last_done.reshape(tc.num_agents, tc.num_envs)
                hstate_reshaped = hstates[0].reshape(tc.num_agents, tc.num_envs, -1)

                def _forward_single_actor(params, hidden, obs, done):
                    ac_in = (obs[np.newaxis, :], done[np.newaxis, :])
                    (new_hidden, _, logits) = actor_template.apply(params, hidden, ac_in)
                    return (new_hidden, logits.squeeze(0))
                (ac_hstate_per_agent, action_logits_per_agent) = jax.vmap(_forward_single_actor)(stacked_actor_params, hstate_reshaped, obs_z_reshaped, done_reshaped)
                ac_hstate = ac_hstate_per_agent.reshape(tc.num_actors, -1)
                action_logits = action_logits_per_agent.reshape(tc.num_actors, -1)
                pi = distrax.Categorical(logits=action_logits)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, tc.num_envs, tc.num_agents)
                if is_smax:
                    env_act = {a: v.squeeze(-1) for (a, v) in env_act.items()}
                world_state = last_obs['world_state'].swapaxes(0, 1).reshape((tc.num_actors, -1))
                cr_in = (world_state[None, :], last_done[np.newaxis, :])
                (cr_hstate, value) = critic_network.apply(critic_train_state_local.params, hstates[1], cr_in)
                (rng, _rng) = jax.random.split(rng)
                rng_step = jax.random.split(_rng, tc.num_envs)
                (obsv, env_state, reward, done, info) = jax.vmap(env.step, in_axes=(0, 0, 0))(rng_step, env_state, env_act)
                info = jax.tree.map(lambda x: x.reshape(tc.num_actors), info)
                done_batch = batchify(done, env.agents, tc.num_actors).squeeze()
                agent_positions = obs_batch[:, sd_pos_start:sd_pos_end]
                transition = Transition(jnp.tile(done['__all__'], tc.num_agents), last_done, action.squeeze(), action_logits, value.squeeze(), batchify(reward, env.agents, tc.num_actors).squeeze(), log_prob.squeeze(), obs_batch, world_state, info, z_i=z_per_actor, intrinsic_reward=None, agent_positions=agent_positions, agent_local_features_curr=agent_positions, agent_local_features_next=None, agent_idx=agent_idx_array)
                runner_state = (train_states, env_state, obsv, done_batch, (ac_hstate, cr_hstate), rng)
                return (runner_state, transition)
            initial_hstates = runner_state[-2]
            (runner_state, traj_batch) = jax.lax.scan(_env_step, runner_state, None, tc.num_steps)
            (train_states, env_state, last_obs, last_done, hstates, rng) = runner_state
            (actor_train_states, critic_train_state, discrim_state, hier_state) = train_states
            logits_reshaped = traj_batch.action_logits.reshape(tc.num_steps, tc.num_agents, tc.num_envs, action_dim)
            mean_logits = jnp.mean(jax.nn.softmax(logits_reshaped, axis=-1), axis=0)
            discrim_input = mean_logits.transpose(1, 0, 2).reshape(tc.num_envs, -1)

            def _discrim_loss(discrim_params):
                pred_logits = discrim_net.apply(discrim_params, discrim_input)
                loss = optax.softmax_cross_entropy_with_integer_labels(pred_logits, z_indices).mean()
                return loss
            (mi_loss, discrim_grads) = jax.value_and_grad(_discrim_loss)(discrim_state.params)
            discrim_state = discrim_state.apply_gradients(grads=discrim_grads)
            pred_logits_for_bonus = discrim_net.apply(jax.lax.stop_gradient(discrim_state.params), discrim_input)
            log_q = jax.nn.log_softmax(pred_logits_for_bonus)
            mi_bonus_per_env = log_q[jnp.arange(tc.num_envs), z_indices]
            mi_bonus_per_env = jnp.clip(mi_bonus_per_env, -5.0, 0.0)
            mi_bonus_per_actor = jnp.tile(mi_bonus_per_env, tc.num_agents)
            mi_bonus_tiled = jnp.broadcast_to(mi_bonus_per_actor[None, :], (tc.num_steps, tc.num_actors))
            reward_reshaped = traj_batch.reward.reshape(tc.num_steps, tc.num_agents, tc.num_envs)
            episode_return = reward_reshaped.sum(axis=1).mean(axis=0)
            ret_normalized = (episode_return - episode_return.mean()) / (episode_return.std() + 1e-08)

            def _hier_loss_fn(hier_params):
                h_logits = hier_net.apply(hier_params, s0)
                h_log_prob = jax.nn.log_softmax(h_logits)[jnp.arange(tc.num_envs), z_indices]
                return -(h_log_prob * jax.lax.stop_gradient(ret_normalized)).mean()
            (hier_loss_val, hier_grads) = jax.value_and_grad(_hier_loss_fn)(hier_state.params)
            hier_state = hier_state.apply_gradients(grads=hier_grads)
            r_ext = traj_batch.reward.astype(jnp.float32)
            r_total = r_ext + tc.mi_loss_weight * mi_bonus_tiled
            last_world_state = last_obs['world_state'].swapaxes(0, 1).reshape((tc.num_actors, -1))
            cr_in = (last_world_state[None, :], last_done[np.newaxis, :])
            (_, last_val) = critic_network.apply(critic_train_state.params, hstates[1], cr_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch_arg, last_val_arg, rewards):

                def _get_advantages(gae_and_next_value, inputs):
                    (gae, next_value) = gae_and_next_value
                    (done, value, reward) = inputs
                    (next_value, value) = (next_value.reshape(-1), value.reshape(-1))
                    delta = reward + tc.gamma * next_value * (1 - done) - value
                    gae = delta + tc.gamma * tc.gae_lambda * (1 - done) * gae
                    return ((gae, value), gae)
                inputs = (traj_batch_arg.global_done, traj_batch_arg.value, rewards)
                (_, advantages) = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val_arg), last_val_arg), inputs, reverse=True, unroll=16)
                return (advantages, advantages + traj_batch_arg.value)
            (advantages, targets) = _calculate_gae(traj_batch, last_val, r_total)

            def _update_epoch(update_state, unused):
                (train_states_upd, init_hstates, traj_batch_upd, advantages_upd, targets_upd, rng_upd) = update_state
                (rng_upd, _rng) = jax.random.split(rng_upd)
                (actor_train_states_upd, critic_train_state_upd, discrim_state_upd, hier_state_upd) = train_states_upd

                def to_per_agent_3d(x):
                    return x.reshape(tc.num_steps, tc.num_agents, tc.num_envs).transpose(1, 0, 2)

                def to_per_agent_4d(x):
                    return x.reshape(tc.num_steps, tc.num_agents, tc.num_envs, -1).transpose(1, 0, 2, 3)
                agent_obs = to_per_agent_4d(traj_batch_upd.obs)
                agent_z = to_per_agent_4d(traj_batch_upd.z_i)
                agent_obs_z = jnp.concatenate([agent_obs, agent_z], axis=-1)
                agent_done = to_per_agent_3d(traj_batch_upd.done)
                agent_action = to_per_agent_3d(traj_batch_upd.action)
                agent_log_prob = to_per_agent_3d(traj_batch_upd.log_prob)
                agent_gae = to_per_agent_3d(advantages_upd)
                agent_init_hstate = init_hstates[0].reshape(tc.num_agents, tc.num_envs, -1)
                actor_template = actor_networks[0]
                stacked_actor_params = stack_params([s.params for s in actor_train_states_upd])

                def _actor_loss_single_agent(actor_params, obs_z, done, action, log_prob, gae, init_hstate, agent_rng):
                    perm = jax.random.permutation(agent_rng, tc.num_envs)
                    obs_z_shuffled = jnp.take(obs_z, perm, axis=1)
                    done_shuffled = jnp.take(done, perm, axis=1)
                    action_shuffled = jnp.take(action, perm, axis=1)
                    log_prob_shuffled = jnp.take(log_prob, perm, axis=1)
                    gae_shuffled = jnp.take(gae, perm, axis=1)
                    hstate_shuffled = jnp.take(init_hstate, perm, axis=0)
                    (_, pi, _) = actor_template.apply(actor_params, hstate_shuffled, (obs_z_shuffled, done_shuffled))
                    new_log_prob = pi.log_prob(action_shuffled)
                    entropy = pi.entropy()
                    logratio = new_log_prob - log_prob_shuffled
                    ratio = jnp.exp(logratio)
                    gae_normalized = (gae_shuffled - gae_shuffled.mean()) / (gae_shuffled.std() + 1e-06)
                    loss_actor1 = ratio * gae_normalized
                    loss_actor2 = jnp.clip(ratio, 1.0 - tc.clip_eps, 1.0 + tc.clip_eps) * gae_normalized
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                    entropy_mean = entropy.mean()
                    approx_kl = (ratio - 1 - logratio).mean()
                    clip_frac = jnp.mean(jnp.abs(ratio - 1) > tc.clip_eps)
                    actor_loss = loss_actor - current_ent_coef * entropy_mean
                    return (actor_loss, (loss_actor, entropy_mean, ratio, approx_kl, clip_frac))

                def _compute_actor_grad(params, obs_z, done, action, log_prob, gae, init_hstate, rng):
                    ((loss, aux), grad) = jax.value_and_grad(_actor_loss_single_agent, has_aux=True)(params, obs_z, done, action, log_prob, gae, init_hstate, rng)
                    return (loss, aux, grad)
                agent_rngs = jax.random.split(_rng, tc.num_agents)
                (actor_losses, actor_aux, actor_grads) = jax.vmap(_compute_actor_grad)(stacked_actor_params, agent_obs_z, agent_done, agent_action, agent_log_prob, agent_gae, agent_init_hstate, agent_rngs)
                new_actor_states = vectorized_apply_gradients(actor_train_states_upd, actor_grads)

                def _critic_loss_fn(critic_params, init_hstate, traj_batch_inner, targets_inner):
                    (_, value) = critic_network.apply(critic_params, init_hstate, (traj_batch_inner.world_state, traj_batch_inner.done))
                    value_pred_clipped = traj_batch_inner.value + (value - traj_batch_inner.value).clip(-tc.clip_eps, tc.clip_eps)
                    value_losses = jnp.square(value - targets_inner)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets_inner)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    return (value_loss, value_loss)
                cr_init_hstate = init_hstates[1]
                critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                (critic_loss, critic_grads) = critic_grad_fn(critic_train_state_upd.params, cr_init_hstate, traj_batch_upd, targets_upd)
                critic_train_state_upd = critic_train_state_upd.apply_gradients(grads=critic_grads)
                actor_loss_mean = actor_losses.mean()
                entropy_mean = actor_aux[1].mean()
                ratio_mean = actor_aux[2].mean()
                approx_kl_mean = actor_aux[3].mean()
                clip_frac_mean = actor_aux[4].mean()
                total_loss = actor_loss_mean + critic_loss[0]
                loss_info = {'total_loss': total_loss, 'actor_loss': actor_loss_mean, 'value_loss': critic_loss[0], 'entropy': entropy_mean, 'ratio': ratio_mean, 'approx_kl': approx_kl_mean, 'clip_frac': clip_frac_mean}
                train_states_upd = (new_actor_states, critic_train_state_upd, discrim_state_upd, hier_state_upd)
                update_state = (train_states_upd, init_hstates, traj_batch_upd, advantages_upd, targets_upd, rng_upd)
                return (update_state, loss_info)
            train_states = (actor_train_states, critic_train_state, discrim_state, hier_state)
            update_state = (train_states, initial_hstates, traj_batch, advantages, targets, rng)
            (update_state, loss_info) = jax.lax.scan(_update_epoch, update_state, None, tc.update_epochs)
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            loss_info['mi_loss'] = mi_loss
            train_states = update_state[0]
            (actor_train_states, critic_train_state, discrim_state, hier_state) = train_states
            metric = traj_batch.info
            loss_info['hier_loss'] = hier_loss_val
            metric['loss'] = loss_info
            metric['tdd_loss'] = jnp.float32(0.0)
            mi_bonus_mean = jnp.mean(mi_bonus_tiled)
            metric['rewards'] = {'extrinsic': jnp.mean(r_ext), 'intrinsic_scaled': tc.mi_loss_weight * mi_bonus_mean, 'combined': jnp.mean(r_total)}
            metric['ent_coef'] = current_ent_coef
            metric['action_logits'] = traj_batch.action_logits
            metric['intrinsic_reward_scaled'] = tc.mi_loss_weight * mi_bonus_tiled
            metric['intrinsic_reward_raw'] = mi_bonus_tiled
            metric['agent_positions'] = traj_batch.agent_positions
            (metric['num_agents'], metric['num_envs']) = (tc.num_agents, tc.num_envs)
            metric['intrinsic_reward_scale'] = jnp.float32(tc.mi_loss_weight)
            metric['actor_params'] = [s.params for s in actor_train_states]
            metric['critic_params'] = critic_train_state.params
            rng = update_state[-1]
            metric['update_steps'] = update_steps
            jax.experimental.io_callback(callback_handler, None, metric, ordered=True)
            update_steps = update_steps + 1
            train_states = (actor_train_states, critic_train_state, discrim_state, hier_state)
            runner_state = (train_states, env_state, last_obs, last_done, hstates, rng)
            return ((runner_state, update_steps), metric)
        (rng, _rng) = jax.random.split(rng)
        train_states = (actor_train_states, critic_train_state, discrim_state, hier_state)
        runner_state = (train_states, env_state, obsv, jnp.zeros(tc.num_actors, dtype=bool), (ac_init_hstate, cr_init_hstate), _rng)
        (runner_state, metric) = jax.lax.scan(_update_step, (runner_state, 0), None, tc.num_updates)
        return {'runner_state': runner_state}
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

@hydra.main(version_base=None, config_path='config', config_name='mappo_maven_corridor')
def main(config):
    config = OmegaConf.to_container(config)
    print('\n' + '=' * 80)
    print('MAVEN-MAPPO: Multi-Agent Variational Exploration')
    print('=' * 80)
    print(f"Environment: {config['ENV_NAME']}")
    print(f"Seed: {config['SEED']}")
    print(f"Total Timesteps: {config['TOTAL_TIMESTEPS']:,.0f}")
    print(f"Noise Dim (z): {config.get('NOISE_DIM', 8)}")
    print(f"MI Loss Weight: {config.get('MI_LOSS_WEIGHT', 1.0)}")
    print('=' * 80 + '\n')
    wandb_project = config.get('PROJECT', 'mappo')
    env_display_name = config['ENV_NAME'].replace('/', '-').replace('_', '-')
    run_name = config.get('RUN_NAME') or f'maven-{env_display_name}'
    run = wandb.init(project=wandb_project, name=run_name, tags=['maven', 'mappo', config['ENV_NAME']], config=config, mode=config['WANDB_MODE'])
    rng = jax.random.PRNGKey(config['SEED'])
    print('Starting training...')
    try:
        with jax.disable_jit(False):
            train_jit = jax.jit(make_train(config))
            start_time = time()
            out = train_jit(rng)
            elapsed_time = time() - start_time
            print(f"\n{'=' * 80}\nTraining completed in {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes)\n{'=' * 80}")
    except KeyboardInterrupt:
        print('\n> Training interrupted by user')
        raise
    except Exception as e:
        print(f'\n> Training failed with error: {e}')
        import traceback
        traceback.print_exc()
        raise
    from core.utils import save_training_checkpoints
    save_training_checkpoints(run, config, out, save_params)
if __name__ == '__main__':
    main()
