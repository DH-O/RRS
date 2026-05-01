import sys
import jax
import jax.numpy as jnp
import numpy as np
import optax
import distrax
import hydra
from omegaconf import OmegaConf
import jaxmarl
from jaxmarl.wrappers.baselines import MPELogWrapper, LogWrapper
import wandb
from time import time
from typing import NamedTuple, Dict
from core import MPEWorldStateWrapper, SMAXWorldStateWrapper, ScannedRNN, create_actor_train_state, create_critic_train_state, create_tdd_train_states, Transition, batchify, unbatchify, discounted_sampling, compute_video_steps, stack_params, save_params, CallbackHandler, compute_intrinsic_reward_sd, mrn_distance, vectorized_apply_gradients
import flax.linen as nn

class RNDNetwork(nn.Module):
    hidden_dim: int = 64
    output_dim: int = 32

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x

class TrainerConfigRCBRSQ(NamedTuple):
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
    intrinsic_reward_beta: float
    intrinsic_reward_scale: float
    warmup_rollouts: int
    seed: int
    tdd_latent_dim: int
    tdd_output_dim: int
    tdd_lr: float
    tdd_batch_size: int
    tdd_update_epochs: int
    tdd_discount: float
    rcb_beta_max: float
    rcb_kappa: float
    rcb_target: float
    rcb_ema_alpha: float
    rcb_beta_min: float
    rcb_decay_start: float
    rsq_lambda: float
    rsq_ref: float
    rsq_h_min: float
    rsq_h_max: float
    rsq_ema_alpha: float
    rsq_waterfilling: bool
    adaptive_target_mode: int
    target_percentile: float
    target_history_size: int
    running_max_ratio: float
    intrinsic_reward_type: int
    allocation_mode: int
    meir_alpha: float
    cift_enabled: bool
    cift_C: float
    cift_floor: float

def create_trainer_config(config: Dict, num_agents: int) -> TrainerConfigRCBRSQ:
    gamma = config['GAMMA']
    gae_lambda = config['GAE_LAMBDA']
    return TrainerConfigRCBRSQ(num_envs=config['NUM_ENVS'], num_steps=config['NUM_STEPS'], num_agents=num_agents, num_actors=config['NUM_ACTORS'], gamma=gamma, gae_lambda=gae_lambda, clip_eps=config['CLIP_EPS'], ent_coef=config['ENT_COEF'], anneal_ent=config.get('ANNEAL_ENT', False), ent_coef_end=config.get('ENT_COEF_END', config['ENT_COEF']), max_grad_norm=config['MAX_GRAD_NORM'], num_updates=config['NUM_UPDATES'], update_epochs=config['UPDATE_EPOCHS'], intrinsic_reward_beta=config.get('INTRINSIC_REWARD_BETA', 0.1), intrinsic_reward_scale=config.get('INTRINSIC_REWARD_SCALE', 1.0), warmup_rollouts=config.get('WARMUP_ROLLOUTS', 0), seed=config['SEED'], tdd_latent_dim=config.get('TDD_LATENT_DIM', 64), tdd_output_dim=config.get('TDD_OUTPUT_DIM', 64), tdd_lr=config.get('TDD_LR', 0.00039), tdd_batch_size=config.get('TDD_BATCH_SIZE', 512), tdd_update_epochs=config.get('TDD_UPDATE_EPOCHS', 144), tdd_discount=config.get('TDD_DISCOUNT', 0.93), rcb_beta_max=config.get('RCB_BETA_MAX', 0.5), rcb_kappa=config.get('RCB_KAPPA', 0.015), rcb_target=config.get('RCB_TARGET', 400.0), rcb_ema_alpha=config.get('RCB_EMA_ALPHA', 0.03), rcb_beta_min=config.get('RCB_BETA_MIN', 0.05), rcb_decay_start=config.get('RCB_DECAY_START', 1.0), rsq_lambda=config.get('RSQ_LAMBDA', 2.0), rsq_ref=config.get('RSQ_REF', 0.5), rsq_h_min=config.get('RSQ_H_MIN', 0.1), rsq_h_max=config.get('RSQ_H_MAX', 2.0), rsq_ema_alpha=config.get('RSQ_EMA_ALPHA', 0.1), rsq_waterfilling=config.get('RSQ_WATERFILLING', False), adaptive_target_mode={'none': 0, 'percentile': 1, 'running_max': 2}.get(config.get('ADAPTIVE_TARGET_MODE', 'none'), 0), target_percentile=config.get('TARGET_PERCENTILE', 50.0), target_history_size=config.get('TARGET_HISTORY_SIZE', 100), running_max_ratio=config.get('RUNNING_MAX_RATIO', 0.8), intrinsic_reward_type={'tdd': 0, 'entropy': 1, 'rnd': 2, 'count': 3}.get(config.get('INTRINSIC_REWARD_TYPE', 'tdd'), 0), allocation_mode={'affine': 0, 'waterfilling': 1, 'meir': 2}.get(config.get('ALLOCATION_MODE', 'waterfilling' if config.get('RSQ_WATERFILLING', False) else 'affine'), 0), meir_alpha=config.get('MEIR_ALPHA', 0.5), cift_enabled=config.get('CIFT_ENABLED', False), cift_C=config.get('CIFT_C', 1.0), cift_floor=config.get('CIFT_FLOOR', 0.1))

def print_training_header(config: Dict):
    print('\n' + '=' * 80)
    print('MAPPO RCB-RSQ - Return-Conditioned Beta + Reward Signal Quality')
    print('=' * 80)
    print(f"Environment: {config['ENV_NAME']}")
    print(f"Seed: {config['SEED']}")
    print(f"Total Timesteps: {config['TOTAL_TIMESTEPS']:,.0f}")
    print(f"RCB: beta_max={config.get('RCB_BETA_MAX')}, beta_min={config.get('RCB_BETA_MIN')}, kappa={config.get('RCB_KAPPA')}, target={config.get('RCB_TARGET')}, decay_start={config.get('RCB_DECAY_START', 1.0)}")
    alloc_mode = config.get('ALLOCATION_MODE', 'affine')
    print(f"RSQ: lambda={config.get('RSQ_LAMBDA')}, ref={config.get('RSQ_REF')}, h=[{config.get('RSQ_H_MIN')}, {config.get('RSQ_H_MAX')}], alloc={alloc_mode}")
    if alloc_mode == 'meir':
        print(f"MEIR: alpha={config.get('MEIR_ALPHA', 0.5)}")
    if config.get('CIFT_ENABLED', False):
        print(f"CIFT: C={config.get('CIFT_C', 1.0)}, floor={config.get('CIFT_FLOOR', 0.1)}")
    ir_type = config.get('INTRINSIC_REWARD_TYPE', 'tdd')
    print(f'Intrinsic Reward Type: {ir_type}')
    adaptive_mode = config.get('ADAPTIVE_TARGET_MODE', 'none')
    if adaptive_mode != 'none':
        print(f"Adaptive Target: mode={adaptive_mode}, percentile={config.get('TARGET_PERCENTILE', 50)}, history={config.get('TARGET_HISTORY_SIZE', 100)}, ratio={config.get('RUNNING_MAX_RATIO', 0.8)}")
    print('=' * 80 + '\n')

def make_train(config):
    env_kwargs = config.get('ENV_KWARGS', {})
    env_name = config['ENV_NAME']
    if 'corridor' in env_name.lower() and 'SMAX' not in env_name:
        env_kwargs['sparse_reward'] = config.get('SPARSE_REWARD', False)
    env = jaxmarl.make(env_name, **env_kwargs)
    num_learning_agents = len(env.agents)
    config['NUM_ACTORS'] = num_learning_agents * config['NUM_ENVS']
    config['NUM_UPDATES'] = config['TOTAL_TIMESTEPS'] // config['NUM_STEPS'] // config['NUM_ENVS']
    tc = create_trainer_config(config, num_learning_agents)
    is_smax = 'SMAX' in env_name or 'smax' in env_name
    if is_smax:
        env = SMAXWorldStateWrapper(env)
        env = LogWrapper(env)
    else:
        env = MPEWorldStateWrapper(env)
        env = MPELogWrapper(env)
    actor_lr_schedule = None
    critic_lr_schedule = None
    _obs_dim_static = env.observation_space(env.agents[0]).shape[0]
    if is_smax:
        sd_pos_start = config.get('SD_POS_START', _obs_dim_static - 9)
        sd_pos_end = config.get('SD_POS_END', _obs_dim_static - 7)
    else:
        sd_pos_start = config.get('SD_POS_START', 2)
        sd_pos_end = config.get('SD_POS_END', 4)
    print(f'[RCB-RSQ] SD position slice: obs[{sd_pos_start}:{sd_pos_end}] (obs_dim={_obs_dim_static})')
    sd_feature_mode = config.get('SD_FEATURE_MODE', 'position')
    combat_mode_active = sd_feature_mode == 'combat' and is_smax
    if combat_mode_active:
        smax_inner = env
        while hasattr(smax_inner, '_env'):
            smax_inner = smax_inner._env
        smax_num_movement_actions = smax_inner.num_movement_actions
        smax_action_dim_raw = env.action_space(env.agents[0]).n
        smax_num_enemies = smax_inner.num_agents - num_learning_agents
        smax_map_width = smax_inner.map_width
        smax_map_height = smax_inner.map_height
        scenario_unit_types = smax_inner.scenario if smax_inner.scenario is not None else jnp.zeros((smax_inner.num_agents,), dtype=jnp.uint8)
        _max_health_per_unit = smax_inner.unit_type_health[scenario_unit_types]
        _max_cooldown_per_unit = smax_inner.unit_type_weapon_cooldowns[scenario_unit_types]
        smax_ally_max_health = _max_health_per_unit[:num_learning_agents]
        smax_ally_max_cooldown = _max_cooldown_per_unit[:num_learning_agents]
        sd_combat_input_dim = 2 + 1 + 1 + smax_num_enemies + smax_num_enemies
        print(f'[RCB-RSQ] SD feature mode: COMBAT (dim={sd_combat_input_dim}, num_enemies={smax_num_enemies})')
    else:
        sd_combat_input_dim = None
        print(f'[RCB-RSQ] SD feature mode: {sd_feature_mode}')

    def train(rng):
        (rng, _rng_actor, _rng_critic, _rng_tdd) = jax.random.split(rng, 4)
        obs_dim = env.observation_space(env.agents[0]).shape[0]
        action_dim = env.action_space(env.agents[0]).n
        (_, _, actor_networks, actor_train_states) = create_actor_train_state(rng=_rng_actor, config=config, action_dim=action_dim, obs_dim=obs_dim, num_agents=tc.num_agents, lr_schedule=actor_lr_schedule, shared_params=False)
        use_centralized_critic = config.get('USE_CENTRALIZED_CRITIC', True)
        critic_input_size = env.world_state_size() if use_centralized_critic else obs_dim
        (critic_network, critic_train_state) = create_critic_train_state(rng=_rng_critic, config=config, world_state_size=critic_input_size, lr_schedule=critic_lr_schedule)
        sd_input_dim = sd_combat_input_dim if combat_mode_active else config.get('SD_INPUT_DIM', 2)
        (tdd_potential_states_raw, tdd_s_encoder_states_raw) = create_tdd_train_states(rng=_rng_tdd, config=config, num_agents=tc.num_agents, input_dim=sd_input_dim)
        tdd_potential_states = tuple(tdd_potential_states_raw)
        tdd_s_encoder_states = tuple(tdd_s_encoder_states_raw)
        _rng_aux = jax.random.PRNGKey(42)
        (_rng_aux, _rng_rnd) = jax.random.split(_rng_aux)
        rnd_output_dim = config.get('RND_OUTPUT_DIM', 32)
        rnd_hidden_dim = config.get('RND_HIDDEN_DIM', 64)
        rnd_net = RNDNetwork(hidden_dim=rnd_hidden_dim, output_dim=rnd_output_dim)
        rnd_dummy = jnp.zeros((1, obs_dim))
        (_rng_rnd_fixed, _rng_rnd_pred) = jax.random.split(_rng_rnd)
        rnd_fixed_params = rnd_net.init(_rng_rnd_fixed, rnd_dummy)
        rnd_pred_params_init = rnd_net.init(_rng_rnd_pred, rnd_dummy)
        rnd_pred_optimizer = optax.adam(config.get('RND_LR', 0.001))
        rnd_pred_opt_state = rnd_pred_optimizer.init(rnd_pred_params_init)
        critic_mode = 'centralized (MAPPO)' if use_centralized_critic else 'independent (IPPO)'
        print(f'[RCB-RSQ] Critic: {critic_mode}, input_dim={critic_input_size}')
        count_hash_dim = config.get('COUNT_HASH_DIM', 8)
        count_table_size = 2 ** count_hash_dim
        (_rng_aux, _rng_hash) = jax.random.split(_rng_aux)
        count_hash_matrix = jax.random.normal(_rng_hash, (obs_dim, count_hash_dim))
        print(f'[RCB-RSQ] RCB: beta_max={tc.rcb_beta_max}, beta_min={tc.rcb_beta_min}, kappa={tc.rcb_kappa}, target={tc.rcb_target}')
        print(f'[RCB-RSQ] RSQ: lambda={tc.rsq_lambda}, ref={tc.rsq_ref}, h=[{tc.rsq_h_min}, {tc.rsq_h_max}], ema_alpha={tc.rsq_ema_alpha}')
        (rng, _rng) = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, tc.num_envs)
        (obsv, env_state) = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        ac_init_hstate = ScannedRNN.initialize_carry(tc.num_actors, config['GRU_HIDDEN_DIM'])
        cr_init_hstate = ScannedRNN.initialize_carry(tc.num_actors, config['GRU_HIDDEN_DIM'])
        raw_env = env
        while hasattr(raw_env, '_env'):
            raw_env = raw_env._env
        video_steps = compute_video_steps(tc.num_updates, tc.warmup_rollouts, config.get('NUM_INTERMEDIATE_VIDEOS', 6))
        callback_handler = CallbackHandler(config=config, env=raw_env, actor_network=actor_networks, critic_network=critic_network, shared_actor_params=False, num_agents=tc.num_agents, video_steps=video_steps)
        agent_idx_array = jnp.repeat(jnp.arange(tc.num_agents), tc.num_envs)
        ir_mu_init = jnp.zeros(tc.num_agents)
        ir_var_init = jnp.ones(tc.num_agents)
        return_history_init = jnp.zeros(tc.target_history_size)
        running_max_init = jnp.float32(0.0)

        def _update_step(update_runner_state, unused):
            (runner_state, update_steps, return_ema, ir_mu, ir_var, return_history, running_max_ema, rnd_pred_params, rnd_pred_opt_state) = update_runner_state
            current_ent_coef = jax.lax.cond(tc.anneal_ent, lambda : tc.ent_coef - (tc.ent_coef - tc.ent_coef_end) * (update_steps / tc.num_updates), lambda : jnp.float32(tc.ent_coef))

            def _env_step(runner_state, unused):
                (train_states, env_state, last_obs, last_done, hstates, rng) = runner_state
                (actor_train_states_local, critic_train_state_local, _, _) = train_states
                (rng, _rng) = jax.random.split(rng)
                obs_batch = batchify(last_obs, env.agents, tc.num_actors)
                actor_template = actor_networks[0]
                stacked_actor_params = stack_params([s.params for s in actor_train_states_local])
                obs_reshaped = obs_batch.reshape(tc.num_agents, tc.num_envs, -1)
                done_reshaped = last_done.reshape(tc.num_agents, tc.num_envs)
                hstate_reshaped = hstates[0].reshape(tc.num_agents, tc.num_envs, -1)

                def _forward_single_actor(params, hidden, obs, done):
                    ac_in = (obs[np.newaxis, :], done[np.newaxis, :])
                    (new_hidden, _, logits) = actor_template.apply(params, hidden, ac_in)
                    return (new_hidden, logits.squeeze(0))
                (ac_hstate_per_agent, action_logits_per_agent) = jax.vmap(_forward_single_actor)(stacked_actor_params, hstate_reshaped, obs_reshaped, done_reshaped)
                ac_hstate = ac_hstate_per_agent.reshape(tc.num_actors, -1)
                action_logits = action_logits_per_agent.reshape(tc.num_actors, -1)
                pi = distrax.Categorical(logits=action_logits)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(action, env.agents, tc.num_envs, tc.num_agents)
                if is_smax:
                    env_act = {a: v.squeeze(-1) for (a, v) in env_act.items()}
                world_state = last_obs['world_state'].swapaxes(0, 1).reshape((tc.num_actors, -1))
                critic_input = obs_batch if not use_centralized_critic else world_state
                cr_in = (critic_input[None, :], last_done[np.newaxis, :])
                (cr_hstate, value) = critic_network.apply(critic_train_state_local.params, hstates[1], cr_in)
                (rng, _rng) = jax.random.split(rng)
                rng_step = jax.random.split(_rng, tc.num_envs)
                (obsv, env_state, reward, done, info) = jax.vmap(env.step, in_axes=(0, 0, 0))(rng_step, env_state, env_act)
                info = jax.tree.map(lambda x: x.reshape(tc.num_actors), info)
                done_batch = batchify(done, env.agents, tc.num_actors).squeeze()
                if combat_mode_active:
                    smax_state = env_state.env_state.state
                    ally_positions = smax_state.unit_positions[:, :tc.num_agents, :]
                    ally_positions_norm = ally_positions / jnp.array([smax_map_width, smax_map_height])
                    ally_health_norm = smax_state.unit_health[:, :tc.num_agents] / smax_ally_max_health[None, :]
                    ally_cd_norm = smax_state.unit_weapon_cooldowns[:, :tc.num_agents] / (smax_ally_max_cooldown[None, :] + 1e-06)
                    prev_atk = smax_state.prev_attack_actions[:, :tc.num_agents]
                    prev_atk_oh = jax.nn.one_hot(prev_atk, num_classes=smax_action_dim_raw)[:, :, smax_num_movement_actions:]
                    action_reshaped = action.reshape(tc.num_agents, tc.num_envs).swapaxes(0, 1)
                    cur_atk_oh = jax.nn.one_hot(action_reshaped, num_classes=smax_action_dim_raw)[:, :, smax_num_movement_actions:]
                    agent_features_combat = jnp.concatenate([ally_positions_norm, ally_health_norm[..., None], jnp.clip(ally_cd_norm[..., None], 0.0, 2.0), prev_atk_oh, cur_atk_oh], axis=-1)
                    sd_features_step = agent_features_combat.swapaxes(0, 1).reshape(tc.num_actors, sd_combat_input_dim)
                    agent_positions = ally_positions_norm.swapaxes(0, 1).reshape(tc.num_actors, 2)
                else:
                    agent_positions = obs_batch[:, sd_pos_start:sd_pos_end]
                    sd_features_step = agent_positions
                transition = Transition(jnp.tile(done['__all__'], tc.num_agents), last_done, action.squeeze(), action_logits, value.squeeze(), batchify(reward, env.agents, tc.num_actors).squeeze(), log_prob.squeeze(), obs_batch, critic_input, info, z_i=jnp.zeros((tc.num_actors,)), intrinsic_reward=None, agent_positions=agent_positions, agent_local_features_curr=sd_features_step, agent_local_features_next=None, agent_idx=agent_idx_array)
                runner_state = (train_states, env_state, obsv, done_batch, (ac_hstate, cr_hstate), rng)
                return (runner_state, transition)
            initial_hstates = runner_state[-2]
            (runner_state, traj_batch) = jax.lax.scan(_env_step, runner_state, None, tc.num_steps)
            (train_states, env_state, last_obs, last_done, hstates, rng) = runner_state
            (actor_train_states, critic_train_state, tdd_potential_states, tdd_s_encoder_states) = train_states
            obs_all = traj_batch.obs
            if combat_mode_active:
                agent_features_curr_all = traj_batch.agent_local_features_curr.reshape(tc.num_steps, tc.num_agents, tc.num_envs, sd_combat_input_dim).transpose(1, 0, 2, 3)
            else:
                obs_reshaped = obs_all.reshape(tc.num_steps, tc.num_agents, tc.num_envs, -1)
                agent_features_curr_all = obs_reshaped[..., sd_pos_start:sd_pos_end].transpose(1, 0, 2, 3)
            agent_features_next_all = jnp.concatenate([agent_features_curr_all[:, 1:, :, :], agent_features_curr_all[:, -1:, :, :]], axis=1)
            (rng, _rng_tdd) = jax.random.split(rng)

            def _tdd_update_step(tdd_states_arg, rng_arg):
                (potential_states_inner, s_encoder_states_inner) = tdd_states_arg
                batch_size = tc.tdd_batch_size
                I = jnp.eye(batch_size)
                agent_rngs = jax.random.split(rng_arg, tc.num_agents)
                potential_apply_fn = potential_states_inner[0].apply_fn
                s_encoder_apply_fn = s_encoder_states_inner[0].apply_fn
                stacked_potential_params = stack_params([s.params for s in potential_states_inner])
                stacked_s_encoder_params = stack_params([s.params for s in s_encoder_states_inner])

                def _tdd_loss_single_agent(potential_params, s_encoder_params, agent_features, agent_rng):
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
                    c_g = potential_apply_fn(potential_params, goal)
                    phi_s = s_encoder_apply_fn(s_encoder_params, obs)
                    phi_g = s_encoder_apply_fn(s_encoder_params, goal)
                    phi_s_expanded = phi_s[:, None, :]
                    phi_g_expanded = phi_g[None, :, :]
                    mrn_dists = mrn_distance(phi_s_expanded, phi_g_expanded)
                    c_g_squeezed = c_g.squeeze(-1)
                    logits = c_g_squeezed[None, :] - mrn_dists
                    log_softmax_logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)
                    loss1_per_sample = -(log_softmax_logits * I).sum(axis=-1)
                    loss1 = jnp.sum(loss1_per_sample * valid_mask_float) / jnp.maximum(num_valid, 1)
                    log_softmax_logits_T = logits.T - jax.scipy.special.logsumexp(logits.T, axis=-1, keepdims=True)
                    loss2_per_sample = -(log_softmax_logits_T * I).sum(axis=-1)
                    loss2 = jnp.sum(loss2_per_sample * valid_mask_float) / jnp.maximum(num_valid, 1)
                    return (loss1 + loss2) / 2

                def _compute_loss_and_grad(pot_params, enc_params, features, rng):
                    (loss, (pot_grad, enc_grad)) = jax.value_and_grad(_tdd_loss_single_agent, argnums=(0, 1))(pot_params, enc_params, features, rng)
                    return (loss, pot_grad, enc_grad)
                (losses, potential_grads, s_encoder_grads) = jax.vmap(_compute_loss_and_grad)(stacked_potential_params, stacked_s_encoder_params, agent_features_curr_all, agent_rngs)
                new_potential_states_list = vectorized_apply_gradients(list(potential_states_inner), potential_grads)
                new_s_encoder_states_list = vectorized_apply_gradients(list(s_encoder_states_inner), s_encoder_grads)
                total_loss = losses.mean()
                return ((tuple(new_potential_states_list), tuple(new_s_encoder_states_list)), total_loss)

            def _tdd_update_epoch(carry, unused):
                (tdd_states_inner, rng_inner) = carry
                (rng_inner, _rng_step) = jax.random.split(rng_inner)
                (new_tdd_states_inner, loss) = _tdd_update_step(tdd_states_inner, _rng_step)
                return ((new_tdd_states_inner, rng_inner), loss)
            init_tdd_states = (tdd_potential_states, tdd_s_encoder_states)

            def _train_tdd():
                ((new_states, _), losses) = jax.lax.scan(_tdd_update_epoch, (init_tdd_states, _rng_tdd), None, tc.tdd_update_epochs)
                return (new_states, losses.mean())

            def _skip_tdd():
                return (init_tdd_states, 0.0)
            (new_tdd_states, tdd_loss) = jax.lax.cond(tc.intrinsic_reward_type == 0, _train_tdd, _skip_tdd)
            tdd_potential_states = tuple(new_tdd_states[0])
            tdd_s_encoder_states = tuple(new_tdd_states[1])

            def _train_rnd():
                obs_flat = obs_all.reshape(-1, obs_all.shape[-1])
                fixed_out = rnd_net.apply(rnd_fixed_params, obs_flat)

                def _rnd_loss_fn(pred_params):
                    pred_out = rnd_net.apply(pred_params, obs_flat)
                    return jnp.mean((pred_out - jax.lax.stop_gradient(fixed_out)) ** 2)
                (loss, grads) = jax.value_and_grad(_rnd_loss_fn)(rnd_pred_params)
                (updates, new_opt_state) = rnd_pred_optimizer.update(grads, rnd_pred_opt_state)
                new_params = optax.apply_updates(rnd_pred_params, updates)
                return (new_params, new_opt_state)

            def _skip_rnd():
                return (rnd_pred_params, rnd_pred_opt_state)
            (new_rnd_pred_params, new_rnd_pred_opt_state) = jax.lax.cond(tc.intrinsic_reward_type == 2, _train_rnd, _skip_rnd)
            tdd_intrinsic = compute_intrinsic_reward_sd(agent_features_curr_all, agent_features_next_all, tdd_s_encoder_states, tdd_shared=False, num_agents=tc.num_agents, num_envs=tc.num_envs, num_steps=tc.num_steps)
            ent_pi = distrax.Categorical(logits=traj_batch.action_logits)
            ent_all = ent_pi.entropy()
            ent_per_agent = ent_all.reshape(tc.num_steps, tc.num_agents, tc.num_envs).transpose(1, 0, 2)
            obs_flat_rnd = obs_all.reshape(-1, obs_all.shape[-1])
            rnd_fixed_out = rnd_net.apply(rnd_fixed_params, obs_flat_rnd)
            rnd_pred_out = rnd_net.apply(new_rnd_pred_params, obs_flat_rnd)
            rnd_error = jnp.mean((rnd_pred_out - rnd_fixed_out) ** 2, axis=-1)
            rnd_per_agent = rnd_error.reshape(tc.num_steps, tc.num_agents, tc.num_envs).transpose(1, 0, 2)
            obs_for_count = obs_all.reshape(tc.num_steps, tc.num_agents, tc.num_envs, -1).transpose(1, 2, 0, 3)
            hash_codes = (obs_for_count @ count_hash_matrix > 0).astype(jnp.int32)
            powers = 2 ** jnp.arange(count_hash_dim)
            hash_ints = (hash_codes * powers[None, None, None, :]).sum(-1) % count_table_size

            def _count_single(hashes):
                table = jnp.zeros(count_table_size, dtype=jnp.int32)
                table = table.at[hashes].add(1)
                return 1.0 / jnp.sqrt(table[hashes].astype(jnp.float32))
            count_rewards = jax.vmap(jax.vmap(_count_single))(hash_ints)
            count_per_agent = count_rewards.transpose(0, 2, 1)
            intrinsic_reward_per_agent = jax.lax.switch(tc.intrinsic_reward_type, [lambda : tdd_intrinsic, lambda : ent_per_agent, lambda : rnd_per_agent, lambda : count_per_agent])
            intrinsic_reward_all = intrinsic_reward_per_agent.transpose(1, 0, 2).reshape(tc.num_steps, tc.num_actors)
            intrinsic_reward_raw = intrinsic_reward_all
            r_int_scaled = intrinsic_reward_all * tc.intrinsic_reward_scale
            ep_returns = traj_batch.info['returned_episode_returns']
            ep_dones = traj_batch.info['returned_episode']
            mean_return = jnp.where(ep_dones.any(), jnp.sum(ep_returns * ep_dones) / jnp.maximum(ep_dones.sum(), 1), return_ema)
            return_ema = tc.rcb_ema_alpha * mean_return + (1 - tc.rcb_ema_alpha) * return_ema
            r_ext = traj_batch.reward.astype(jnp.float32)
            use_intrinsic = update_steps >= tc.warmup_rollouts
            progress = update_steps / tc.num_updates
            decay_factor = jnp.where(progress > tc.rcb_decay_start, jnp.clip(1.0 - (progress - tc.rcb_decay_start) / (1.0 - tc.rcb_decay_start + 1e-08), 0.0, 1.0), 1.0)
            effective_beta_max = tc.rcb_beta_min + (tc.rcb_beta_max - tc.rcb_beta_min) * decay_factor
            return_history = jnp.concatenate([return_history[1:], jnp.array([mean_return])])
            running_max_ema = jnp.maximum(running_max_ema, mean_return) * 0.999 + mean_return * 0.001
            adaptive_target = jax.lax.switch(tc.adaptive_target_mode, [lambda : tc.rcb_target, lambda : jnp.percentile(return_history, tc.target_percentile), lambda : tc.running_max_ratio * running_max_ema])
            sigmoid_val = jax.nn.sigmoid(tc.rcb_kappa * (adaptive_target - return_ema))
            adaptive_beta = tc.rcb_beta_min + (effective_beta_max - tc.rcb_beta_min) * sigmoid_val
            batch_mean = jnp.mean(intrinsic_reward_per_agent, axis=(1, 2))
            batch_var = jnp.var(intrinsic_reward_per_agent, axis=(1, 2))
            ir_mu_new = tc.rsq_ema_alpha * batch_mean + (1 - tc.rsq_ema_alpha) * ir_mu
            ir_var_new = tc.rsq_ema_alpha * batch_var + (1 - tc.rsq_ema_alpha) * ir_var
            mu_sq = ir_mu_new ** 2
            rsq = mu_sq / (mu_sq + ir_var_new + 1e-08)

            def _waterfilling_g(snr_vals, beta_global, n_agents):
                inv_snr = 1.0 / (snr_vals + 1e-08)
                B = n_agents * beta_global ** 2
                sorted_idx = jnp.argsort(inv_snr)
                inv_snr_sorted = inv_snr[sorted_idx]
                cumsum_inv = jnp.cumsum(inv_snr_sorted)
                ks = jnp.arange(1, n_agents + 1, dtype=jnp.float32)
                W_candidates = (B + cumsum_inv) / ks
                valid = W_candidates > inv_snr_sorted
                k_star = jnp.sum(valid.astype(jnp.int32))
                k_star = jnp.maximum(k_star, 1)
                W = (B + cumsum_inv[k_star - 1]) / k_star
                p_wf = jnp.maximum(W - inv_snr, 0.0)
                beta_wf = jnp.sqrt(p_wf)
                g_wf = beta_wf / (beta_global + 1e-08)
                return g_wf
            snr = mu_sq / (ir_var_new + 1e-08)

            def _affine_g():
                effective_rsq_ref = jax.lax.cond(tc.rsq_ref < 0, lambda : jnp.mean(rsq), lambda : jnp.float32(tc.rsq_ref))
                return jnp.clip(1.0 + tc.rsq_lambda * (rsq - effective_rsq_ref), tc.rsq_h_min, tc.rsq_h_max)

            def _wf_g():
                return _waterfilling_g(snr, adaptive_beta, tc.num_agents)

            def _meir_g():
                snr_powered = snr ** tc.meir_alpha
                g_raw = snr_powered / (jnp.mean(snr_powered) + 1e-08)
                return jnp.clip(g_raw, tc.rsq_h_min, tc.rsq_h_max)
            g = jax.lax.switch(tc.allocation_mode, [_affine_g, _wf_g, _meir_g])

            def _apply_cift(beta_in):
                agent_cv = jnp.std(ir_mu_new) / (jnp.mean(jnp.abs(ir_mu_new)) + 1e-08)
                beta_cift = tc.cift_C / (jnp.sqrt(jnp.float32(tc.num_agents)) * jnp.maximum(agent_cv, tc.cift_floor))
                return jnp.minimum(beta_in, beta_cift)
            adaptive_beta = jax.lax.cond(tc.cift_enabled, _apply_cift, lambda b: b, adaptive_beta)
            g_broadcast = jnp.repeat(g, tc.num_envs)
            g_broadcast = jnp.broadcast_to(g_broadcast, (tc.num_steps, tc.num_actors))
            r_int_modulated = r_int_scaled * g_broadcast
            r_combined = jax.lax.cond(use_intrinsic, lambda : r_ext + adaptive_beta * r_int_modulated, lambda : r_ext)
            traj_batch = traj_batch._replace(intrinsic_reward=r_int_scaled)
            last_world_state = last_obs['world_state'].swapaxes(0, 1).reshape((tc.num_actors, -1))
            last_critic_input = batchify(last_obs, env.agents, tc.num_actors) if not use_centralized_critic else last_world_state
            cr_in = (last_critic_input[None, :], last_done[np.newaxis, :])
            (_, last_val) = critic_network.apply(critic_train_state.params, hstates[1], cr_in)
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch_arg, last_val_arg, combined_rewards):

                def _get_advantages(gae_and_next_value, inputs):
                    (gae, next_value) = gae_and_next_value
                    (done, value, reward) = inputs
                    (next_value, value) = (next_value.reshape(-1), value.reshape(-1))
                    delta = reward + tc.gamma * next_value * (1 - done) - value
                    gae = delta + tc.gamma * tc.gae_lambda * (1 - done) * gae
                    return ((gae, value), gae)
                inputs = (traj_batch_arg.global_done, traj_batch_arg.value, combined_rewards)
                (_, advantages) = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val_arg), last_val_arg), inputs, reverse=True, unroll=16)
                return (advantages, advantages + traj_batch_arg.value)
            (advantages, targets) = _calculate_gae(traj_batch, last_val, r_combined)

            def _update_epoch(update_state, unused):
                (train_states_upd, init_hstates, traj_batch_upd, advantages_upd, targets_upd, rng_upd) = update_state
                (rng_upd, _rng) = jax.random.split(rng_upd)
                (actor_train_states_upd, critic_train_state_upd) = train_states_upd

                def to_per_agent_3d(x):
                    return x.reshape(tc.num_steps, tc.num_agents, tc.num_envs).transpose(1, 0, 2)

                def to_per_agent_4d(x):
                    return x.reshape(tc.num_steps, tc.num_agents, tc.num_envs, -1).transpose(1, 0, 2, 3)
                agent_obs = to_per_agent_4d(traj_batch_upd.obs)
                agent_done = to_per_agent_3d(traj_batch_upd.done)
                agent_action = to_per_agent_3d(traj_batch_upd.action)
                agent_log_prob = to_per_agent_3d(traj_batch_upd.log_prob)
                agent_gae = to_per_agent_3d(advantages_upd)
                agent_init_hstate = init_hstates[0].reshape(tc.num_agents, tc.num_envs, -1)
                actor_template = actor_networks[0]
                stacked_actor_params = stack_params([s.params for s in actor_train_states_upd])

                def _actor_loss_single_agent(actor_params, obs, done, action, log_prob, gae, init_hstate, agent_rng):
                    perm = jax.random.permutation(agent_rng, tc.num_envs)
                    obs_shuffled = jnp.take(obs, perm, axis=1)
                    done_shuffled = jnp.take(done, perm, axis=1)
                    action_shuffled = jnp.take(action, perm, axis=1)
                    log_prob_shuffled = jnp.take(log_prob, perm, axis=1)
                    gae_shuffled = jnp.take(gae, perm, axis=1)
                    hstate_shuffled = jnp.take(init_hstate, perm, axis=0)
                    (_, pi, _) = actor_template.apply(actor_params, hstate_shuffled, (obs_shuffled, done_shuffled))
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

                def _compute_actor_grad(params, obs, done, action, log_prob, gae, init_hstate, rng):
                    ((loss, aux), grad) = jax.value_and_grad(_actor_loss_single_agent, has_aux=True)(params, obs, done, action, log_prob, gae, init_hstate, rng)
                    return (loss, aux, grad)
                agent_rngs = jax.random.split(_rng, tc.num_agents)
                (actor_losses, actor_aux, actor_grads) = jax.vmap(_compute_actor_grad)(stacked_actor_params, agent_obs, agent_done, agent_action, agent_log_prob, agent_gae, agent_init_hstate, agent_rngs)
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
                actor_grad_norms = jax.vmap(optax.global_norm)(actor_grads)
                critic_grad_norm = optax.global_norm(critic_grads)
                loss_info = {'total_loss': total_loss, 'actor_loss': actor_loss_mean, 'value_loss': critic_loss[0], 'entropy': entropy_mean, 'ratio': ratio_mean, 'approx_kl': approx_kl_mean, 'clip_frac': clip_frac_mean, 'actor_grad_norm': actor_grad_norms.mean(), 'critic_grad_norm': critic_grad_norm}
                train_states_upd = (new_actor_states, critic_train_state_upd)
                update_state = (train_states_upd, init_hstates, traj_batch_upd, advantages_upd, targets_upd, rng_upd)
                return (update_state, loss_info)
            train_states = (actor_train_states, critic_train_state)
            update_state = (train_states, initial_hstates, traj_batch, advantages, targets, rng)
            (update_state, loss_info) = jax.lax.scan(_update_epoch, update_state, None, tc.update_epochs)
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            train_states = update_state[0]
            (actor_train_states, critic_train_state) = train_states
            metric = traj_batch.info
            metric['loss'] = loss_info
            metric['tdd_loss'] = tdd_loss
            metric['linear'] = {'beta': adaptive_beta, 'r_ext_mean': jnp.mean(r_ext), 'r_int_mean': jnp.mean(r_int_scaled), 'r_combined_mean': jnp.mean(r_combined)}
            metric['rcb'] = {'return_ema': return_ema, 'adaptive_beta': adaptive_beta, 'mean_return': mean_return, 'sigmoid_val': sigmoid_val}
            p_i = (adaptive_beta * g) ** 2
            B_i = mu_sq + ir_var_new
            meir_cost = jnp.sum(p_i * B_i)
            meir_info = jnp.sum(0.5 * jnp.log1p(p_i * snr))
            gamma_team = meir_cost / (meir_info + 1e-08)
            metric['rsq'] = {'rsq_mean': jnp.mean(rsq), 'rsq_min': jnp.min(rsq), 'rsq_max': jnp.max(rsq), 'g_mean': jnp.mean(g), 'g_min': jnp.min(g), 'g_max': jnp.max(g), 'ir_mu_mean': jnp.mean(ir_mu_new), 'ir_var_mean': jnp.mean(ir_var_new), 'gamma_team': gamma_team, '_per_agent/rsq': rsq, '_per_agent/mu': ir_mu_new, '_per_agent/sigma_sq': ir_var_new, '_per_agent/g_individual': g, '_per_agent/g_final': g}
            metric['rewards'] = {'extrinsic': jnp.mean(r_ext), 'intrinsic_scaled': jnp.mean(r_int_scaled), 'combined': jnp.mean(r_combined)}
            metric['ent_coef'] = current_ent_coef
            metric['action_logits'] = traj_batch.action_logits
            metric['intrinsic_reward_scaled'] = traj_batch.intrinsic_reward
            metric['intrinsic_reward_raw'] = intrinsic_reward_raw
            metric['agent_positions'] = traj_batch.agent_positions
            (metric['num_agents'], metric['num_envs']) = (tc.num_agents, tc.num_envs)
            metric['intrinsic_reward_scale'] = tc.intrinsic_reward_scale
            metric['actor_params'] = [s.params for s in actor_train_states]
            metric['critic_params'] = critic_train_state.params
            metric['s_encoder_params'] = tdd_s_encoder_states[0].params
            metric['tdd_potential_params'] = [s.params for s in tdd_potential_states]
            metric['tdd_s_encoder_params'] = [s.params for s in tdd_s_encoder_states]
            rng = update_state[-1]
            metric['update_steps'] = update_steps
            jax.experimental.io_callback(callback_handler, None, metric, ordered=True)
            update_steps = update_steps + 1
            train_states = (actor_train_states, critic_train_state, tdd_potential_states, tdd_s_encoder_states)
            runner_state = (train_states, env_state, last_obs, last_done, hstates, rng)
            return ((runner_state, update_steps, return_ema, ir_mu_new, ir_var_new, return_history, running_max_ema, new_rnd_pred_params, new_rnd_pred_opt_state), metric)
        (rng, _rng) = jax.random.split(rng)
        train_states = (actor_train_states, critic_train_state, tdd_potential_states, tdd_s_encoder_states)
        runner_state = (train_states, env_state, obsv, jnp.zeros(tc.num_actors, dtype=bool), (ac_init_hstate, cr_init_hstate), _rng)
        (runner_state, metric) = jax.lax.scan(_update_step, (runner_state, 0, jnp.float32(0.0), ir_mu_init, ir_var_init, return_history_init, running_max_init, rnd_pred_params_init, rnd_pred_opt_state), None, tc.num_updates)
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

@hydra.main(version_base=None, config_path='config', config_name='rcb_rsq_corridor')
def main(config):
    config = OmegaConf.to_container(config)
    print_training_header(config)
    wandb_project = config.get('PROJECT', 'mappo-rcb-rsq')
    env_display_name = config['ENV_NAME'].replace('/', '-').replace('_', '-')
    run_name = config.get('RUN_NAME') or f'mappo-rcb-rsq-{env_display_name}'
    run = wandb.init(project=wandb_project, name=run_name, tags=['mappo', 'rcb-rsq', config['ENV_NAME']], config=config, mode=config['WANDB_MODE'])
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
