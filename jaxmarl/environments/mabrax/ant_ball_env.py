"""Ant with ball-on-tray: 4-legged ant must walk forward while keeping a ball on a tray.

The ball falling off the tray constitutes a catastrophic failure.
This environment demonstrates that smooth, coordinated locomotion (not just speed)
is critical when agents share a fragile equilibrium constraint.
"""

from pathlib import Path

from brax import base, math
from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
from jax import numpy as jp
import mujoco


# Ball body is the last body in the model (index depends on XML structure)
# Torso is body index 0 in the worldbody
_TORSO_IDX = 0


class AntBall(PipelineEnv):
    """Ant that carries a ball on a tray attached to its torso.

    Reward: forward_velocity + healthy_reward + ball_on_tray_bonus - ctrl_cost
    Done: ant unhealthy OR ball fallen off tray (catastrophic failure)
    """

    def __init__(
        self,
        ctrl_cost_weight=0.5,
        healthy_reward=1.0,
        ball_on_tray_bonus=3.0,
        forward_reward_weight=1.0,
        ball_drop_penalty=0.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        ball_fall_threshold=0.08,
        ball_xy_threshold=0.15,
        reset_noise_scale=0.1,
        backend='positional',
        **kwargs,
    ):
        # Pop MABraxEnv-level kwargs that PipelineEnv doesn't accept
        kwargs.pop('episode_length', None)
        kwargs.pop('action_repeat', None)
        kwargs.pop('auto_reset', None)

        xml_path = Path(__file__).parent / 'ant_ball.xml'
        sys = mjcf.load(str(xml_path))

        n_frames = 5
        if backend in ['spring', 'positional']:
            sys = sys.tree_replace({'opt.timestep': 0.005})
            n_frames = 10

        if backend == 'positional':
            sys = sys.replace(
                actuator=sys.actuator.replace(
                    gear=200 * jp.ones_like(sys.actuator.gear)
                )
            )

        kwargs['n_frames'] = kwargs.get('n_frames', n_frames)
        super().__init__(sys=sys, backend=backend, **kwargs)

        self._ctrl_cost_weight = ctrl_cost_weight
        self._healthy_reward = healthy_reward
        self._ball_on_tray_bonus = ball_on_tray_bonus
        self._forward_reward_weight = forward_reward_weight
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._ball_fall_threshold = ball_fall_threshold
        self._ball_drop_penalty = ball_drop_penalty
        self._ball_xy_threshold = ball_xy_threshold
        self._reset_noise_scale = reset_noise_scale

        # Ball is the last link (last body after worldbody in the XML)
        self._ball_idx = sys.num_links() - 1

    def reset(self, rng: jax.Array) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)

        low, hi = -self._reset_noise_scale, self._reset_noise_scale
        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=hi
        )
        # Keep ball position noise smaller to ensure it starts on tray
        ball_q_start = self.sys.q_size() - 7  # last 7 DOF = ball free joint
        q = q.at[ball_q_start:ball_q_start + 3].set(
            self.sys.init_q[ball_q_start:ball_q_start + 3]
            + 0.02 * jax.random.uniform(rng1, (3,), minval=-1, maxval=1)
        )
        # Reset ball quaternion to identity
        q = q.at[ball_q_start + 3:ball_q_start + 7].set(jp.array([1.0, 0.0, 0.0, 0.0]))

        qd = hi * jax.random.normal(rng2, (self.sys.qd_size(),))
        # Zero out ball velocities initially
        ball_qd_start = self.sys.qd_size() - 6
        qd = qd.at[ball_qd_start:].set(0.0)

        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)

        reward, done, zero = jp.zeros(3)
        metrics = {
            'reward_forward': zero,
            'reward_survive': zero,
            'reward_ctrl': zero,
            'reward_ball': zero,
            'ball_on_tray': jp.ones(()),
            'ball_rel_x': zero,
            'ball_rel_y': zero,
            'ball_rel_z': zero,
            'x_position': zero,
            'y_position': zero,
            'x_velocity': zero,
        }
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state0 = state.pipeline_state
        assert pipeline_state0 is not None
        pipeline_state = self.pipeline_step(pipeline_state0, action)

        # --- Locomotion reward (same as ant) ---
        torso_pos = pipeline_state.x.pos[_TORSO_IDX]
        torso_pos0 = pipeline_state0.x.pos[_TORSO_IDX]
        velocity = (torso_pos - torso_pos0) / self.dt
        forward_reward = self._forward_reward_weight * velocity[0]

        # --- Ant health ---
        min_z, max_z = self._healthy_z_range
        ant_healthy = jp.where(torso_pos[2] < min_z, 0.0, 1.0)
        ant_healthy = jp.where(torso_pos[2] > max_z, 0.0, ant_healthy)

        # --- Ball status ---
        ball_pos = pipeline_state.x.pos[self._ball_idx]
        # Ball position relative to torso center (for obs)
        ball_rel = ball_pos - torso_pos
        # Ball position relative to tray center (tray offset +0.10 in x from torso)
        tray_center = torso_pos + jp.array([0.10, 0.0, 0.0])
        ball_rel_tray = ball_pos - tray_center
        ball_xy_dist = jp.sqrt(ball_rel_tray[0] ** 2 + ball_rel_tray[1] ** 2)

        # Ball is "on tray" if it's above the torso and within xy bounds of tray
        ball_on_tray = jp.where(ball_rel[2] < self._ball_fall_threshold, 0.0, 1.0)
        ball_on_tray = jp.where(ball_xy_dist > self._ball_xy_threshold, 0.0, ball_on_tray)

        # --- Rewards ---
        healthy_reward = self._healthy_reward
        ctrl_cost = self._ctrl_cost_weight * jp.sum(jp.square(action))
        ball_reward = self._ball_on_tray_bonus * ball_on_tray
        # Penalty when ball falls off tray (applied on the terminal step)
        drop_penalty = self._ball_drop_penalty * (1.0 - ball_on_tray)

        reward = forward_reward + healthy_reward + ball_reward - ctrl_cost - drop_penalty

        # --- Done ---
        is_healthy = ant_healthy * ball_on_tray
        done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0

        obs = self._get_obs(pipeline_state)
        state.metrics.update(
            reward_forward=forward_reward,
            reward_survive=healthy_reward,
            reward_ctrl=-ctrl_cost,
            reward_ball=ball_reward - drop_penalty,
            ball_on_tray=ball_on_tray,
            ball_rel_x=ball_rel[0],
            ball_rel_y=ball_rel[1],
            ball_rel_z=ball_rel[2],
            x_position=torso_pos[0],
            y_position=torso_pos[1],
            x_velocity=velocity[0],
        )
        return state.replace(
            pipeline_state=pipeline_state, obs=obs, reward=reward, done=done
        )

    def _get_obs(self, pipeline_state: base.State) -> jax.Array:
        """Observe ant body + ball relative position and velocity."""
        # Ant observations (same as standard ant, excluding x,y)
        qpos = pipeline_state.q[2:]  # exclude x, y of torso
        qvel = pipeline_state.qd

        # Ball relative position (relative to torso)
        torso_pos = pipeline_state.x.pos[_TORSO_IDX]
        ball_pos = pipeline_state.x.pos[self._ball_idx]
        ball_rel_pos = ball_pos - torso_pos

        # Ball velocity (absolute, last 6 elements of qd)
        ball_vel = pipeline_state.qd[-6:-3]  # translational velocity only

        return jp.concatenate([qpos, qvel, ball_rel_pos, ball_vel])

    @property
    def observation_size(self):
        # Original ant: 13 qpos (after removing x,y) + 14 qvel = 27
        # With ball: +7 qpos_ball (in qpos already) + 6 qvel_ball (in qvel already)
        # Plus 3 ball_rel_pos + 3 ball_vel = 6 extra
        # Total: q_size - 2 + qd_size + 6
        return self.sys.q_size() - 2 + self.sys.qd_size() + 6
