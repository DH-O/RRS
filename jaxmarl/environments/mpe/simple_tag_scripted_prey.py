"""
SimpleTagScriptedPreyMPE: Simple Tag with Scripted Prey Policy

Based on SimpleFacmacMPE but with DISCRETE actions for adversaries.
Prey uses scripted heuristic policy (from Facmac), making this a
homogeneous environment for adversary-only learning.

Key differences from SimpleTagMPE:
- Only adversaries are learning agents (homogeneous obs, dim depends on config)
- Prey uses scripted escape policy (not learned)
- Discrete action space maintained for adversaries

Key differences from SimpleFacmacMPE:
- Discrete actions instead of continuous
- Prey policy outputs discrete action (converted internally)
"""

import jax
import jax.numpy as jnp
import chex
from typing import Dict
from functools import partial
from jaxmarl.environments.mpe.simple_tag import SimpleTagMPE
from jaxmarl.environments.mpe.simple import State
from jaxmarl.environments.spaces import Box, Discrete
from jaxmarl.environments.mpe.default_params import DISCRETE_ACT


class SimpleTagScriptedPreyMPE(SimpleTagMPE):
    """
    Simple Tag environment where prey uses scripted escape policy.
    Only adversaries are learning agents, making observation space homogeneous.

    Agents: adversary_0, ... (obs_dim = 6 + n_obs*2 + (n_agents-1)*2)
    Scripted: agent_0, ... (prey, uses heuristic escape)

    reward_mode options:
    - "sparse": Original collision-only reward (+10 on catch)
    - "dense": Collision reward + distance shaping (approach prey → positive reward)
    """

    def __init__(
        self,
        num_good_agents=1,
        num_adversaries=3,
        num_obs=2,
        score_function="min",  # "min" or "sum" (from Facmac)
        reward_mode="sparse",  # "sparse" or "dense"
        dense_reward_scale=1.0,  # Scale for distance-based shaping reward
        **kwargs,
    ):
        # Initialize parent (SimpleTagMPE) - this sets up physics, rewards, etc.
        super().__init__(
            num_good_agents=num_good_agents,
            num_adversaries=num_adversaries,
            num_obs=num_obs,
            action_type=DISCRETE_ACT,
            **kwargs,
        )

        # Store original agent info for physics simulation
        # IMPORTANT: Don't override num_agents - physics needs it!
        self._all_agents = list(self.agents)  # ["adversary_0", ..., "agent_0"]
        self._learning_agents = list(self.adversaries)  # ["adversary_0", "adversary_1", "adversary_2"]

        # Override agents to only include adversaries (for MARL interface)
        # But keep num_agents for physics simulation
        self.agents = self._learning_agents

        # Override observation spaces - adversaries only (homogeneous)
        # Formula from simple_tag.py _adversary(): vel(2) + pos(2) + landmarks(n_obs*2) + others((n_agents-1)*2) + last_vel(2)
        # = 6 + num_landmarks * 2 + (num_agents - 1) * 2
        adv_obs_dim = 6 + self.num_landmarks * 2 + (self.num_agents - 1) * 2
        self.observation_spaces = {
            a: Box(-jnp.inf, jnp.inf, (adv_obs_dim,)) for a in self._learning_agents
        }

        # Override action spaces - adversaries only (discrete)
        self.action_spaces = {a: Discrete(5) for a in self._learning_agents}

        # Prey policy configuration (from Facmac)
        self.score_function = score_function

        # Reward configuration
        self.reward_mode = reward_mode
        self.dense_reward_scale = dense_reward_scale

        # For distance shaping (stores previous min distance to prey)
        # Initialized in reset, updated in step
        self._prev_min_dist_to_prey = None

    def set_actions(self, actions: Dict):
        """Override to use _all_agents for physics simulation."""
        # actions dict should have keys for _all_agents
        actions_array = jnp.array([actions[i] for i in self._all_agents]).reshape(
            (self.num_agents, -1)
        )
        return self.action_decoder(self.agent_range, actions_array)

    def _prey_policy(self, key: chex.PRNGKey, state: State, aidx: int) -> jnp.ndarray:
        """
        Scripted prey escape policy (copied from SimpleFacmacMPE).
        Returns continuous action [dx, dy] for physics simulation.

        Args:
            key: Random key
            state: Current environment state
            aidx: Agent index (prey index in _all_agents)

        Returns:
            Continuous action [dx, dy] representing escape direction
        """
        n = 100  # number of positions sampled

        # Sample random directions
        key, _key = jax.random.split(key)
        length = jnp.sqrt(jax.random.uniform(_key, (n,), minval=0., maxval=1.))
        key, _key = jax.random.split(key)
        angle = jnp.pi * jnp.sqrt(jax.random.uniform(_key, (n,), minval=0., maxval=2.))
        x = length * jnp.cos(angle)
        y = length * jnp.sin(angle)

        # Evaluate score for each position
        scores = jnp.zeros(n, dtype=jnp.float32)

        if self.score_function == "sum":
            # Sum of distances to all adversaries
            n_iter = 5
            for i in range(n_iter):
                waypoints_length = (length / float(n_iter)) * (i + 1)
                x_wp = waypoints_length * jnp.cos(angle)
                y_wp = waypoints_length * jnp.sin(angle)
                proj_pos = jnp.vstack((x_wp, y_wp)).transpose() + state.p_pos[aidx]
                delta_pos = state.p_pos[None, :, :] - proj_pos[:, None, :]
                dist = jnp.sqrt(jnp.sum(jnp.square(delta_pos), axis=2))
                dist_min = self.rad + self.rad[aidx]
                # If collision detected, mark as invalid (set to -9999999)
                collision_mask = (dist < dist_min[None]).sum(axis=1) > 0
                scores = jnp.where(collision_mask, -9999999.0, scores)
                if i == n_iter - 1:
                    scores += dist[:, :self.num_adversaries].sum(axis=1)
        elif self.score_function == "min":
            # Distance to closest adversary (default, more intelligent)
            proj_pos = jnp.vstack((x, y)).transpose() + state.p_pos[aidx]
            # Fixed: add axis=1 to compute per-adversary distances
            rel_dis = jnp.sqrt(jnp.sum(jnp.square(
                state.p_pos[aidx] - state.p_pos[:self.num_adversaries]
            ), axis=1))
            min_dist_adv_idx = jnp.argmin(rel_dis)
            delta_pos = state.p_pos[:self.num_adversaries][None, :, :] - proj_pos[:, None, :]
            dist = jnp.sqrt(jnp.sum(jnp.square(delta_pos), axis=2))
            dist_min = self.rad[:self.num_adversaries] + self.rad[aidx]
            # If collision detected, mark as invalid (set to -9999999)
            collision_mask = (dist < dist_min[None]).sum(axis=1) > 0
            scores = jnp.where(collision_mask, -9999999.0, scores)
            scores += dist[:, min_dist_adv_idx]
        else:
            # Fallback to no-op
            return jnp.zeros(2, dtype=jnp.float32)

        # Move to best position
        best_idx = jnp.argmax(scores)
        chosen_action = jnp.array([x[best_idx], y[best_idx]], dtype=jnp.float32)
        # If all scores are bad, stay still
        chosen_action = jax.lax.cond(
            scores[best_idx] < 0,
            lambda: chosen_action * 0.0,
            lambda: chosen_action
        )
        return chosen_action

    @partial(jax.jit, static_argnums=[0])
    def step_env(self, key: chex.PRNGKey, state: State, actions: dict):
        """
        Step environment with scripted prey.

        Actions dict only contains adversary actions (discrete).
        Prey action is generated by scripted policy.
        """
        # Create full actions dict including dummy prey action
        # (will be replaced by scripted policy in force space)
        full_actions = dict(actions)
        # Get shape from an existing adversary action to match shapes
        # (important for vmap compatibility where actions might be (1,) instead of scalar)
        sample_action_shape = next(iter(actions.values())).shape
        for prey_name in self.good_agents:
            full_actions[prey_name] = jnp.zeros(sample_action_shape, dtype=jnp.int32)  # dummy, will be overwritten

        # Convert discrete actions to force vectors using parent's set_actions
        # This handles the discrete -> continuous force conversion
        # Note: _all_agents has correct order for set_actions
        u, c = self.set_actions(full_actions)

        # Replace prey actions with scripted policy output
        # u shape: (num_agents, 2) where last num_good_agents are prey
        u_adversaries = u[:self.num_adversaries]

        # Generate scripted prey actions
        prey_actions = []
        for i in range(self.num_good_agents):
            prey_idx = self.num_adversaries + i
            key, _key = jax.random.split(key)
            prey_action = self._prey_policy(_key, state, prey_idx)
            prey_actions.append(prey_action)

        # Concatenate adversary forces with prey scripted actions
        u_prey = jnp.stack(prey_actions, axis=0)
        u = jnp.concatenate([u_adversaries, u_prey], axis=0)

        # Handle communication channels
        if c.shape[1] < self.dim_c:
            c = jnp.concatenate(
                [c, jnp.zeros((self.num_agents, self.dim_c - c.shape[1]))],
                axis=1
            )

        # Run physics simulation
        key, key_w = jax.random.split(key)
        p_pos, p_vel = self._world_step(key_w, state, u)

        # Apply communication
        key_c = jax.random.split(key, self.num_agents)
        c = self._apply_comm_action(key_c, c, self.c_noise, self.silent)

        # Check done
        done = jnp.full((self.num_agents,), state.step >= self.max_steps)

        # Update state
        state = state.replace(
            p_pos=p_pos,
            p_vel=p_vel,
            c=c,
            done=done,
            step=state.step + 1,
        )

        # Get rewards and observations
        all_rewards = self.rewards(state)
        all_obs = self.get_obs(state)

        # Filter to only return adversary info (learning agents)
        reward = {a: all_rewards[a] for a in self._learning_agents}
        obs = {a: all_obs[a] for a in self._learning_agents}

        dones = {a: done[i] for i, a in enumerate(self._learning_agents)}
        dones.update({"__all__": jnp.all(done[:self.num_adversaries])})

        info = {}

        return obs, state, reward, dones, info

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: chex.PRNGKey):
        """Reset environment and return only adversary observations."""
        # Parent reset uses self.num_agents (4) and self.get_obs
        # get_obs uses self.adversaries/good_agents directly, so it works
        key, _key = jax.random.split(key)

        # Initialize positions
        key_a, key_l = jax.random.split(_key)
        sr = getattr(self, 'spawn_range', 1.0)
        p_pos = jnp.concatenate([
            jax.random.uniform(key_a, (self.num_agents, 2), minval=-sr, maxval=+sr),
            jax.random.uniform(key_l, (self.num_landmarks, 2), minval=-sr, maxval=+sr),
        ])

        state = State(
            p_pos=p_pos,
            p_vel=jnp.zeros((self.num_entities, self.dim_p)),
            c=jnp.zeros((self.num_agents, self.dim_c)),
            done=jnp.full((self.num_agents,), False),
            step=0,
        )

        # Get all observations and filter to learning agents
        all_obs = self.get_obs(state)
        obs = {a: all_obs[a] for a in self._learning_agents}

        return obs, state

    def rewards(self, state: State) -> Dict[str, float]:
        """
        Compute rewards with optional dense shaping.

        sparse mode: +10 on collision (original)
        dense mode: +10 on collision + distance-based shaping
        """
        # Get sparse collision reward from parent
        sparse_rewards = super().rewards(state)

        if self.reward_mode == "sparse":
            return sparse_rewards

        # === Dense reward: add distance-based shaping ===
        # Compute min distance from each adversary to any prey
        # adversary positions: state.p_pos[:num_adversaries]
        # prey positions: state.p_pos[num_adversaries:num_adversaries+num_good_agents]

        adv_pos = state.p_pos[:self.num_adversaries]  # (num_adv, 2)
        prey_pos = state.p_pos[self.num_adversaries:self.num_adversaries + self.num_good_agents]  # (num_prey, 2)

        # Distance from each adversary to each prey: (num_adv, num_prey)
        delta = adv_pos[:, None, :] - prey_pos[None, :, :]  # (num_adv, num_prey, 2)
        dists = jnp.sqrt(jnp.sum(jnp.square(delta), axis=-1))  # (num_adv, num_prey)

        # Min distance to any prey for each adversary
        min_dist_per_adv = jnp.min(dists, axis=-1)  # (num_adv,)

        # Team min distance (closest adversary to prey)
        team_min_dist = jnp.min(min_dist_per_adv)

        # Dense shaping: negative distance (closer = higher reward)
        # Normalize by typical map size (~2.0 diagonal)
        # Reward range: roughly [-1, 0] when scale=1.0
        dense_shaping = -team_min_dist * self.dense_reward_scale

        # Add dense shaping to all adversaries (shared team reward)
        dense_rewards = {}
        for a in self.adversaries:
            dense_rewards[a] = sparse_rewards[a] + dense_shaping

        # Keep prey rewards unchanged
        for a in self.good_agents:
            dense_rewards[a] = sparse_rewards[a]

        return dense_rewards

    def get_obs(self, state: State) -> Dict[str, chex.Array]:
        """
        Get observations for all agents.
        Only adversary observations are returned to the learning algorithm.
        """
        # Use parent's get_obs which handles the observation construction
        return super().get_obs(state)

    def observation_space(self, agent: str):
        """Return observation space for agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        """Return action space for agent."""
        return self.action_spaces[agent]


SimpleTagScriptedPrey6v2 = lambda **kwargs: SimpleTagScriptedPreyMPE(
    num_adversaries=6, num_good_agents=2, num_obs=3, **kwargs
)
