"""
SMAX Corridor Environment: A variant of SMAX with wall obstacles that create
a chokepoint, forcing units through a narrow corridor.

Layout (32x32 map):
    +--------------------------------+
    |                                |
    | Team 0        ####   Team 1    |
    | (left)        ####   (right)   |
    |               ####             |
    |           gap>    <gap         |
    |               ####             |
    | (left)        ####   (right)   |
    |               ####             |
    |                                |
    +--------------------------------+

The wall is a vertical barrier in the center of the map with a configurable gap.
Units must navigate through the gap to reach the other side.
"""

import jax.numpy as jnp
import jax
import chex
import numpy as np
from typing import Tuple, Dict, Optional
from flax.struct import dataclass as flax_dataclass
from functools import partial

from jaxmarl.environments.smax.smax_env import SMAX, State, Scenario


# Wall rectangle: (x_min, y_min, x_max, y_max)
# We define walls as a pair of rectangles (upper wall and lower wall)
# with a gap in between.


class SMAXCorridor(SMAX):
    """SMAX variant with wall obstacles creating a corridor/chokepoint.

    The map has a vertical wall in the center with a configurable gap.
    Team 0 spawns on the left, Team 1 on the right. Units must navigate
    through the chokepoint to engage the enemy.

    Args:
        wall_x: X-coordinate of the wall center. Default: map_width / 2.
        wall_thickness: Thickness of the wall in map units. Default: 2.0.
        gap_width: Width of the gap in the wall (in y-axis units). Default: 6.0.
        gap_y: Y-coordinate of the gap center. Default: map_height / 2.
        spawn_spread: Random noise spread for spawn positions. Default: 3.0.
        num_wall_obs_features: Number of wall-related observation features. Default: 0.
            If > 0, adds features like distance to wall gap.
    """

    def __init__(
        self,
        wall_x: Optional[float] = None,
        wall_thickness: float = 2.0,
        gap_width: float = 6.0,
        gap_y: Optional[float] = None,
        spawn_spread: float = 3.0,
        num_wall_obs_features: int = 0,
        **env_kwargs,
    ) -> None:
        super().__init__(**env_kwargs)

        # Wall geometry
        self.wall_x = wall_x if wall_x is not None else self.map_width / 2.0
        self.wall_thickness = wall_thickness
        self.gap_width = gap_width
        self.gap_y = gap_y if gap_y is not None else self.map_height / 2.0
        self.spawn_spread = spawn_spread
        self.num_wall_obs_features = num_wall_obs_features

        # Precompute wall rectangles as static arrays.
        # Wall is centered at (wall_x, gap_y) with a gap of gap_width.
        # Upper wall: from gap top to map top
        # Lower wall: from map bottom to gap bottom
        wall_x_min = self.wall_x - self.wall_thickness / 2.0
        wall_x_max = self.wall_x + self.wall_thickness / 2.0
        gap_half = self.gap_width / 2.0

        # walls[i] = (x_min, y_min, x_max, y_max)
        # Lower wall: y from 0 to gap_y - gap_half
        # Upper wall: y from gap_y + gap_half to map_height
        self.walls = jnp.array([
            [wall_x_min, 0.0, wall_x_max, self.gap_y - gap_half],           # lower wall
            [wall_x_min, self.gap_y + gap_half, wall_x_max, self.map_height], # upper wall
        ])
        self.num_walls = 2

        # Gap center for observation features
        self.gap_center = jnp.array([self.wall_x, self.gap_y])

        # Update obs_size if we add wall observation features
        if self.num_wall_obs_features > 0:
            self.own_features = self.own_features + [
                f"wall_feat_{i}" for i in range(self.num_wall_obs_features)
            ]
            self.obs_size = self._get_obs_size()
            # Rebuild observation/action spaces
            from jaxmarl.environments.spaces import Box, Discrete
            self.observation_spaces = {
                i: Box(low=-1.0, high=1.0, shape=(self.obs_size,))
                for i in self.agents
            }
            self.state_size = (len(self.own_features) + 2) * self.num_agents

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        """Reset with corridor-aware spawn positions.

        Team 0 spawns on the left side (x ~ map_width/4),
        Team 1 spawns on the right side (x ~ 3*map_width/4).
        Both teams spawn near the gap y-coordinate to encourage
        immediate corridor engagement.
        """
        key, team_0_key, team_1_key = jax.random.split(key, num=3)

        # Team 0: left side, spread around (map_width/4, gap_y)
        team_0_center = jnp.array([self.map_width / 4.0, self.gap_y])
        team_0_noise = jax.random.uniform(
            team_0_key,
            shape=(self.num_allies, 2),
            minval=-self.spawn_spread,
            maxval=self.spawn_spread,
        )
        team_0_start = jnp.stack([team_0_center] * self.num_allies) + team_0_noise

        # Team 1: right side, spread around (3*map_width/4, gap_y)
        team_1_center = jnp.array([3.0 * self.map_width / 4.0, self.gap_y])
        team_1_noise = jax.random.uniform(
            team_1_key,
            shape=(self.num_enemies, 2),
            minval=-self.spawn_spread,
            maxval=self.spawn_spread,
        )
        team_1_start = jnp.stack([team_1_center] * self.num_enemies) + team_1_noise

        unit_positions = jnp.concatenate([team_0_start, team_1_start])

        # Clamp to map boundaries
        unit_positions = jnp.clip(
            unit_positions,
            jnp.array([0.5, 0.5]),
            jnp.array([self.map_width - 0.5, self.map_height - 0.5]),
        )

        # Push away from walls
        unit_positions = self._clamp_positions_to_walls(unit_positions)

        # Handle SMACv2 position generation: override if enabled (unlikely for corridor)
        key, pos_key = jax.random.split(key)
        generated_unit_positions = self.position_generator.generate(pos_key)
        unit_positions = jax.lax.select(
            self.smacv2_position_generation, generated_unit_positions, unit_positions
        )

        unit_teams = jnp.zeros((self.num_agents,))
        unit_teams = unit_teams.at[self.num_allies:].set(1)
        unit_weapon_cooldowns = jnp.zeros((self.num_agents,))

        # Unit types
        unit_types = (
            jnp.zeros((self.num_agents,), dtype=jnp.uint8)
            if self.scenario is None
            else self.scenario
        )
        key, unit_type_key = jax.random.split(key)
        generated_unit_types = self.unit_type_generator.generate(unit_type_key)
        unit_types = jax.lax.select(
            self.smacv2_unit_type_generation, generated_unit_types, unit_types
        )
        unit_health = self.unit_type_health[unit_types]

        state = State(
            unit_positions=unit_positions,
            unit_alive=jnp.ones((self.num_agents,), dtype=jnp.bool_),
            unit_teams=unit_teams,
            unit_health=unit_health,
            unit_types=unit_types,
            prev_movement_actions=jnp.zeros((self.num_agents, 2)),
            prev_attack_actions=jnp.zeros((self.num_agents,), dtype=jnp.int32),
            time=0,
            terminal=False,
            unit_weapon_cooldowns=unit_weapon_cooldowns,
        )
        state = self._push_units_away(state)
        # Re-clamp after pushing (push might move units into walls)
        state = state.replace(
            unit_positions=self._clamp_positions_to_walls(state.unit_positions)
        )
        obs = self.get_obs(state)
        world_state = self.get_world_state(state)
        obs["world_state"] = jax.lax.stop_gradient(world_state)
        return obs, state

    def _point_in_rect(self, pos: chex.Array, rect: chex.Array) -> chex.Array:
        """Check if a 2D point is inside a rectangle.

        Args:
            pos: (2,) array [x, y]
            rect: (4,) array [x_min, y_min, x_max, y_max]

        Returns:
            Boolean scalar
        """
        return (
            (pos[0] >= rect[0])
            & (pos[0] <= rect[2])
            & (pos[1] >= rect[1])
            & (pos[1] <= rect[3])
        )

    def _resolve_wall_collision_single(
        self, old_pos: chex.Array, new_pos: chex.Array, rect: chex.Array
    ) -> chex.Array:
        """Resolve collision with a single wall rectangle.

        If new_pos is inside the wall, project it to the nearest edge.
        Uses the movement direction to determine which face to project onto.

        Args:
            old_pos: (2,) previous position
            new_pos: (2,) proposed new position
            rect: (4,) wall [x_min, y_min, x_max, y_max]

        Returns:
            (2,) resolved position (outside the wall)
        """
        inside = self._point_in_rect(new_pos, rect)

        # Compute distance to each face of the rectangle
        # We want to push the unit to the nearest face
        dist_left = jnp.abs(new_pos[0] - rect[0])
        dist_right = jnp.abs(new_pos[0] - rect[2])
        dist_bottom = jnp.abs(new_pos[1] - rect[1])
        dist_top = jnp.abs(new_pos[1] - rect[3])

        # Find minimum distance face and project to it
        # Use movement direction to break ties: prefer projecting
        # in the direction we came from
        dx = new_pos[0] - old_pos[0]
        dy = new_pos[1] - old_pos[1]

        # Bias distances based on movement direction:
        # If moving right (dx > 0), prefer projecting to left face
        # If moving up (dy > 0), prefer projecting to bottom face
        eps = 0.01
        dist_left = jnp.where(dx > 0, dist_left, dist_left + eps)
        dist_right = jnp.where(dx < 0, dist_right, dist_right + eps)
        dist_bottom = jnp.where(dy > 0, dist_bottom, dist_bottom + eps)
        dist_top = jnp.where(dy < 0, dist_top, dist_top + eps)

        dists = jnp.array([dist_left, dist_right, dist_bottom, dist_top])
        min_face = jnp.argmin(dists)

        # Project to the nearest face with a small offset to avoid exact boundary
        offset = 0.05
        projected = jnp.where(
            min_face == 0,
            jnp.array([rect[0] - offset, new_pos[1]]),  # left face
            jnp.where(
                min_face == 1,
                jnp.array([rect[2] + offset, new_pos[1]]),  # right face
                jnp.where(
                    min_face == 2,
                    jnp.array([new_pos[0], rect[1] - offset]),  # bottom face
                    jnp.array([new_pos[0], rect[3] + offset]),  # top face
                ),
            ),
        )

        return jnp.where(inside, projected, new_pos)

    def _resolve_all_wall_collisions(
        self, old_pos: chex.Array, new_pos: chex.Array
    ) -> chex.Array:
        """Resolve collisions with all walls for a single unit.

        Args:
            old_pos: (2,) previous position
            new_pos: (2,) proposed new position

        Returns:
            (2,) resolved position
        """
        def resolve_one_wall(carry_pos, rect):
            resolved = self._resolve_wall_collision_single(old_pos, carry_pos, rect)
            return resolved, None

        resolved_pos, _ = jax.lax.scan(resolve_one_wall, new_pos, self.walls)
        return resolved_pos

    def _clamp_positions_to_walls(self, positions: chex.Array) -> chex.Array:
        """Clamp an array of positions so none are inside walls.

        For initial positioning (no old_pos), we use the gap center as the
        'movement source' to push units outward.

        Args:
            positions: (num_agents, 2)

        Returns:
            (num_agents, 2) clamped positions
        """
        def clamp_single(pos):
            # Use gap center as the reference for initial clamping
            return self._resolve_all_wall_collisions(self.gap_center, pos)

        return jax.vmap(clamp_single)(positions)

    @partial(jax.jit, static_argnums=(0,))
    def _world_step(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Tuple[chex.Array, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """World step with wall collision resolution.

        After computing new positions via the parent's movement logic,
        resolve any wall collisions by projecting units out of walls.
        """
        old_positions = state.unit_positions

        # Call parent world step to get basic movement + combat
        state = super()._world_step(key=key, state=state, actions=actions)

        # Resolve wall collisions for all units
        new_positions = state.unit_positions

        def resolve_unit(args):
            old_pos, new_pos = args
            return self._resolve_all_wall_collisions(old_pos, new_pos)

        resolved_positions = jax.vmap(resolve_unit)(
            (old_positions, new_positions)
        )

        # Clamp to map boundaries after wall resolution
        resolved_positions = jnp.clip(
            resolved_positions,
            jnp.zeros((2,)),
            jnp.array([self.map_width, self.map_height]),
        )

        state = state.replace(unit_positions=resolved_positions)
        return state

    def _push_units_away(self, state: State, firmness: float = 1.0):
        """Push overlapping units apart, then re-clamp to walls."""
        state = super()._push_units_away(state, firmness)
        # After pushing, ensure no unit ended up inside a wall
        state = state.replace(
            unit_positions=self._clamp_positions_to_walls(state.unit_positions)
        )
        return state

    def _get_own_features(self, state: State, i: int):
        """Get own features, optionally with wall-relative observations."""
        base_features = super()._get_own_features(state, i)

        if self.num_wall_obs_features == 0:
            return base_features

        # Add wall-related features (normalized)
        pos = state.unit_positions[i]
        wall_features = jnp.zeros((self.num_wall_obs_features,))

        # Feature 0: normalized x-distance to wall center
        wall_features = wall_features.at[0].set(
            (pos[0] - self.wall_x) / self.map_width
        )

        if self.num_wall_obs_features >= 2:
            # Feature 1: normalized y-distance to gap center
            wall_features = wall_features.at[1].set(
                (pos[1] - self.gap_y) / self.map_height
            )

        if self.num_wall_obs_features >= 3:
            # Feature 2: is the unit on the same side as the gap?
            # (can it see through the gap?)
            dist_to_gap = jnp.sqrt(
                (pos[0] - self.wall_x) ** 2 + (pos[1] - self.gap_y) ** 2
            )
            wall_features = wall_features.at[2].set(
                dist_to_gap / jnp.sqrt(self.map_width**2 + self.map_height**2)
            )

        if self.num_wall_obs_features >= 4:
            # Feature 3: is the unit close enough to pass through the gap?
            # 1.0 if y is within gap, 0.0 otherwise
            in_gap_y = (
                (pos[1] >= self.gap_y - self.gap_width / 2.0)
                & (pos[1] <= self.gap_y + self.gap_width / 2.0)
            ).astype(jnp.float32)
            wall_features = wall_features.at[3].set(in_gap_y)

        # Insert wall features at the end of own_features
        # base_features has length: health(1) + pos(2) + cooldown(1) + type_bits(6) = 10
        # We need to append our wall features
        n_base = len(self.own_features) - self.num_wall_obs_features
        result = jnp.zeros((len(self.own_features),))
        result = result.at[:n_base].set(base_features[:n_base])
        result = result.at[n_base:].set(wall_features)

        # Mask by alive status (same as parent)
        empty_features = jnp.zeros(shape=(len(self.own_features),))
        return jax.lax.cond(
            state.unit_alive[i], lambda: result, lambda: empty_features
        )

    def init_render(self, ax, state, step, env_step):
        """Render with walls drawn as gray rectangles."""
        from matplotlib.patches import Rectangle as MplRect
        import numpy as np_render

        # Call parent render first
        result = super().init_render(ax, state, step, env_step)

        # Draw walls
        for w in range(self.num_walls):
            wall = self.walls[w]
            x_min, y_min, x_max, y_max = float(wall[0]), float(wall[1]), float(wall[2]), float(wall[3])
            rect = MplRect(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                facecolor="gray",
                edgecolor="black",
                alpha=0.85,
                zorder=3,
            )
            ax.add_patch(rect)

        return result


# ============================================================
# Corridor Scenario Definitions
# ============================================================

def make_corridor_scenario(map_name: str) -> dict:
    """Create corridor-specific kwargs from a map name.

    Returns both the Scenario and extra kwargs for SMAXCorridor.
    """
    CORRIDOR_CONFIGS = {
        "corridor_3s5z": {
            "scenario": Scenario(
                unit_types=np.array(
                    [2, 2, 2, 3, 3, 3, 3, 3] * 2, dtype=np.uint8
                ),
                num_allies=8,
                num_enemies=8,
                smacv2_position_generation=False,
                smacv2_unit_type_generation=False,
            ),
            "map_width": 32,
            "map_height": 32,
            "wall_thickness": 2.0,
            "gap_width": 6.0,
            "spawn_spread": 3.0,
        },
        "corridor_3m": {
            "scenario": Scenario(
                unit_types=np.zeros((6,), dtype=np.uint8),
                num_allies=3,
                num_enemies=3,
                smacv2_position_generation=False,
                smacv2_unit_type_generation=False,
            ),
            "map_width": 32,
            "map_height": 32,
            "wall_thickness": 2.0,
            "gap_width": 4.0,
            "spawn_spread": 2.0,
        },
        "corridor_8m": {
            "scenario": Scenario(
                unit_types=np.zeros((16,), dtype=np.uint8),
                num_allies=8,
                num_enemies=8,
                smacv2_position_generation=False,
                smacv2_unit_type_generation=False,
            ),
            "map_width": 32,
            "map_height": 32,
            "wall_thickness": 2.0,
            "gap_width": 6.0,
            "spawn_spread": 3.0,
        },
        "corridor_5m_vs_6m": {
            "scenario": Scenario(
                unit_types=np.zeros((11,), dtype=np.uint8),
                num_allies=5,
                num_enemies=6,
                smacv2_position_generation=False,
                smacv2_unit_type_generation=False,
            ),
            "map_width": 32,
            "map_height": 32,
            "wall_thickness": 2.0,
            "gap_width": 5.0,
            "spawn_spread": 2.5,
        },
        # --- Harder corridor variants ---
        "corridor_10m_narrow": {
            "scenario": Scenario(
                unit_types=np.zeros((20,), dtype=np.uint8),  # 10 marines per side
                num_allies=10,
                num_enemies=10,
                smacv2_position_generation=False,
                smacv2_unit_type_generation=False,
            ),
            "map_width": 32,
            "map_height": 32,
            "wall_thickness": 2.0,
            "gap_width": 3.0,
            "spawn_spread": 3.0,
        },
        "corridor_5m5z_narrow": {
            "scenario": Scenario(
                unit_types=np.array(
                    [0, 0, 0, 0, 0, 3, 3, 3, 3, 3] * 2, dtype=np.uint8
                ),  # 5 marines + 5 zealots per side
                num_allies=10,
                num_enemies=10,
                smacv2_position_generation=False,
                smacv2_unit_type_generation=False,
            ),
            "map_width": 48,
            "map_height": 48,
            "wall_thickness": 2.0,
            "gap_width": 3.0,
            "spawn_spread": 4.0,
        },
        "corridor_5s10z_tight": {
            "scenario": Scenario(
                unit_types=np.array(
                    [2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3] * 2,
                    dtype=np.uint8,
                ),  # 5 stalkers + 10 zealots per side (15v15)
                num_allies=15,
                num_enemies=15,
                smacv2_position_generation=False,
                smacv2_unit_type_generation=False,
            ),
            "map_width": 48,
            "map_height": 48,
            "wall_thickness": 2.0,
            "gap_width": 4.0,
            "spawn_spread": 5.0,
        },
    }

    if map_name not in CORRIDOR_CONFIGS:
        raise ValueError(
            f"Unknown corridor map: {map_name}. "
            f"Available: {list(CORRIDOR_CONFIGS.keys())}"
        )

    return CORRIDOR_CONFIGS[map_name]
