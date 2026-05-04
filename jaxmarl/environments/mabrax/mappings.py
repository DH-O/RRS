from typing import Dict, List, Tuple, Union
import jax.numpy as jnp

# TODO: programatically generate these mappings from the kinematic trees
#       and add an observation distance parameter to the environment


_agent_action_mapping = {
    "ant_4x2": {
        "agent_0": jnp.array([0, 1]),
        "agent_1": jnp.array([2, 3]),
        "agent_2": jnp.array([4, 5]),
        "agent_3": jnp.array([6, 7]),
    },
    "ant_ball_4x2": {
        "agent_0": jnp.array([0, 1]),
        "agent_1": jnp.array([2, 3]),
        "agent_2": jnp.array([4, 5]),
        "agent_3": jnp.array([6, 7]),
    },
    "halfcheetah_6x1": {
        "agent_0": jnp.array([0]),
        "agent_1": jnp.array([1]),
        "agent_2": jnp.array([2]),
        "agent_3": jnp.array([3]),
        "agent_4": jnp.array([4]),
        "agent_5": jnp.array([5]),
    },
    "hopper_3x1": {
        "agent_0": jnp.array([0]),
        "agent_1": jnp.array([1]),
        "agent_2": jnp.array([2]),
    },
    "humanoid_9|8": {
        "agent_0": jnp.array([0, 1, 2, 11, 12, 13, 14, 15, 16]),
        "agent_1": jnp.array([3, 4, 5, 6, 7, 8, 9, 10]),
    },
    "walker2d_2x3": {
        "agent_0": jnp.array([0, 1, 2]),
        "agent_1": jnp.array([3, 4, 5]),
    },
}


def listerize(ranges: List[Union[int, Tuple[int, int]]]) -> List[int]:
    return [
        i
        for r in ranges
        for i in (range(r[0], r[1] + 1) if isinstance(r, tuple) else [r])
    ]


ranges: Dict[str, Dict[str, List[Union[int, Tuple[int, int]]]]] = {
    # ant_ball_4x2: partial observability (24 dims per agent)
    # Follows ant_4x2 pattern + ball info. Each agent sees:
    #   torso state (z, quat), own+neighbor joint pos, torso vel,
    #   neighbor joint vel, ball_rel_pos, ball_vel
    # Full obs: qpos[2:](20) + qd(20) + ball_rel_pos(3) + ball_vel(3) = 46
    # Joint pos: hip1=5, ank1=6, hip2=7, ank2=8, hip3=9, ank3=10, hip4=11, ank4=12
    # Joint vel: h1v=26, a1v=27, h2v=28, a2v=29, h3v=30, a3v=31, h4v=32, a4v=33
    # Agent 0: actuators 0,1 = hip_4, ankle_4 (right_back_leg)
    # Agent 1: actuators 2,3 = hip_1, ankle_1 (front_left_leg)
    # Agent 2: actuators 4,5 = hip_2, ankle_2 (front_right_leg)
    # Agent 3: actuators 6,7 = hip_3, ankle_3 (back_leg)
    "ant_ball_4x2": {
        "agent_0": [(0, 5), 6, 7, 9, 11, (20, 25), 26, 27, (40, 45)],
        "agent_1": [(0, 5), 7, 8, 9, 11, (20, 25), 28, 29, (40, 45)],
        "agent_2": [(0, 5), 7, 9, 10, 11, (20, 25), 30, 31, (40, 45)],
        "agent_3": [(0, 5), 7, 9, 11, 12, (20, 25), 32, 33, (40, 45)],
    },
    "ant_4x2": {
        "agent_0": [(0, 5), 6, 7, 9, 11, (13, 18), 19, 20],
        "agent_1": [(0, 5), 7, 8, 9, 11, (13, 18), 21, 22],
        "agent_2": [(0, 5), 7, 9, 10, 11, (13, 18), 23, 24],
        "agent_3": [(0, 5), 7, 9, 11, 12, (13, 18), 25, 26],
    },
    "halfcheetah_6x1": {
        "agent_0": [(1, 2), 3, 4, 6, (9, 11), 12],
        "agent_1": [(1, 2), 3, 4, 5, (9, 11), 13],
        "agent_2": [(1, 2), 4, 5, (9, 11), 14],
        "agent_3": [(1, 2), 3, 6, 7, (9, 11), 15],
        "agent_4": [(1, 2), 6, 7, 8, (9, 11), 16],
        "agent_5": [(1, 2), 7, 8, (9, 11), 17],
    },
    "hopper_3x1": {
        "agent_0": [(0, 1), 2, 3, (5, 7), 8],
        "agent_1": [(0, 1), 2, 3, 4, (5, 7), 9],
        "agent_2": [(0, 1), 3, 4, (5, 7), 10],
    },
    "humanoid_9|8": {
        "agent_0": [
            (0, 10),
            (12, 14),
            (16, 30),
            (39, 44),
            (55, 94),
            (115, 124),
            (145, 184),
            (191, 214),
            (227, 232),
            (245, 277),
            (286, 291),
            (298, 321),
            (334, 339),
            (352, 375),
        ],
        "agent_1": [
            (0, 15),
            (22, 27),
            (31, 38),
            (85, 144),
            (209, 244),
            (269, 274),
            (278, 285),
            (316, 351),
        ],
    },
    "walker2d_2x3": {
        "agent_0": [0, (2, 5), (8, 9), (11, 13)],
        "agent_1": [0, 2, (5, 9), (14, 16)],
    },
}

_agent_observation_mapping = {
    k: {k_: jnp.array(listerize(v_)) for k_, v_ in v.items()} for k, v in ranges.items()
}
