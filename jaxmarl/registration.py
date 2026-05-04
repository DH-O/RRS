from .environments import (
    SimpleTagScriptedPrey6v2,
    SimpleCorridorMPE,
    Ant,
    AntBall,
    Humanoid,
    Hopper,
    Walker2d,
    HalfCheetah,
    SMAX,
    HeuristicEnemySMAX,
    HeuristicEnemyCorridorSMAX,
    LearnedPolicyEnemySMAX,
    map_name_to_scenario,
    SMAXCorridor,
    make_corridor_scenario,
)

# SMAX map names for convenience registration
SMAX_MAP_NAMES = [
    "3m",
    "2s3z",
    "25m",
    "3s5z",
    "8m",
    "5m_vs_6m",
    "10m_vs_11m",
    "27m_vs_30m",
    "3s5z_vs_3s6z",
    "3s_vs_5z",
    "6h_vs_8z",
    "smacv2_5_units",
    "smacv2_10_units",
    "smacv2_20_units",
    "12h_vs_16z",
    "6s10z_vs_6s12z",
]

# Corridor map names
SMAX_CORRIDOR_MAP_NAMES = [
    "corridor_3s5z",
    "corridor_3m",
    "corridor_8m",
    "corridor_5m_vs_6m",
    "corridor_10m_narrow",
    "corridor_5m5z_narrow",
    "corridor_5s10z_tight",
]


def make(env_id: str, **env_kwargs):
    if env_id not in registered_envs:
        raise ValueError(f"{env_id} is not in registered jaxmarl environments.")

    # MPE Environments
    if env_id == "MPE_simple_tag_scripted_prey_6v2":
        env = SimpleTagScriptedPrey6v2(**env_kwargs)
    elif env_id == "MPE_simple_corridor_v3":
        env = SimpleCorridorMPE(**env_kwargs)
    # MABrax Environments
    elif env_id == "ant_4x2":
        env = Ant(**env_kwargs)
    elif env_id == "ant_ball_4x2":
        env = AntBall(**env_kwargs)
    elif env_id == "halfcheetah_6x1":
        env = HalfCheetah(**env_kwargs)
    elif env_id == "hopper_3x1":
        env = Hopper(**env_kwargs)
    elif env_id == "humanoid_9|8":
        env = Humanoid(**env_kwargs)
    elif env_id == "walker2d_2x3":
        env = Walker2d(**env_kwargs)
    # SMAX Environments
    elif env_id == "SMAX":
        env = SMAX(**env_kwargs)
    elif env_id == "HeuristicEnemySMAX":
        env = HeuristicEnemySMAX(**env_kwargs)
    elif env_id == "LearnedPolicyEnemySMAX":
        env = LearnedPolicyEnemySMAX(**env_kwargs)
    # SMAX Corridor environments (with heuristic enemies)
    elif env_id.startswith("SMAXCorridor_"):
        corridor_map_name = env_id[len("SMAXCorridor_"):]
        corridor_config = make_corridor_scenario(corridor_map_name)
        scenario = corridor_config.pop("scenario")
        # Merge corridor-specific kwargs with user kwargs (user overrides)
        merged_kwargs = {**corridor_config, **env_kwargs}
        env = HeuristicEnemyCorridorSMAX(scenario=scenario, **merged_kwargs)
    # SMAX map-specific convenience environments
    elif env_id.startswith("SMAX_"):
        map_name = env_id[5:]  # strip "SMAX_" prefix
        scenario = map_name_to_scenario(map_name)
        env = HeuristicEnemySMAX(scenario=scenario, **env_kwargs)
    else:
        raise ValueError(f"Unknown env_id: {env_id}")

    return env


registered_envs = [
    "MPE_simple_tag_scripted_prey_6v2",
    "MPE_simple_corridor_v3",
    # MABrax Environments
    "ant_4x2",
    "ant_ball_4x2",
    "halfcheetah_6x1",
    "hopper_3x1",
    "humanoid_9|8",
    "walker2d_2x3",
    # SMAX Environments
    "SMAX",
    "HeuristicEnemySMAX",
    "LearnedPolicyEnemySMAX",
] + [f"SMAX_{name}" for name in SMAX_MAP_NAMES] + [
    f"SMAXCorridor_{name}" for name in SMAX_CORRIDOR_MAP_NAMES
]
