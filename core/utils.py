import os
import jax
import jax.numpy as jnp
import optax
import yaml
from safetensors.flax import save_file, load_file
from flax.traverse_util import flatten_dict, unflatten_dict

def stack_params(params_list):
    return jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *params_list)

class ParamsHolder:

    def __init__(self, params):
        self.params = params

def save_params(params, filename):
    flattened_dict = flatten_dict(params, sep=',')
    save_file(flattened_dict, filename)

def load_params(filename):
    flat_dict = load_file(filename)
    tuple_key_dict = {tuple(k.split(',')): v for (k, v) in flat_dict.items()}
    return unflatten_dict(tuple_key_dict)

def vectorized_apply_gradients(train_states, stacked_grads):
    stacked_params = stack_params([s.params for s in train_states])
    stacked_opt_state = jax.tree.map(lambda *xs: jnp.stack(xs), *[s.opt_state for s in train_states])
    tx = train_states[0].tx

    def _apply_single(params, opt_state, grads):
        (updates, new_opt_state) = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return (new_params, new_opt_state)
    (new_stacked_params, new_stacked_opt_state) = jax.vmap(_apply_single)(stacked_params, stacked_opt_state, stacked_grads)
    num_agents = len(train_states)
    return [train_states[i].replace(params=jax.tree.map(lambda x: x[i], new_stacked_params), opt_state=jax.tree.map(lambda x: x[i], new_stacked_opt_state), step=train_states[i].step + 1) for i in range(num_agents)]

def save_training_checkpoints(run, config, out, save_params_fn):
    import wandb
    if config.get('WANDB_MODE') == 'disabled' or not hasattr(run, 'dir'):
        return
    save_dir = os.path.join(run.dir, 'params')
    os.makedirs(save_dir, exist_ok=True)
    train_states = out['runner_state'][0][0]
    actor_train_states = train_states[0]
    if isinstance(actor_train_states, (tuple, list)) and len(actor_train_states) > 1:
        params = jax.tree.map(lambda *xs: jnp.stack(xs), *[s.params for s in actor_train_states])
    else:
        s = actor_train_states[0] if isinstance(actor_train_states, (tuple, list)) else actor_train_states
        params = s.params
    save_params_fn(params, os.path.join(save_dir, 'actor_params.safetensors'))
    save_params_fn(train_states[1].params, os.path.join(save_dir, 'critic_params.safetensors'))
    if len(train_states) > 3:
        tdd_shared = config.get('TDD_SHARED_PARAMS', True)
        for idx, name in [(2, 'tdd_potential'), (3, 'tdd_s_encoder')]:
            try:
                states = train_states[idx]
                if tdd_shared:
                    p = states[0].params if isinstance(states, (tuple, list)) else states.params
                else:
                    p = jax.tree.map(lambda *xs: jnp.stack(xs), *[ts.params for ts in states])
                save_params_fn(p, os.path.join(save_dir, f'{name}_params.safetensors'))
            except Exception:
                pass
    if len(train_states) > 4:
        try:
            save_params_fn(train_states[4].params, os.path.join(save_dir, 'v_int_params.safetensors'))
        except Exception:
            pass
    with open(os.path.join(save_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f'Checkpoints saved to {save_dir}')
    if hasattr(run, 'log_artifact'):
        artifact = wandb.Artifact(f'{run.name}-checkpoint', type='checkpoint')
        for fn in os.listdir(save_dir):
            artifact.add_file(os.path.join(save_dir, fn))
        for h in ['training_history.csv', 'training_history.json']:
            p = os.path.join(run.dir, h)
            if os.path.exists(p):
                artifact.add_file(p)
        run.log_artifact(artifact)
