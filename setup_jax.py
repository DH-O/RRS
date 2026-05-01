import os
import logging

def _resolve_cuda_data_dir():
    try:
        import nvidia.cuda_nvcc
        return nvidia.cuda_nvcc.__path__[0]
    except (ImportError, AttributeError):
        return None

def _configure_xla_flags():
    xla_flags = os.environ.get('XLA_FLAGS', '')
    cuda_dir = _resolve_cuda_data_dir()
    if cuda_dir and 'xla_gpu_cuda_data_dir' not in xla_flags:
        xla_flags += f' --xla_gpu_cuda_data_dir={cuda_dir}'
    if os.environ.get('JAX_DETERMINISTIC', '1') == '1':
        if '--xla_gpu_deterministic_ops' not in xla_flags:
            xla_flags += ' --xla_gpu_deterministic_ops=true'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['XLA_FLAGS'] = xla_flags.strip()
_configure_xla_flags()
logging.getLogger('jax._src.xla_bridge').setLevel(logging.WARNING)
