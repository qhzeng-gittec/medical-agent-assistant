"""Shared Qwen loading with verified Flash Linear Attention dispatch."""
import inspect
import logging
import os
from pathlib import Path
import sys


LAB = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())
logger.propagate = False
_backend = None


def configure_linear_attention():
    global _backend
    requested = os.environ.get('MEDICAL_CPT_FAST_KERNELS', '1')
    if requested not in ('0', '1'):
        raise ValueError('MEDICAL_CPT_FAST_KERNELS must be 0 or 1.')
    backend = 'fla' if requested == '1' else 'torch'
    if _backend is not None:
        if _backend != backend:
            raise RuntimeError('Select the attention backend before loading any models.')
        return _backend

    if backend == 'fla':
        sys.path.insert(0, str(LAB / '.gpu-kernels'))
        if sys.platform == 'win32':
            os.environ.setdefault('CC', str(LAB / '.gpu-kernels/triton/runtime/tcc/tcc.exe'))
            os.environ.setdefault('CUDA_PATH', str(LAB / '.gpu-kernels/triton/backends/nvidia'))
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

    from transformers.integrations.hub_kernels import use_kernel_func_from_hub_with_fallback
    from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

    functions = {
        'torch_chunk_gated_delta_rule': 'chunk_gated_delta_rule',
        'torch_recurrent_gated_delta_rule': 'fused_recurrent_gated_delta_rule',
    }
    expected = (chunk_gated_delta_rule, fused_recurrent_gated_delta_rule) if backend == 'fla' else (None, None)
    for (name, kernel_name), kernel in zip(functions.items(), expected, strict=True):
        original = inspect.unwrap(getattr(qwen, name))
        if backend == 'fla':
            # PEFT may have imported Qwen before FLA became available. Rebind the
            # official dispatcher after importing FLA; retain its kwargs filtering.
            wrapped = use_kernel_func_from_hub_with_fallback(kernel_name, 'fla')(original)
            implementation = inspect.getclosurevars(wrapped).nonlocals['implementation']
            if implementation is not kernel:
                raise RuntimeError(f'{name} did not select the requested FLA kernel.')
        else:
            wrapped = original
        setattr(qwen, name, wrapped)

    _backend = backend
    logger.info('Linear attention backend=%s; chunk and recurrent dispatch verified (pid=%s)', backend, os.getpid())
    return backend


def load_qwen_model(*args, **kwargs):
    configure_linear_attention()
    from transformers import Qwen3_5ForConditionalGeneration
    return Qwen3_5ForConditionalGeneration.from_pretrained(*args, **kwargs)
