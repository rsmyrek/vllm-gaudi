# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

from vllm_gaudi.v1.worker.hpu_worker import HPUWorker


class _DummyProfiler:

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def get_summary_string(self):
        return "0 B"


def test_determine_available_memory_reserves_full_hybrid_dummy_block(monkeypatch):
    worker = HPUWorker.__new__(HPUWorker)

    attn_spec = FullAttentionSpec(
        block_size=1,
        num_kv_heads=1,
        head_size=3,
        head_size_v=2,
        dtype=torch.bfloat16,
        page_size_padded=20,
    )
    mamba_spec = MambaSpec(
        block_size=1,
        shapes=((4, ), ),
        dtypes=(torch.float32, ),
        page_size_padded=20,
        mamba_type="gdn_attention",
    )
    forward_context = {
        "attn": SimpleNamespace(kv_cache="set"),
        "gdn": SimpleNamespace(kv_cache="set"),
    }

    worker.cache_config = SimpleNamespace(gpu_memory_utilization=1.0)
    worker.model_config = SimpleNamespace(use_mla=False, enforce_eager=False)
    worker.vllm_config = SimpleNamespace(compilation_config=SimpleNamespace(static_forward_context=forward_context))
    worker.model_runner = SimpleNamespace(
        get_kv_cache_spec=lambda: {
            "attn": attn_spec,
            "gdn": mamba_spec,
        },
        attn_backend=SimpleNamespace(get_kv_cache_shape=lambda *args: (1, )),
        profile_run=lambda initialize_only=True: None,
        mem_margin=None,
    )

    monkeypatch.setattr(
        "vllm_gaudi.v1.worker.hpu_worker.bind_kv_cache",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "vllm_gaudi.v1.worker.hpu_worker.HabanaMemoryProfiler",
        _DummyProfiler,
    )
    monkeypatch.setattr("vllm_gaudi.v1.worker.hpu_worker.gc.collect", lambda: None)
    monkeypatch.setattr("vllm_gaudi.v1.worker.hpu_worker.is_fake_hpu", lambda: False)
    monkeypatch.setattr(torch, "zeros", lambda *args, **kwargs: object())
    monkeypatch.setattr(torch, "ones", lambda *args, **kwargs: object())
    monkeypatch.setattr(torch.hpu, "synchronize", lambda: None)
    monkeypatch.setattr(torch.hpu, "mem_get_info", lambda: (1000, 1000))

    available = worker.determine_available_memory()

    assert available == 729
    assert worker.model_runner.mem_margin == 0
    assert forward_context["attn"].kv_cache is None
    assert forward_context["gdn"].kv_cache is None
