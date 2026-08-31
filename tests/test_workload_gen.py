"""
Unit tests for common/workload_gen.py.

Required at minimum per CODING_AGENT_PLAN.md §3: shape arithmetic is one of the two
places a silent bug would invisibly corrupt every downstream number (the other is
metrics.py, see test_metrics.py).
"""
from __future__ import annotations

import pytest

from common.workload_gen import (
    AttentionWorkload,
    gemm_shapes_to_topology_rows,
    workload_to_gemm_shapes,
)


def test_qkv_projection_shape() -> None:
    """QKV projection is one combined GEMM: (M=seq_len, K=hidden_dim, N=3*hidden_dim)."""
    workload = AttentionWorkload(seq_len=512, hidden_dim=768, num_heads=12)
    shapes = {s.name: s for s in workload_to_gemm_shapes(workload)}
    qkv = shapes["qkv_projection"]
    assert (qkv.m, qkv.k, qkv.n, qkv.repeat) == (512, 768, 3 * 768, 1)


def test_per_head_shapes_use_head_dim_and_repeat_by_num_heads() -> None:
    """QK^T and score*V operate on (seq_len, head_dim) slices, repeated once per head."""
    workload = AttentionWorkload(seq_len=512, hidden_dim=768, num_heads=12)
    head_dim = 768 // 12
    shapes = {s.name: s for s in workload_to_gemm_shapes(workload)}

    qk = shapes["qk_transpose"]
    assert (qk.m, qk.k, qk.n, qk.repeat) == (512, head_dim, 512, 12)

    sv = shapes["score_times_v"]
    assert (sv.m, sv.k, sv.n, sv.repeat) == (512, 512, head_dim, 12)


def test_num_heads_must_divide_hidden_dim() -> None:
    """A hidden_dim not divisible by num_heads must raise, not silently truncate."""
    with pytest.raises(ValueError):
        AttentionWorkload(seq_len=512, hidden_dim=768, num_heads=5)


def test_non_positive_dimensions_rejected() -> None:
    with pytest.raises(ValueError):
        AttentionWorkload(seq_len=0, hidden_dim=768, num_heads=12)


def test_head_dim_property() -> None:
    workload = AttentionWorkload(seq_len=128, hidden_dim=256, num_heads=8)
    assert workload.head_dim == 32


def test_topology_rows_expand_repeated_stages_and_count_matches() -> None:
    """A repeat=N stage becomes N distinct topology rows; total row count = 1 + 2*num_heads."""
    workload = AttentionWorkload(seq_len=64, hidden_dim=128, num_heads=4)
    shapes = workload_to_gemm_shapes(workload)
    rows = gemm_shapes_to_topology_rows(shapes)

    assert len(rows) == 1 + 4 + 4  # qkv_projection(1) + qk_transpose(4) + score_times_v(4)
    names = [r["LayerName"] for r in rows]
    assert names[0] == "qkv_projection"
    assert "qk_transpose_head0" in names and "qk_transpose_head3" in names
    assert "score_times_v_head0" in names and "score_times_v_head3" in names


def test_topology_row_conv_lowering_matches_gemm_dims() -> None:
    """Each topology row's conv-lowering fields must reproduce the source GEMM's (M,K,N)."""
    workload = AttentionWorkload(seq_len=64, hidden_dim=128, num_heads=4)
    shapes = workload_to_gemm_shapes(workload)
    qkv_shape = next(s for s in shapes if s.name == "qkv_projection")
    row = gemm_shapes_to_topology_rows([qkv_shape])[0]

    assert row["IFMAP Height"] == 1
    assert row["IFMAP Width"] == qkv_shape.m
    assert row["Channels"] == qkv_shape.k
    assert row["Filter Height"] == 1
    assert row["Filter Width"] == 1
    assert row["Num Filter"] == qkv_shape.n
    assert row["Strides"] == 1
