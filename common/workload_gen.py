"""
workload_gen.py — Transformer hyperparameters -> QKV / QK^T / score*V GEMM shapes -> SCALE-Sim input.

Converts a transformer attention configuration (sequence length, hidden dimension,
number of heads) into the GEMM shape triples SCALE-Sim expects for each of the three
matmuls in scaled-dot-product attention, then into SCALE-Sim topology-CSV rows.

Design decisions (documented here since they aren't obvious, per plan §3):

1. **QKV projection is one combined GEMM**, not three separate ones. Shape
   (M=seq_len, K=hidden_dim, N=3*hidden_dim). This mirrors how real transformer
   implementations typically fuse the Q/K/V projection into a single matmul against
   a concatenated weight matrix, and keeps the array-mapping search space for this
   stage the same size regardless of num_heads.
2. **QK^T and score*V are per-head GEMMs**, computed on (seq_len, head_dim) slices,
   not full (seq_len, hidden_dim) ones — this is architecturally how attention is
   actually computed (each head attends independently). Since every head has an
   identical shape (head_dim = hidden_dim / num_heads is constant across heads),
   `GemmShape.repeat` records that the same (M,K,N) occurs `num_heads` times, rather
   than emitting num_heads separate identical entries.
3. hidden_dim must be exactly divisible by num_heads (standard transformer
   constraint) — enforced with a clear ValueError, not silent floor-division.

Unit tests for this module live at tests/test_workload_gen.py — the shape arithmetic
here is one of the two places (with metrics.py) a silent bug would invisibly corrupt
every downstream number (plan §3).
"""
from __future__ import annotations

from dataclasses import dataclass

import yaml


@dataclass(frozen=True)
class AttentionWorkload:
    """One transformer attention configuration to simulate.

    Attributes:
        seq_len: Sequence length (number of tokens).
        hidden_dim: Model hidden dimension.
        num_heads: Number of attention heads (hidden_dim must be divisible by this).
    """
    seq_len: int
    hidden_dim: int
    num_heads: int

    def __post_init__(self) -> None:
        if self.seq_len <= 0 or self.hidden_dim <= 0 or self.num_heads <= 0:
            raise ValueError(
                f"seq_len, hidden_dim, and num_heads must all be positive; got "
                f"seq_len={self.seq_len}, hidden_dim={self.hidden_dim}, num_heads={self.num_heads}."
            )
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads "
                f"({self.num_heads}); got head_dim = {self.hidden_dim / self.num_heads} (non-integer)."
            )

    @property
    def head_dim(self) -> int:
        """Per-head dimension. Guaranteed integer by __post_init__'s divisibility check."""
        return self.hidden_dim // self.num_heads


@dataclass(frozen=True)
class GemmShape:
    """One matmul's (M, K, N) dimensions plus how many times it recurs per forward pass.

    Attributes:
        m: Number of rows of the left operand / output.
        k: Shared (contracted) dimension.
        n: Number of columns of the right operand / output.
        name: Stage name — one of "qkv_projection", "qk_transpose", "score_times_v".
        repeat: Number of times this exact shape occurs per attention block (1 for
            the combined QKV projection, num_heads for the two per-head stages).
    """
    m: int
    k: int
    n: int
    name: str
    repeat: int = 1

    def __post_init__(self) -> None:
        if self.m <= 0 or self.k <= 0 or self.n <= 0 or self.repeat <= 0:
            raise ValueError(f"GemmShape dimensions and repeat must all be positive; got {self}.")


def workload_to_gemm_shapes(workload: AttentionWorkload) -> list[GemmShape]:
    """Derive the QKV/QK^T/score*V GEMM shapes for one attention workload.

    Args:
        workload: The transformer attention configuration to convert.

    Returns:
        Three GemmShape entries, in execution order: QKV projection (repeat=1),
        QK^T (repeat=num_heads), score*V (repeat=num_heads). See module docstring
        for why per-head stages are recorded once with a repeat count rather than
        expanded here.
    """
    head_dim = workload.head_dim
    return [
        GemmShape(m=workload.seq_len, k=workload.hidden_dim, n=3 * workload.hidden_dim,
                  name="qkv_projection", repeat=1),
        GemmShape(m=workload.seq_len, k=head_dim, n=workload.seq_len,
                  name="qk_transpose", repeat=workload.num_heads),
        GemmShape(m=workload.seq_len, k=workload.seq_len, n=head_dim,
                  name="score_times_v", repeat=workload.num_heads),
    ]


def gemm_shapes_to_topology_rows(shapes: list[GemmShape]) -> list[dict[str, int | str]]:
    """Convert GEMM shapes into SCALE-Sim's native GEMM-mode topology rows.

    SCALE-Sim (invoked with `-i gemm`) reads each topology row directly as
    (LayerName, M, N, K) — no conv-lowering needed. This was verified against the
    actual installed `scalesim` package (v3.0.0; see
    `scalesim/topology_utils.py::load_arrays_gemm`, read 2026-08-31): each row is
    `elems[0]=name, elems[1]=M, elems[2]=N, elems[3]=K`, i.e. **column order is
    M, N, K — not M, K, N** — confirmed by an actual end-to-end run, not assumed.

    A stage with repeat>1 (a per-head stage) is expanded into `repeat` numbered rows
    (e.g. "qk_transpose_head0", "qk_transpose_head1", ...) since SCALE-Sim's topology
    format has no native notion of a repeated layer.

    If you upgrade the installed `scalesim` version, re-verify this against
    `topology_utils.py::load_arrays_gemm` in that version before trusting output —
    this is exactly the kind of silent-corruption risk CODING_AGENT_PLAN.md §3 warns
    about, and the fix belongs in this one function.

    Args:
        shapes: Output of workload_to_gemm_shapes.

    Returns:
        One dict per topology row: {"LayerName": str, "M": int, "N": int, "K": int}.
    """
    rows: list[dict[str, int | str]] = []
    for shape in shapes:
        for i in range(shape.repeat):
            layer_name = shape.name if shape.repeat == 1 else f"{shape.name}_head{i}"
            rows.append({"LayerName": layer_name, "M": shape.m, "N": shape.n, "K": shape.k})
    return rows


def load_workload_yaml(path: str) -> AttentionWorkload:
    """Load an AttentionWorkload from a config/workloads/*.yaml file.

    Args:
        path: Path to a YAML file with at least `seq_len`, `hidden_dim`, `num_heads` keys.

    Returns:
        The parsed AttentionWorkload.

    Raises:
        ValueError: If a required key is missing.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    required = ("seq_len", "hidden_dim", "num_heads")
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"{path}: missing required key(s) {missing}; workload YAML needs {required}.")
    return AttentionWorkload(seq_len=data["seq_len"], hidden_dim=data["hidden_dim"], num_heads=data["num_heads"])
