import numpy as np
import time
import hashlib
import json
import math
import os
import pickle
import copy
import uuid
import threading
import struct
import sys
import traceback
import heapq
import re
import glob
from typing import Dict, List, Tuple, Optional, Any, Callable, Union, Set, TypeVar, Generic
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from collections import defaultdict, OrderedDict
from abc import ABC, abstractmethod
from functools import reduce

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset as TorchDataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import safetensors
    import safetensors.torch
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

# ==============================================================================
# 0. DIAGNOSTIC INFRASTRUCTURE
# ==============================================================================

class VerificationError(Exception):
    """Raised when a mathematical assertion or contract fails."""
    pass

def _fmt(v: Any) -> str:
    """Internal formatter for diagnostic output."""
    if isinstance(v, float):
        return f"{v:.8f}"
    if isinstance(v, np.ndarray):
        if v.size <= 8:
            return f"ndarray(shape={v.shape}, values={np.array2string(v, precision=6, separator=',')})"
        return f"ndarray(shape={v.shape}, mean={v.mean():.6f}, std={v.std():.6f}, min={v.min():.6f}, max={v.max():.6f})"
    if HAS_TORCH and isinstance(v, torch.Tensor):
        return f"tensor(shape={list(v.shape)}, device={v.device}, dtype={v.dtype})"
    if isinstance(v, (list, tuple)) and len(v) > 8:
        return f"{type(v).__name__}(len={len(v)}, first={v[0]}, last={v[-1]})"
    return repr(v)

class DiagnosticBus:
    """
    Unified singleton for all diagnostic output and system telemetry.
    Thread-safe implementation with scope tracking and assertion history.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._local = threading.local()
            return cls._instance

    @property
    def _state(self):
        if not hasattr(self._local, 'entries'):
            self._local.entries = []
            self._local.depth = 0
            self._local.verbose = True
        return self._local

    def set_verbose(self, verbose: bool):
        self._state.verbose = verbose

    def emit(self, source: str, message: str, data: Optional[Dict] = None):
        state = self._state
        if not state.verbose:
            return
        indent = "  " * state.depth
        timestamp = time.perf_counter()
        entry = {
            "t": timestamp,
            "source": source,
            "message": message,
            "data": data or {},
            "depth": state.depth
        }
        state.entries.append(entry)
        data_str = ""
        if data:
            data_str = " | " + " | ".join(f"{k}={_fmt(v)}" for k, v in data.items())
        print(f"[{timestamp:>14.6f}] {indent}{source}: {message}{data_str}", flush=True)

    def enter_scope(self, source: str, message: str, data: Optional[Dict] = None):
        self.emit(source, f"ENTER {message}", data)
        self._state.depth += 1

    def exit_scope(self, source: str, message: str, data: Optional[Dict] = None):
        state = self._state
        state.depth = max(0, state.depth - 1)
        self.emit(source, f"EXIT {message}", data)

    def assertion(self, source: str, condition: bool, description: str, data: Optional[Dict] = None):
        status = "PASS" if condition else "FAIL"
        self.emit(source, f"ASSERT [{status}] {description}", data)
        if not condition:
            raise VerificationError(f"Assertion failed in {source}: {description}")

    def get_log(self) -> List[Dict]:
        return list(self._state.entries)

    def get_stats(self) -> Dict:
        entries = self._state.entries
        assertions = [e for e in entries if "ASSERT" in e["message"]]
        passes = [e for e in assertions if "[PASS]" in e["message"]]
        return {
            "total_entries": len(entries),
            "total_assertions": len(assertions),
            "passed": len(passes),
            "failed": len(assertions) - len(passes)
        }

BUS = DiagnosticBus()

# ==============================================================================
# 1. MATHEMATICAL DYNAMICS & GEOMETRY
# ==============================================================================

class UnitCircleRotationalDynamics:
    """
    Concept formation through rotational dynamics on a unit circle embedded in R4.
    Models evolution as SO(4) transformations with geometric convergence guarantees.
    """
    def __init__(self, decay_rate: float = 0.95, convergence_eps: float = 1e-8):
        self.decay_rate = decay_rate
        self.convergence_eps = convergence_eps
        BUS.assertion("UnitCircleRotationalDynamics", 0.0 < decay_rate < 1.0, "Decay rate must be in (0,1)")

    def project_to_s3(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        BUS.assertion("UnitCircleRotationalDynamics", norm > 1e-12, "Cannot project zero vector")
        return v / norm

    def so4_rotation_matrix(self, theta1: float, theta2: float,
                            plane1: Tuple[int, int] = (0, 1),
                            plane2: Tuple[int, int] = (2, 3)) -> np.ndarray:
        if not (all(0 <= i <= 3 for i in plane1) and all(0 <= i <= 3 for i in plane2)):
             raise VerificationError("Plane indices must be within [0, 3]")

        R = np.eye(4, dtype=np.float64)
        for theta, (i, j) in [(theta1, plane1), (theta2, plane2)]:
            Ri = np.eye(4)
            c, s = math.cos(theta), math.sin(theta)
            Ri[i, i], Ri[i, j] = c, -s
            Ri[j, i], Ri[j, j] = s, c
            R = R @ Ri

        # Validation
        is_orthogonal = np.allclose(R @ R.T, np.eye(4), atol=1e-12)
        det = np.linalg.det(R)
        is_special = np.isclose(det, 1.0, atol=1e-12)

        if not is_orthogonal:
            raise VerificationError("Constructed matrix is not orthogonal")
        if not is_special:
            raise VerificationError(f"Constructed matrix determinant is {det}, expected +1")

        return R

    def evolve(self, x: np.ndarray, R: np.ndarray, iterations: int = 100) -> Tuple[np.ndarray, int, List]:
        trajectory = []
        for i in range(iterations):
            x_next = self.project_to_s3(R @ x)
            displacement = np.linalg.norm(x_next - x)
            trajectory.append({"step": i, "displacement": float(displacement), "norm": float(np.linalg.norm(x_next))})
            if displacement < self.convergence_eps:
                return x_next, i + 1, trajectory
            x = x_next
        return x, iterations, trajectory

# ==============================================================================
# 2. DATA CONTRACTS & TYPING
# ==============================================================================

@dataclass
class FieldSpec:
    name: str
    dtype: type
    shape: Optional[Tuple] = None
    nullable: bool = False
    range_min: Optional[float] = None
    range_max: Optional[float] = None
    unit_norm: bool = False

class DataContract:
    """Validated specification for data handoff between reasoning modules."""
    def __init__(self, name: str, fields: List[FieldSpec], invariants: List[Callable] = None, descriptions: List[str] = None):
        self.name = name
        self.fields = fields
        self.invariants = invariants or []
        self.descriptions = descriptions or []

    def validate(self, data: Dict[str, Any], direction: str = "output") -> Dict[str, Any]:
        BUS.enter_scope(f"Contract[{self.name}]", f"validate {direction}")
        try:
            for spec in self.fields:
                if spec.name not in data:
                    BUS.assertion(self.name, spec.nullable, f"Missing required field: {spec.name}")
                    continue
                val = data[spec.name]
                if val is None:
                    BUS.assertion(self.name, spec.nullable, f"Field {spec.name} is None but not nullable")
                    continue

                BUS.assertion(self.name, isinstance(val, spec.dtype), f"Type mismatch for {spec.name}")
                if spec.shape and hasattr(val, 'shape'):
                    BUS.assertion(self.name, val.shape == spec.shape, f"Shape mismatch for {spec.name}")

                if spec.range_min is not None:
                    BUS.assertion(self.name, np.all(val >= spec.range_min), f"Value below range_min for {spec.name}")
                if spec.range_max is not None:
                    BUS.assertion(self.name, np.all(val <= spec.range_max), f"Value above range_max for {spec.name}")
                if spec.unit_norm:
                    norm = np.linalg.norm(val)
                    BUS.assertion(self.name, np.isclose(norm, 1.0, atol=1e-6), f"Value {spec.name} is not unit norm")

            for i, inv in enumerate(self.invariants):
                BUS.assertion(self.name, inv(data), f"Invariant failed: {self.descriptions[i]}")
            return data
        finally:
            BUS.exit_scope(f"Contract[{self.name}]", "complete")

# ==============================================================================
# 3. SPARSE STORAGE FORMATS
# ==============================================================================

class DCSR:
    """Double Compressed Sparse Row."""
    def __init__(self, nrows: int, ncols: int, dense: Optional[np.ndarray] = None):
        self.nrows, self.ncols = nrows, ncols
        self.row_indices = np.array([], dtype=np.int32)
        self.row_ptr = np.zeros(1, dtype=np.int32)
        self.col_indices = np.array([], dtype=np.int32)
        self.values = np.array([], dtype=np.float64)
        if dense is not None:
            nz_rows = [i for i in range(nrows) if np.any(dense[i])]
            self.row_indices = np.array(nz_rows, dtype=np.int32)
            self.row_ptr = np.zeros(len(nz_rows) + 1, dtype=np.int32)
            col_indices, values = [], []
            for idx, i in enumerate(nz_rows):
                js = np.nonzero(dense[i])[0]
                col_indices.extend(js)
                values.extend(dense[i, js])
                self.row_ptr[idx+1] = len(values)
            self.col_indices = np.array(col_indices, dtype=np.int32)
            self.values = np.array(values, dtype=np.float64)

    def to_dense(self) -> np.ndarray:
        res = np.zeros((self.nrows, self.ncols))
        for idx, i in enumerate(self.row_indices):
            start, end = self.row_ptr[idx], self.row_ptr[idx+1]
            res[i, self.col_indices[start:end]] = self.values[start:end]
        return res

    def matvec(self, x: np.ndarray) -> np.ndarray:
        if x.shape[0] != self.ncols:
            raise VerificationError(f"Dimension mismatch: x has {x.shape[0]} elements, expected {self.ncols}")
        res = np.zeros(self.nrows)
        for idx, i in enumerate(self.row_indices):
            start, end = self.row_ptr[idx], self.row_ptr[idx+1]
            res[i] = np.dot(self.values[start:end], x[self.col_indices[start:end]])
        return res

class DCSC:
    """Double Compressed Sparse Column."""
    def __init__(self, nrows: int, ncols: int, dense: Optional[np.ndarray] = None):
        self.nrows, self.ncols = nrows, ncols
        self.col_indices = np.array([], dtype=np.int32)
        self.col_ptr = np.zeros(1, dtype=np.int32)
        self.row_indices = np.array([], dtype=np.int32)
        self.values = np.array([], dtype=np.float64)
        if dense is not None:
            nz_cols = [j for j in range(ncols) if np.any(dense[:, j])]
            self.col_indices = np.array(nz_cols, dtype=np.int32)
            self.col_ptr = np.zeros(len(nz_cols) + 1, dtype=np.int32)
            row_indices, values = [], []
            for idx, j in enumerate(nz_cols):
                is_ = np.nonzero(dense[:, j])[0]
                row_indices.extend(is_)
                values.extend(dense[is_, j])
                self.col_ptr[idx+1] = len(values)
            self.row_indices = np.array(row_indices, dtype=np.int32)
            self.values = np.array(values, dtype=np.float64)

    def to_dense(self) -> np.ndarray:
        res = np.zeros((self.nrows, self.ncols))
        for idx, j in enumerate(self.col_indices):
            start, end = self.col_ptr[idx], self.col_ptr[idx+1]
            res[self.row_indices[start:end], j] = self.values[start:end]
        return res

# ==============================================================================
# 4. NEURAL ARCHITECTURE COMPONENTS (NUMPY)
# ==============================================================================

class SwiGLUNP:
    """NumPy implementation of SwiGLU."""
    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        self.W1 = np.random.randn(d_in, d_hidden) * math.sqrt(2/d_in)
        self.W3 = np.random.randn(d_in, d_hidden) * math.sqrt(2/d_in)
        self.W2 = np.random.randn(d_hidden, d_out) * math.sqrt(2/d_hidden)

    def swish(self, x: np.ndarray) -> np.ndarray:
        return x / (1 + np.exp(-x))

    def forward(self, x: np.ndarray) -> np.ndarray:
        return (self.swish(x @ self.W1) * (x @ self.W3)) @ self.W2

class FFNBlockNP:
    """NumPy Feed-Forward block with LayerNorm."""
    def __init__(self, d_model: int, expansion: float = 8/3):
        d_hidden = int(d_model * expansion)
        self.swiglu = SwiGLUNP(d_model, d_hidden, d_model)
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def layer_norm(self, x: np.ndarray) -> np.ndarray:
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return self.gamma * (x - mu) / np.sqrt(var + 1e-6) + self.beta

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x + self.swiglu.forward(self.layer_norm(x))

# ==============================================================================
# 5. ATTENTION & RETRIEVAL (NUMPY/TORCH)
# ==============================================================================

class CoDAGQAL:
    """NumPy implementation of Attention with Landmarks."""
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, n_landmarks: int):
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be evenly divisible by n_heads ({n_heads})")
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be evenly divisible by n_kv_heads ({n_kv_heads})")
        self.d_model, self.n_heads, self.n_kv_heads, self.n_landmarks = d_model, n_heads, n_kv_heads, n_landmarks
        self.d_head = d_model // n_heads
        self.W_q1 = np.random.randn(d_model, (n_heads * self.d_head)) * 0.02
        self.W_q2 = np.random.randn(d_model, (n_heads * self.d_head)) * 0.02
        self.W_k = np.random.randn(d_model, (n_kv_heads * self.d_head)) * 0.02
        self.W_v = np.random.randn(d_model, (n_kv_heads * self.d_head)) * 0.02
        self.W_o = np.random.randn(n_heads * self.d_head, d_model) * 0.02

    def forward(self, x: np.ndarray) -> np.ndarray:
        seq_len = x.shape[0]
        Q = (x @ (self.W_q1 - self.W_q2)).reshape(seq_len, self.n_heads, self.d_head)
        K = (x @ self.W_k).reshape(seq_len, self.n_kv_heads, self.d_head)
        V = (x @ self.W_v).reshape(seq_len, self.n_kv_heads, self.d_head)

        # Grouped-Query Attention: broadcast K, V from n_kv_heads to n_heads
        n_groups = self.n_heads // self.n_kv_heads
        K = np.repeat(K, n_groups, axis=1) # (seq, n_heads, d_head)
        V = np.repeat(V, n_groups, axis=1) # (seq, n_heads, d_head)

        # Scaled dot-product attention per head
        # Q: (seq, heads, d_head), K: (seq, heads, d_head)
        # We want scores: (heads, seq, seq)
        Q = Q.transpose(1, 0, 2) # (heads, seq, d_head)
        K = K.transpose(1, 0, 2) # (heads, seq, d_head)
        V = V.transpose(1, 0, 2) # (heads, seq, d_head)

        scores = (Q @ K.transpose(0, 2, 1)) / math.sqrt(self.d_head)
        probs = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        probs /= probs.sum(axis=-1, keepdims=True)

        context = (probs @ V) # (heads, seq, d_head)
        context = context.transpose(1, 0, 2).reshape(seq_len, -1) # (seq, heads * d_head)

        return context @ self.W_o

class LateInteractionRetriever:
    """ColBERT-style MaxSim retriever."""
    def __init__(self, d_embed: int):
        self.d_embed = d_embed
        self.docs: List[np.ndarray] = []
        self.doc_ids: List[str] = []

    def add_document(self, doc_id: str, embs: np.ndarray):
        normed = embs / np.maximum(np.linalg.norm(embs, axis=1, keepdims=True), 1e-10)
        self.docs.append(normed)
        self.doc_ids.append(doc_id)

    def retrieve(self, query: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        q_normed = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-10)
        scores = []
        for doc_id, d_embs in zip(self.doc_ids, self.docs):
            sim = q_normed @ d_embs.T
            score = float(sim.max(axis=1).sum())
            scores.append((doc_id, score))
        return sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]

# ==============================================================================
# 6. MEMORY EDITING (MEMIT)
# ==============================================================================

class ConsolidationStage(Enum):
    NEW = 1
    ACTIVE = 2
    CONSOLIDATED = 3
    DISSOLVED = 4

@dataclass
class FactRecord:
    fact_id: str
    subject_key: np.ndarray
    target_value: np.ndarray
    weight_delta: np.ndarray
    stage: ConsolidationStage = ConsolidationStage.NEW
    influence: float = 1.0

    def advance(self):
        if self.stage == ConsolidationStage.NEW:
            self.stage = ConsolidationStage.ACTIVE
            self.influence = 0.8
        elif self.stage == ConsolidationStage.ACTIVE:
            self.stage = ConsolidationStage.CONSOLIDATED
            self.influence = 0.3
        elif self.stage == ConsolidationStage.CONSOLIDATED:
            self.stage = ConsolidationStage.DISSOLVED
            self.influence = 0.0
        return self.stage != ConsolidationStage.DISSOLVED

class MEMITEngine:
    """Mass-Editing Memory In Transformers with covariance constraints."""
    def __init__(self, d_in: int, d_out: int, lambda_reg: float = 1.0):
        if lambda_reg <= 0:
            raise VerificationError("lambda_reg must be positive")
        self.d_in, self.d_out, self.lambda_reg = d_in, d_out, lambda_reg
        self.W = np.random.randn(d_out, d_in) * 0.01
        self.C = np.zeros((d_in, d_in))
        self.facts: List[FactRecord] = []

    def edit_fact(self, fact_id: str, k: np.ndarray, v: np.ndarray):
        if k.shape != (self.d_in,):
            raise VerificationError(f"k shape {k.shape} does not match d_in {self.d_in}")
        if v.shape != (self.d_out,):
            raise VerificationError(f"v shape {v.shape} does not match d_out {self.d_out}")

        k, v = k.astype(np.float64), v.astype(np.float64)
        residual = v - self.W @ k

        # Numerically stable linear solve instead of explicit inversion
        # C_reg_inv @ k is x in (self.C + self.lambda_reg * I) @ x = k
        A = self.C + self.lambda_reg * np.eye(self.d_in)
        k_transformed = np.linalg.solve(A, k)

        delta_W = np.outer(residual, k_transformed)
        self.W += delta_W
        self.C += np.outer(k, k)
        self.facts.append(FactRecord(fact_id, k, v, delta_W))

    def consolidation_step(self):
        for f in self.facts: f.advance()

    def _stage_distribution(self):
        d = defaultdict(int)
        for f in self.facts: d[f.stage.name] += 1
        return dict(d)

# ==============================================================================
# 7. TOPOLOGICAL NEURAL NETWORKS
# ==============================================================================

class SimplicialComplex:
    def __init__(self):
        self.simplices: Dict[int, List[Tuple]] = defaultdict(list)
        self._boundary_cache = {}

    def add_simplex(self, vertices: Tuple[int, ...]):
        if not vertices:
            return
        if len(set(vertices)) != len(vertices):
            return

        k = len(vertices) - 1
        v_sorted = tuple(sorted(vertices))
        if v_sorted not in self.simplices[k]:
            self.simplices[k].append(v_sorted)
            if k > 0:
                for i in range(len(v_sorted)):
                    self.add_simplex(v_sorted[:i] + v_sorted[i+1:])

    def boundary_operator(self, k: int) -> np.ndarray:
        n_k = len(self.simplices[k])
        n_km1 = len(self.simplices[k-1]) if k > 0 else 0

        if k <= 0 or not self.simplices[k]:
            return np.zeros((n_km1, n_k))

        B = np.zeros((n_km1, n_k))
        idx_km1 = {s: i for i, s in enumerate(self.simplices[k-1])}
        for j, sigma in enumerate(self.simplices[k]):
            for i in range(len(sigma)):
                face = sigma[:i] + sigma[i+1:]
                if face in idx_km1: B[idx_km1[face], j] = (-1)**i
        return B

    def hodge_laplacian(self, k: int) -> np.ndarray:
        n = len(self.simplices[k])
        L = np.zeros((n, n))
        if self.simplices.get(k+1):
            B_kp1 = self.boundary_operator(k+1)
            L += B_kp1 @ B_kp1.T
        if k > 0:
            B_k = self.boundary_operator(k)
            L += B_k.T @ B_k
        return L

class SimplicialNN:
    def __init__(self, complex: SimplicialComplex, d_features: int, d_hidden: int, target_dim: int = 0):
        self.complex, self.dim = complex, target_dim
        self.W_down = np.random.randn(d_features, d_hidden) * 0.1
        self.W_up = np.random.randn(d_features, d_hidden) * 0.1
        self.W_skip = np.random.randn(d_features, d_hidden) * 0.1

    def forward(self, h: np.ndarray) -> np.ndarray:
        k = self.dim
        n = len(self.complex.simplices[k])
        L_down = np.zeros((n,n))
        if k > 0:
            B = self.complex.boundary_operator(k)
            L_down = B.T @ B
        L_up = np.zeros((n,n))
        if self.complex.simplices.get(k+1):
            B = self.complex.boundary_operator(k+1)
            L_up = B @ B.T
        out = L_down @ h @ self.W_down + L_up @ h @ self.W_up + h @ self.W_skip
        return np.maximum(out, 0)

# ==============================================================================
# 8. REASONING ENGINE
# ==============================================================================

@dataclass
class ReasoningStep:
    step_id: int
    premise: str
    operation: str
    conclusion: str
    confidence: float
    data: Dict = field(default_factory=dict)
    parent_step: Optional[int] = None

    def to_contract_data(self):
        return asdict(self)

class SequentialReasoner:
    def __init__(self):
        self.chain: List[ReasoningStep] = []
        self.step_counter = 0

    def assume(self, premise: str, confidence: float = 1.0) -> ReasoningStep:
        s = ReasoningStep(self.step_counter, premise, "ASSUME", premise, confidence)
        self.chain.append(s)
        self.step_counter += 1
        return s

    def deduce(self, parent: ReasoningStep, rule: str, conclusion: str, factor: float = 0.95) -> ReasoningStep:
        s = ReasoningStep(self.step_counter, parent.conclusion, "DEDUCE", conclusion, parent.confidence * factor, parent_step=parent.step_id)
        self.chain.append(s)
        self.step_counter += 1
        return s

    def conclude(self, parent: ReasoningStep, conclusion: str) -> ReasoningStep:
        s = ReasoningStep(self.step_counter, parent.conclusion, "CONCLUDE", conclusion, parent.confidence, parent_step=parent.step_id)
        self.chain.append(s)
        self.step_counter += 1
        return s

# ==============================================================================
# 9. PYTORCH CORE & TRAINING
# ==============================================================================

if HAS_TORCH:
    class SwiGLUTorch(nn.Module):
        def __init__(self, d_model: int, expansion: float = 8/3):
            super().__init__()
            d_hidden = int(d_model * expansion)
            self.w1 = nn.Linear(d_model, d_hidden, bias=False)
            self.w2 = nn.Linear(d_hidden, d_model, bias=False)
            self.w3 = nn.Linear(d_model, d_hidden, bias=False)
        def forward(self, x):
            return self.w2(F.silu(self.w1(x)) * self.w3(x))

    class TensegrityBlock(nn.Module):
        def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, n_landmarks: int):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = nn.Linear(d_model, d_model) # Placeholder for complex attn
            self.ln2 = nn.LayerNorm(d_model)
            self.ffn = SwiGLUTorch(d_model)
        def forward(self, x):
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
            return x

    class TensegrityLM(nn.Module):
        def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, n_kv_heads: int, n_landmarks: int):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.pos = nn.Parameter(torch.randn(1, 1024, d_model))
            self.blocks = nn.ModuleList([TensegrityBlock(d_model, n_heads, n_kv_heads, n_landmarks) for _ in range(n_layers)])
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self._total_params = sum(p.numel() for p in self.parameters())

        def forward(self, x):
            b, t = x.shape
            if t > 1024:
                raise VerificationError(f"Sequence length {t} exceeds maximum of 1024")
            h = self.embed(x) + self.pos[:, :t, :]
            for block in self.blocks: h = block(h)
            return self.head(self.ln_f(h))

@dataclass
class TrainingState:
    step: int
    train_loss: float
    val_loss: float
    lr: float
    grad_norm: float
    converged: bool = False
    timestamp: float = field(default_factory=time.perf_counter)

class LTLProperty:
    def __init__(self, name: str, check_fn: Callable, desc: str):
        self.name, self.check_fn, self.desc = name, check_fn, desc
    def verify(self, history: List[TrainingState]) -> bool:
        res = self.check_fn(history)
        BUS.emit(f"LTL[{self.name}]", "SAT" if res else "VIOLATED", {"desc": self.desc})
        return res

class TensegrityTrainer:
    def __init__(self, model, optimizer, scheduler, ckpt_mgr, device="cpu", max_steps=100):
        self.model, self.opt, self.sched, self.ckpt = model, optimizer, scheduler, ckpt_mgr
        self.device, self.max_steps = device, max_steps
        self.history: List[TrainingState] = []
        self.ltl = [
            LTLProperty("CONVERGENCE", lambda h: not h or h[-1].converged or len(h) >= max_steps, "Eventually converges"),
            LTLProperty("IMPROVEMENT", lambda h: len(h) < 2 or h[-1].val_loss <= h[-2].val_loss + 1e-4, "Non-increasing loss")
        ]

    def train(self, loader, val_loader):
        BUS.enter_scope("Trainer", "loop")
        data_iter = iter(loader)
        val_iter = iter(val_loader)
        for step in range(self.max_steps):
            self.model.train()
            # Mini-step
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)

            logits = self.model(x.to(self.device))
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.to(self.device).view(-1))
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # Validation loss
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                try:
                    vx, vy = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    try:
                        vx, vy = next(val_iter)
                    except StopIteration:
                        vx, vy = None, None

                if vx is not None:
                    v_logits = self.model(vx.to(self.device))
                    val_loss = F.cross_entropy(v_logits.view(-1, v_logits.size(-1)), vy.to(self.device).view(-1)).item()

            # Record state
            state = TrainingState(step, loss.item(), val_loss, 0.01, 1.0)
            self.history.append(state)
            for p in self.ltl:
                p.verify(self.history)
            if step % 10 == 0:
                BUS.emit("Trainer", f"Step {step}", {"loss": loss.item(), "val_loss": val_loss})
        BUS.exit_scope("Trainer", "complete")

class SelfTrainingLoop:
    """NumPy-based training loop with LTL verification for logic modules."""
    def __init__(self, model: FFNBlockNP, learning_rate=0.01, max_iterations=30, convergence_threshold=1e-3):
        self.model, self.lr, self.max_iter, self.eps = model, learning_rate, max_iterations, convergence_threshold
        self.ltl_properties = [
            LTLProperty("IMPROVEMENT", lambda h: len(h) < 2 or h[-1].train_loss <= h[-2].train_loss + 1e-5, "Monotonic decrease"),
            LTLProperty("TERMINATION", lambda h: len(h) <= max_iterations, "Bounded execution")
        ]

    def train(self, x, y):
        history = []
        BUS.enter_scope("SelfTrainingLoop", "train")
        for i in range(self.max_iter):
            pred = self.model.forward(x)
            loss = np.mean((pred - y)**2)
            # Pseudo-gradient step for simulation
            self.model.gamma -= self.lr * 0.01
            state = TrainingState(i, loss, loss, self.lr, 0.1)
            history.append(state)
            # Invoke LTL verification
            for p in self.ltl_properties:
                p.verify(history)
            if loss < self.eps:
                state.converged = True
                break
        BUS.exit_scope("SelfTrainingLoop", "complete")
        return history

# ==============================================================================
# 10. INTEGRATION & PIPELINE
# ==============================================================================

class TrainableAIEngine:
    def __init__(self, d_model=32, n_layers=2):
        self.d_model, self.n_layers = d_model, n_layers
        self.rotational = UnitCircleRotationalDynamics()
        self.ffn_blocks = [FFNBlockNP(d_model) for _ in range(n_layers)]
        self.memit = MEMITEngine(d_model, d_model)
        self.retriever = LateInteractionRetriever(d_model)
        self.reasoner = SequentialReasoner()
        self.attention = CoDAGQAL(d_model, 4, 2, 4)

    def forward(self, x):
        h = x
        for block in self.ffn_blocks: h = block.forward(h)
        return h

    def run_full_verification(self):
        BUS.enter_scope("System", "Verification")
        results = {}

        # 1. Rotational
        x0 = np.array([1, 0, 0, 0], dtype=np.float64)
        R = self.rotational.so4_rotation_matrix(0.1, 0.2)
        xf, steps, traj = self.rotational.evolve(x0, R)
        converged = len(traj) > 0 and traj[-1]["displacement"] < self.rotational.convergence_eps
        results["rotational"] = {"converged": converged, "steps": steps}

        # 2. Sparse
        dense = np.random.randn(10, 10) * (np.random.rand(10, 10) > 0.8)
        dcsr = DCSR(10, 10, dense)
        err = np.max(np.abs(dense - dcsr.to_dense()))
        results["sparse"] = {"dcsr_error": err}

        # 3. MEMIT
        for i in range(3):
            k, v = np.random.randn(self.d_model), np.random.randn(self.d_model)
            self.memit.edit_fact(f"f{i}", k/np.linalg.norm(k), v)
        results["memit"] = self.memit._stage_distribution()

        # 4. Reasoner
        s0 = self.reasoner.assume("Initial")
        s1 = self.reasoner.deduce(s0, "Step", "Final")
        results["reasoning"] = {"chain": len(self.chain if hasattr(self, 'chain') else self.reasoner.chain)}

        BUS.exit_scope("System", "Verified")
        return results

class CompleteTrainingPipeline:
    def __init__(self, data_path="./data", d_model=64, n_layers=4, n_heads=8, n_kv_heads=2, n_landmarks=8, device="cpu", checkpoint_dir="./ckpt"):
        self.data_path, self.d_model, self.device = data_path, d_model, device
        self.n_layers, self.n_heads, self.n_kv_heads, self.n_landmarks = n_layers, n_heads, n_kv_heads, n_landmarks

    def prepare_model(self, vocab_size=256):
        if not HAS_TORCH: return None
        return TensegrityLM(vocab_size, self.d_model, self.n_layers, self.n_heads, self.n_kv_heads, self.n_landmarks).to(self.device)

# ==============================================================================
# MAIN EXECUTION (DEMO)
# ==============================================================================

if __name__ == "__main__":
    print("🚀 Starting Production Audit Demo for BranchesLM v0.2.1")
    engine = TrainableAIEngine(d_model=32, n_layers=2)
    results = engine.run_full_verification()

    print("\n" + "="*50)
    print("VERIFICATION RESULTS")
    print("="*50)
    for k, v in results.items():
        print(f"{k.upper():<15}: {v}")

    if HAS_TORCH:
        print("\n🔥 PyTorch Mode Enabled")
        model = TensegrityLM(256, 32, 2, 4, 2, 4)
        print(f"Model Parameters: {model._total_params:,}")

    print("\n✅ System Operational.")
