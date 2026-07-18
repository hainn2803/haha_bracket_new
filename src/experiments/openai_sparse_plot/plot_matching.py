from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .schema import EffectSignatureTable


CostMode = Literal["squared", "cosine", "centered_cosine"]


@dataclass(frozen=True)
class MatchingResult:
    cost: torch.Tensor
    coupling: torch.Tensor
    method: str
    row_labels: tuple[str, ...]
    col_labels: tuple[str, ...]

    def top_matches(self, top_k: int = 3) -> dict[str, list[tuple[str, float]]]:
        out: dict[str, list[tuple[str, float]]] = {}
        k = min(int(top_k), self.coupling.size(1))
        for i, row_label in enumerate(self.row_labels):
            vals, idx = torch.topk(self.coupling[i], k=k)
            out[row_label] = [(self.col_labels[int(j)], float(v)) for v, j in zip(vals, idx)]
        return out


def _as_tensor(rows: tuple[tuple[float, ...], ...]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32)


def cost_matrix(
    abstract_signatures: torch.Tensor,
    neural_signatures: torch.Tensor,
    *,
    mode: CostMode = "centered_cosine",
) -> torch.Tensor:
    if abstract_signatures.ndim != 2 or neural_signatures.ndim != 2:
        raise ValueError("signature tensors must be rank-2")
    if abstract_signatures.size(1) != neural_signatures.size(1):
        raise ValueError("signature feature dimensions must match")
    x = abstract_signatures.to(torch.float32)
    y = neural_signatures.to(torch.float32)
    if mode == "squared":
        diff = x[:, None, :] - y[None, :, :]
        return (diff * diff).sum(dim=-1)
    if mode == "centered_cosine":
        x = x - x.mean(dim=1, keepdim=True)
        y = y - y.mean(dim=1, keepdim=True)
        mode = "cosine"
    if mode == "cosine":
        x = x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)
        y = y / y.norm(dim=1, keepdim=True).clamp_min(1e-12)
        return 1.0 - x @ y.t()
    raise ValueError(f"unknown cost mode: {mode!r}")


def rowwise_argmin(cost: torch.Tensor) -> torch.Tensor:
    if cost.ndim != 2:
        raise ValueError("cost must be rank-2")
    coupling = torch.zeros_like(cost, dtype=torch.float32)
    coupling[torch.arange(cost.size(0)), torch.argmin(cost, dim=1)] = 1.0
    return coupling


def rowwise_softmax(cost: torch.Tensor, *, temperature: float = 0.1) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    return torch.softmax(-cost.to(torch.float32) / float(temperature), dim=1)


def sinkhorn_uniform(cost: torch.Tensor, *, epsilon: float = 0.1, n_iter: int = 200) -> torch.Tensor:
    if epsilon <= 0 or n_iter <= 0:
        raise ValueError("epsilon and n_iter must be > 0")
    m, n = cost.shape
    a = torch.full((m,), 1.0 / m, dtype=torch.float32)
    b = torch.full((n,), 1.0 / n, dtype=torch.float32)
    kernel = torch.exp(-cost.to(torch.float32) / float(epsilon)).clamp_min(1e-30)
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(int(n_iter)):
        u = a / (kernel @ v).clamp_min(1e-30)
        v = b / (kernel.t() @ u).clamp_min(1e-30)
    return (u[:, None] * kernel) * v[None, :]


def sinkhorn_one_sided_uot(
    cost: torch.Tensor,
    *,
    epsilon: float = 0.1,
    beta_neural: float = 0.1,
    n_iter: int = 200,
) -> torch.Tensor:
    if epsilon <= 0 or beta_neural <= 0 or n_iter <= 0:
        raise ValueError("epsilon, beta_neural, and n_iter must be > 0")
    m, n = cost.shape
    a = torch.full((m,), 1.0 / m, dtype=torch.float32)
    b = torch.full((n,), 1.0 / n, dtype=torch.float32)
    kernel = torch.exp(-cost.to(torch.float32) / float(epsilon)).clamp_min(1e-30)
    rho_b = float(beta_neural / (beta_neural + epsilon))
    u = torch.ones_like(a)
    v = torch.ones_like(b)
    for _ in range(int(n_iter)):
        u = a / (kernel @ v).clamp_min(1e-30)
        v = (b / (kernel.t() @ u).clamp_min(1e-30)).pow(rho_b)
    coupling = (u[:, None] * kernel) * v[None, :]
    return coupling / coupling.sum(dim=1, keepdim=True).clamp_min(1e-30)


def fit_matching(
    table: EffectSignatureTable,
    *,
    cost_mode: CostMode = "centered_cosine",
    method: Literal["argmin", "softmax", "sinkhorn", "uot"] = "uot",
    temperature: float = 0.1,
    epsilon: float = 0.1,
    beta_neural: float = 0.1,
    n_iter: int = 200,
) -> MatchingResult:
    table.validate()
    cost = cost_matrix(_as_tensor(table.abstract_signatures), _as_tensor(table.neural_signatures), mode=cost_mode)
    if method == "argmin":
        coupling = rowwise_argmin(cost)
    elif method == "softmax":
        coupling = rowwise_softmax(cost, temperature=temperature)
    elif method == "sinkhorn":
        coupling = sinkhorn_uniform(cost, epsilon=epsilon, n_iter=n_iter)
    elif method == "uot":
        coupling = sinkhorn_one_sided_uot(cost, epsilon=epsilon, beta_neural=beta_neural, n_iter=n_iter)
    else:
        raise ValueError(f"unknown matching method: {method!r}")
    return MatchingResult(
        cost=cost.detach().cpu(),
        coupling=coupling.detach().cpu(),
        method=method,
        row_labels=table.abstract_variable_ids,
        col_labels=table.neural_site_ids,
    )
