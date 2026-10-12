"""Temperature scaling (TRD §4.6): one scalar T fitted on validation logits by minimizing NLL."""

import numpy as np
import torch


def fit_temperature(logits: np.ndarray, y: np.ndarray, max_iter: int = 200) -> float:
    logits_t = torch.as_tensor(logits, dtype=torch.float64)
    y_t = torch.as_tensor(y, dtype=torch.long)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)  # optimize log T (keeps T > 0)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits_t / log_t.exp(), y_t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().clamp(0.05, 20.0))


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = logits / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)
