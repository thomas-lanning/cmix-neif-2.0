"""Fit a small integer diagonal metric for reranking four-bit semantic-cache keys."""
import numpy as np


def moments(keys, query, groups):
    width = query.size // groups
    key = np.rint(keys.astype(np.float64) * (7 / 127)).reshape(-1, groups, width)
    q = query.astype(np.float64).reshape(groups, width)
    return (np.einsum("ngd,gd->ng", key, q),
            np.einsum("ngd,ngd->ng", key, key), np.einsum("gd,gd->g", q, q))


def similarity_and_gradient(parts, weights):
    dot, key_energy, query_energy = parts
    a = np.maximum(key_energy @ weights, 1e-30)
    b = max(query_energy @ weights, 1e-30)
    norm = np.sqrt(a*b)
    sim = (dot @ weights)/norm
    # Derivative of weighted cosine with respect to log(weight).
    grad = weights * (dot/norm[:, None] - .5*sim[:, None] *
                      (key_energy/a[:, None] + query_energy/b))
    return sim, grad


def fit(rows, groups, steps=16):
    """Train on chronological fitting queries only. Each group costs one byte."""
    examples = [(moments(r["x"],r["q"],groups), r["values"] == r["target"], r["tf"]) for r in rows]
    theta = np.zeros(groups)
    mean = np.zeros(groups)
    variance = np.zeros(groups)
    for step in range(1, steps+1):
        grad = np.zeros(groups)
        weights = np.exp(theta)
        for parts, hit, transformer in examples:
            sim, jacobian = similarity_and_gradient(parts, weights)
            top = np.argsort(-sim,kind="stable")[:32]
            scores = sim[top]
            mass = np.exp((scores - scores.max())/.020)
            mass /= mass.sum()
            target_mass = hit[top] @ mass
            mixed = max(.85*transformer + .15*target_mass, 1e-30)
            grad -= (.15/(mixed*.020)) * ((hit[top]-target_mass)*mass) @ jacobian[top]
        grad = grad/len(examples) + .025*theta
        grad -= grad.mean()
        mean = .9*mean + .1*grad
        variance = .999*variance + .001*grad**2
        theta -= .025 * (mean/(1-.9**step)) / (np.sqrt(variance/(1-.999**step)) + 1e-8)
        theta = np.clip(theta-theta.mean(), -1.3, 1.3)
    return np.clip(np.rint(32*np.exp(theta)), 1, 255).astype(int).tolist()
