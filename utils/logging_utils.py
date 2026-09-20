"""Scalar logging and gradient diagnostic helpers."""

from collections import defaultdict
import math

import torch


def _resolve_map_log_key(map_type_name, map_writers):
    key = str(map_type_name).strip().lower().replace("-", "_")
    return key if key in map_writers else "global"


def _resolve_tb_writer(map_type_name, writer, map_writers):
    key = _resolve_map_log_key(map_type_name, map_writers)
    if key == "global":
        return writer
    return map_writers[key]


def smooth_dict(ori_dict, scaler_q_by_map, map_log_key="global"):
    q = scaler_q_by_map[map_log_key]
    for k, v in ori_dict.items():
        if isinstance(v, torch.Tensor):
            v = v.item()
        q[k].append(float(v))


def release_autograd_graph_references(namespace):
    """Drop module-scope references that keep a completed iteration graph alive.

    The training loop currently runs at module scope, so ordinary loop locals
    remain in ``globals()`` until the same names are overwritten.  In
    particular, diagnostic dictionaries and LGN hypergradient tuples can keep
    an entire higher-order graph alive while the next rollout is being built.
    Only tensors with a ``grad_fn`` (or containers holding such tensors) are
    released; parameters, detached tensors, modules, optimizers, and caches are
    left untouched.
    """

    def holds_graph(value, seen):
        if isinstance(value, torch.Tensor):
            return value.grad_fn is not None
        if not isinstance(value, (dict, list, tuple)):
            return False
        value_id = id(value)
        if value_id in seen:
            return False
        seen.add(value_id)
        items = value.values() if isinstance(value, dict) else value
        return any(holds_graph(item, seen) for item in items)

    released = []
    for name, value in tuple(namespace.items()):
        if name.startswith("__"):
            continue
        if holds_graph(value, set()):
            namespace[name] = None
            released.append(name)
    return tuple(released)


def is_artifact_save_iter(i, args):
    """Use one interval for checkpoints, trajectories, and videos."""
    interval = int(args.artifact_save_interval)
    return interval > 0 and (i + 1) % interval == 0


def is_debug_tb_step(step, args):
    interval = int(args.debug_tb_interval)
    return interval > 0 and (step % interval == 0 or step == args.num_iters)


def get_grad_stats(module):
    total_sq = 0.0
    max_abs = 0.0
    nonfinite_cnt = 0
    grad_elem_cnt = 0
    for p in module.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        finite_mask = torch.isfinite(g)
        nonfinite_cnt += int((~finite_mask).sum().item())
        if finite_mask.any():
            g_finite = g[finite_mask]
            total_sq += float((g_finite * g_finite).sum().item())
            max_abs = max(max_abs, float(g_finite.abs().max().item()))
        grad_elem_cnt += g.numel()
    global_norm = math.sqrt(total_sq)
    return global_norm, max_abs, nonfinite_cnt, grad_elem_cnt


def get_grad_norm_from_grads(grads):
    total_sq = 0.0
    nonfinite_cnt = 0
    grad_elem_cnt = 0
    for g in grads:
        if g is None:
            continue
        g = g.detach()
        finite_mask = torch.isfinite(g)
        nonfinite_cnt += int((~finite_mask).sum().item())
        if finite_mask.any():
            g_finite = g[finite_mask]
            total_sq += float((g_finite * g_finite).sum().item())
        grad_elem_cnt += g.numel()
    return math.sqrt(total_sq), nonfinite_cnt, grad_elem_cnt


def summarize_temporal_weight_sensitivity(weight_sensitivity, eps=1e-30):
    """Summarize a ``[time, ...]`` sparse-loss sensitivity tensor.

    The time axis is split into equal early/middle/late chunks.  Both ratio
    directions are returned because ``late/early`` is the intuitive temporal
    decay factor, while ``early/late`` and its log10 value match the historical
    SparseInfluence diagnostics.
    """
    if not isinstance(weight_sensitivity, torch.Tensor):
        raise TypeError("weight_sensitivity must be a torch.Tensor")
    if weight_sensitivity.ndim < 1 or weight_sensitivity.shape[0] <= 0:
        raise ValueError("weight_sensitivity must have a non-empty time axis")

    raw = weight_sensitivity.detach().float()
    finite_fraction = float(torch.isfinite(raw).float().mean().item())
    sensitivity_abs = torch.nan_to_num(
        raw,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).abs()
    temporal_chunks = torch.tensor_split(sensitivity_abs, 3, dim=0)
    temporal_means = [
        float(chunk.mean().item()) if chunk.numel() > 0 else 0.0
        for chunk in temporal_chunks
    ]
    early_mean, middle_mean, late_mean = temporal_means
    ratio_valid = (
        finite_fraction == 1.0
        and early_mean > eps
        and late_mean > eps
    )

    return {
        "norm": float(sensitivity_abs.norm().item()),
        "abs_mean": float(sensitivity_abs.mean().item()),
        "nonzero_fraction": float(
            (sensitivity_abs > 1e-12).float().mean().item()
        ),
        "finite_fraction": finite_fraction,
        "early_abs_mean": early_mean,
        "middle_abs_mean": middle_mean,
        "late_abs_mean": late_mean,
        "early_to_late_ratio": early_mean / max(late_mean, eps),
        "late_to_early_decay_factor": late_mean / max(early_mean, eps),
        "early_to_late_log10_ratio": (
            math.log10(max(early_mean, eps))
            - math.log10(max(late_mean, eps))
        ),
        "ratio_valid": 1.0 if ratio_valid else 0.0,
    }


def merge_task_priority_gradients(
    proxy_grads,
    task_grads,
    max_proxy_to_task_ratio=0.5,
    eps=1e-12,
):
    """Merge Worker gradients while keeping the direct task objective primary.

    The proxy gradient is norm-limited relative to the arrival-task gradient.
    When they conflict, the proxy component opposing the task gradient is
    removed. Returned gradients are detached because this is only used for the
    persistent Worker update; the differentiable inner loop remains unchanged.
    """
    proxy_grads = tuple(proxy_grads)
    task_grads = tuple(task_grads)
    if len(proxy_grads) != len(task_grads):
        raise ValueError("proxy_grads and task_grads must have the same length")
    ratio = float(max_proxy_to_task_ratio)
    if ratio < 0.0 or not math.isfinite(ratio):
        raise ValueError("max_proxy_to_task_ratio must be finite and non-negative")

    proxy_sq = 0.0
    task_sq = 0.0
    dot = 0.0
    for proxy_grad, task_grad in zip(proxy_grads, task_grads):
        if proxy_grad is not None:
            proxy_d = proxy_grad.detach().double()
            proxy_sq += float((proxy_d * proxy_d).sum().item())
        if task_grad is not None:
            task_d = task_grad.detach().double()
            task_sq += float((task_d * task_d).sum().item())
        if proxy_grad is not None and task_grad is not None:
            dot += float(torch.sum(
                proxy_grad.detach().double() * task_grad.detach().double()
            ).item())

    proxy_norm = math.sqrt(max(0.0, proxy_sq))
    task_norm = math.sqrt(max(0.0, task_sq))
    paired_usable = proxy_norm > eps and task_norm > eps
    cosine = dot / (proxy_norm * task_norm) if paired_usable else 0.0
    cosine = max(-1.0, min(1.0, cosine)) if math.isfinite(cosine) else 0.0

    if proxy_norm <= eps:
        proxy_scale = 0.0
    elif task_norm <= eps:
        # At an exact task stationary point retain the proxy instead of
        # permanently freezing the Worker.
        proxy_scale = 1.0
    else:
        proxy_scale = min(1.0, ratio * task_norm / (proxy_norm + eps))

    conflict_projected = paired_usable and dot < 0.0
    projection_coeff = dot / (task_sq + eps) if conflict_projected else 0.0
    merged = []
    for proxy_grad, task_grad in zip(proxy_grads, task_grads):
        proxy_adjusted = None
        if proxy_grad is not None:
            proxy_adjusted = proxy_grad.detach()
            if conflict_projected and task_grad is not None:
                proxy_adjusted = proxy_adjusted - projection_coeff * task_grad.detach()
            proxy_adjusted = proxy_adjusted * proxy_scale

        if task_grad is None:
            merged_grad = proxy_adjusted
        elif proxy_adjusted is None:
            merged_grad = task_grad.detach()
        else:
            merged_grad = task_grad.detach() + proxy_adjusted
        merged.append(merged_grad)

    return tuple(merged), {
        'proxy_norm': proxy_norm,
        'task_norm': task_norm,
        'proxy_scale': proxy_scale,
        'cosine': cosine,
        'conflict_projected': 1.0 if conflict_projected else 0.0,
    }


def compute_meta_hypergrads_from_fast_grads(
    fast_params,
    meta_grads,
    meta_params,
    retain_graph=False,
):
    """Reuse ``d(meta_loss)/d(fast_params)`` to form meta-parameter hypergradients.

    When the outer/meta objective depends on ``meta_params`` only through the
    differentiable inner update ``fast_params(meta_params)``, the chain rule is

        dM/dphi = (d fast_params / dphi)^T @ (dM/d fast_params).

    ``meta_grads`` is treated as a detached cotangent.  This avoids traversing
    the validation rollout again while preserving the exact first derivative
    of the composed outer objective.
    """
    fast_params = tuple(fast_params)
    meta_grads = tuple(meta_grads)
    meta_params = tuple(meta_params)

    if len(fast_params) != len(meta_grads):
        raise ValueError(
            f"fast_params/meta_grads length mismatch: {len(fast_params)} != {len(meta_grads)}"
        )
    if len(meta_params) == 0:
        return tuple()

    outputs = []
    grad_outputs = []
    for fast_param, meta_grad in zip(fast_params, meta_grads):
        if meta_grad is None or not getattr(fast_param, "requires_grad", False):
            continue
        outputs.append(fast_param)
        grad_outputs.append(
            meta_grad.detach().to(device=fast_param.device, dtype=fast_param.dtype)
        )

    if len(outputs) == 0:
        return tuple(None for _ in meta_params)

    return torch.autograd.grad(
        tuple(outputs),
        meta_params,
        grad_outputs=tuple(grad_outputs),
        allow_unused=True,
        retain_graph=retain_graph,
        create_graph=False,
    )


def get_gradient_alignment_stats(left_grads, right_grads, eps=1e-12):
    """Return detached cosine-alignment diagnostics for paired gradient tuples."""
    dot = 0.0
    left_sq = 0.0
    right_sq = 0.0
    paired_elements = 0

    for left, right in zip(left_grads, right_grads):
        if left is None or right is None or left.shape != right.shape:
            continue
        left_d = left.detach().reshape(-1)
        right_d = right.detach().reshape(-1)
        finite = torch.isfinite(left_d) & torch.isfinite(right_d)
        if not bool(finite.any().item()):
            continue
        left_f = left_d[finite].double()
        right_f = right_d[finite].double()
        dot += float(torch.dot(left_f, right_f).item())
        left_sq += float(torch.dot(left_f, left_f).item())
        right_sq += float(torch.dot(right_f, right_f).item())
        paired_elements += int(finite.sum().item())

    left_norm = math.sqrt(left_sq)
    right_norm = math.sqrt(right_sq)
    usable = (
        paired_elements > 0
        and left_norm > eps
        and right_norm > eps
        and math.isfinite(dot)
        and math.isfinite(left_norm)
        and math.isfinite(right_norm)
    )
    cosine = dot / (left_norm * right_norm) if usable else 0.0
    cosine = max(-1.0, min(1.0, cosine)) if math.isfinite(cosine) else 0.0
    return {
        'cosine': cosine,
        'dot': dot,
        'left_norm': left_norm,
        'right_norm': right_norm,
        'paired_elements': float(paired_elements),
        'usable': 1.0 if usable else 0.0,
    }


def compute_gradient_alignment_loss(left_grads, right_grads, reference, eps=1e-12):
    """Differentiable ``1 - cosine`` alignment against a detached target gradient.

    ``left_grads`` is expected to be created with ``create_graph=True`` so the
    returned loss can update the module that generated those gradients.  The
    right/meta gradients are deliberately detached: they define the desired
    Worker-update direction without introducing third-order meta derivatives.
    """
    dot = reference.new_zeros(())
    left_sq = reference.new_zeros(())
    right_sq = reference.new_zeros(())
    paired_tensors = 0

    for left, right in zip(left_grads, right_grads):
        if left is None or right is None or left.shape != right.shape:
            continue
        left_f = torch.nan_to_num(left, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
        right_f = torch.nan_to_num(
            right.detach(), nan=0.0, posinf=0.0, neginf=0.0
        ).reshape(-1).to(device=left_f.device, dtype=left_f.dtype)
        dot = dot + torch.dot(left_f, right_f)
        left_sq = left_sq + torch.dot(left_f, left_f)
        right_sq = right_sq + torch.dot(right_f, right_f)
        paired_tensors += 1

    usable = (
        paired_tensors > 0
        and bool(torch.isfinite(left_sq.detach()).item())
        and bool(torch.isfinite(right_sq.detach()).item())
        and float(left_sq.detach().item()) > float(eps)
        and float(right_sq.detach().item()) > float(eps)
    )
    if not usable:
        zero = reference * 0.0
        return zero, zero, False

    denom = torch.sqrt(left_sq.clamp_min(eps)) * torch.sqrt(right_sq.clamp_min(eps))
    cosine = (dot / denom.clamp_min(eps)).clamp(-1.0, 1.0)
    return 1.0 - cosine, cosine, True


def _diag_should_log(iter_idx, args):
    return args.diag_interval > 0 and (iter_idx % args.diag_interval == 0)


def _diag_grad_meta(x):
    if x is None:
        return "None"
    gfn = type(x.grad_fn).__name__ if getattr(x, 'grad_fn', None) is not None else "None"
    return f"requires_grad={x.requires_grad}, is_leaf={x.is_leaf}, grad_fn={gfn}"


def _diag_tensor_finite(tag, x, iter_idx):
    if x is None:
        print(f"[DIAG iter={iter_idx}] {tag}: None")
        return
    with torch.no_grad():
        xd = x.detach()
        finite_mask = torch.isfinite(xd)
        finite_cnt = int(finite_mask.sum().item())
        total_cnt = int(xd.numel())
        nonfinite_cnt = total_cnt - finite_cnt
        if finite_cnt > 0:
            vals = xd[finite_mask]
            vmin = float(vals.min().item())
            vmax = float(vals.max().item())
        else:
            vmin = float('nan')
            vmax = float('nan')
    print(
        f"[DIAG iter={iter_idx}] {tag}: finite={finite_cnt}/{total_cnt}, "
        f"nonfinite={nonfinite_cnt}/{total_cnt}, min={vmin:.6g}, max={vmax:.6g}"
    )


def _diag_grad_tuple_to_params(tag, grad_tuple, params, iter_idx, retain_graph=True):
    params = list(params)
    total_params = len(params)
    if total_params == 0:
        print(f"[DIAG iter={iter_idx}] {tag}: None=0/0, NonZero=0/0, Norm=0.000000, NonFinite=0/0")
        return

    if grad_tuple is None:
        print(f"[DIAG iter={iter_idx}] {tag}: None={total_params}/{total_params}, NonZero=0/{total_params}, Norm=0.000000, NonFinite=0/0")
        return

    grads = [g for g in grad_tuple if g is not None]
    if len(grads) == 0:
        print(f"[DIAG iter={iter_idx}] {tag}: None={total_params}/{total_params}, NonZero=0/{total_params}, Norm=0.000000, NonFinite=0/0")
        return

    probe = None
    for g in grads:
        s = g.sum()
        probe = s if probe is None else (probe + s)

    try:
        mapped = torch.autograd.grad(
            probe,
            params,
            allow_unused=True,
            retain_graph=retain_graph,
            create_graph=False,
        )
    except Exception as e:
        print(f"[DIAG iter={iter_idx}] {tag}: grad-check failed: {e}")
        return

    none_cnt = sum(g is None for g in mapped)
    nonzero_cnt = 0
    total_sq = 0.0
    nonfinite = 0
    total_elems = 0
    for g in mapped:
        if g is None:
            continue
        gd = g.detach()
        finite_mask = torch.isfinite(gd)
        nonfinite += int((~finite_mask).sum().item())
        total_elems += gd.numel()
        if finite_mask.any():
            vals = gd[finite_mask]
            total_sq += float((vals * vals).sum().item())
            if float(vals.abs().sum().item()) > 1e-12:
                nonzero_cnt += 1

    print(
        f"[DIAG iter={iter_idx}] {tag}: None={none_cnt}/{total_params}, "
        f"NonZero={nonzero_cnt}/{total_params}, Norm={math.sqrt(total_sq):.6f}, "
        f"NonFinite={nonfinite}/{total_elems}"
    )


def _diag_output_to_params(tag, output, params, iter_idx, retain_graph=True):
    params = list(params)
    total_params = len(params)
    if total_params == 0:
        print(
            f"[DIAG iter={iter_idx}] {tag}: None=0/0, NonZero=0/0, "
            "Norm=0.000000, MaxAbs=0.000000, NonFinite=0/0"
        )
        return
    if not getattr(output, "requires_grad", False):
        print(
            f"[DIAG iter={iter_idx}] {tag}: requires_grad=False, "
            f"None={total_params}/{total_params}, NonZero=0/{total_params}, "
            "Norm=0.000000, MaxAbs=0.000000, NonFinite=0/0"
        )
        return
    try:
        grads = torch.autograd.grad(
            output,
            params,
            allow_unused=True,
            retain_graph=retain_graph,
            create_graph=False,
        )
    except Exception as e:
        print(f"[DIAG iter={iter_idx}] {tag}: grad-check failed: {e}")
        return

    none_cnt = sum(g is None for g in grads)
    nonzero_cnt = 0
    total_sq = 0.0
    max_abs = 0.0
    nonfinite = 0
    grad_elems = 0
    for grad in grads:
        if grad is None:
            continue
        grad_detached = grad.detach()
        finite_mask = torch.isfinite(grad_detached)
        nonfinite += int((~finite_mask).sum().item())
        grad_elems += grad_detached.numel()
        if finite_mask.any():
            finite_values = grad_detached[finite_mask]
            total_sq += float((finite_values * finite_values).sum().item())
            current_max = float(finite_values.abs().max().item())
            max_abs = max(max_abs, current_max)
            if float(finite_values.abs().sum().item()) > 1e-12:
                nonzero_cnt += 1

    print(
        f"[DIAG iter={iter_idx}] {tag}: None={none_cnt}/{total_params}, "
        f"NonZero={nonzero_cnt}/{total_params}, Norm={math.sqrt(total_sq):.6f}, "
        f"MaxAbs={max_abs:.6f}, NonFinite={nonfinite}/{grad_elems}"
    )


def _diag_output_to_params_count(tag, output, params, iter_idx, retain_graph=True):
    params = list(params)
    total_params = len(params)
    if total_params == 0:
        print(f"[DIAG iter={iter_idx}] {tag}: None=0/0, NonZero=0/0")
        return
    try:
        grads = torch.autograd.grad(
            output,
            params,
            allow_unused=True,
            retain_graph=retain_graph,
            create_graph=False,
        )
    except Exception as e:
        print(f"[DIAG iter={iter_idx}] {tag}: grad-check failed: {e}")
        return

    none_cnt = sum(g is None for g in grads)
    nonzero_cnt = 0
    for g in grads:
        if g is None:
            continue
        if float(g.detach().abs().sum().item()) > 1e-12:
            nonzero_cnt += 1
    print(f"[DIAG iter={iter_idx}] {tag}: None={none_cnt}/{total_params}, NonZero={nonzero_cnt}/{total_params}")


def _grad_or_none_tuple(loss, params, create_graph=True, retain_graph=True):
    params = tuple(params)
    if not getattr(loss, "requires_grad", False):
        return tuple(None for _ in params)
    return torch.autograd.grad(
        loss,
        params,
        create_graph=create_graph,
        allow_unused=True,
        retain_graph=retain_graph,
    )
