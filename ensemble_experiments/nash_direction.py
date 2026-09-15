"""Two-task Nash direction, with norm matched to ordinary mean gradients.

This is a Nash-inspired AdamW experiment, not the paper's full optimizer.
Only shared backbone gradients change; private gradients retain mean-loss scale.
"""
import torch


def combine(a, b):
    na, nb = a.norm(), b.norm()
    mean = (a+b)*.5
    cosine = torch.dot(a,b)/(na*nb).clamp_min(1e-20)
    direction = a/na.clamp_min(1e-20)+b/nb.clamp_min(1e-20)
    fallback = (na < 1e-12) | (nb < 1e-12) | (direction.norm() < 1e-6)
    balanced = direction*(mean.norm()/direction.norm().clamp_min(1e-20))
    result = torch.where(fallback, mean, balanced)
    metrics = dict(cosine=cosine, norm_ratio_ts_atm=na/nb.clamp_min(1e-20),
                   conflict=(cosine<0).float(), fallback=fallback.float(),
                   mean_dot_ts=torch.dot(a,mean), mean_dot_atm=torch.dot(b,mean),
                   nash_dot_ts=torch.dot(a,result), nash_dot_atm=torch.dot(b,result),
                   mean_hurts_either=((torch.dot(a,mean)<0)|(torch.dot(b,mean)<0)).float())
    return result, metrics


def backward(model, features, positives, rule='nash'):
    losses = [c.multi_positive_pair_loss(e.float(), i.float(), positives)
              for c,(e,i) in zip(model.criteria,features)]
    assert len(losses)==2
    named = [(name,p) for name,p in model.named_parameters() if p.requires_grad]
    params = [p for _,p in named]
    grads = [torch.autograd.grad(loss, params, retain_graph=j==0, allow_unused=True)
             for j,loss in enumerate(losses)]
    shared = [j for j,(name,_) in enumerate(named) if name.startswith('backbone.')]
    a,b = [torch.cat([g[j].flatten() for j in shared]) for g in grads]
    direction, metrics = combine(a,b)
    if rule == 'mean': direction = (a+b)*.5
    offset = 0
    for j,(_,p) in enumerate(named):
        if j in shared:
            p.grad = direction[offset:offset+p.numel()].view_as(p).clone()
            offset += p.numel()
        else:
            parts = [g[j] for g in grads if g[j] is not None]
            p.grad = sum(parts)*.5 if parts else None
    for group in ['tsconv', 'projection', 'proj_eeg']:
        indices = [j for j in shared if named[j][0].startswith(f'backbone.{group}.')]
        if indices:
            ga,gb = [torch.cat([g[j].flatten() for j in indices]) for g in grads]
            metrics[f'{group}_cosine'] = torch.dot(ga,gb)/(ga.norm()*gb.norm()).clamp_min(1e-20)
    return torch.stack(losses).mean().detach(), metrics, (a,b)


def actual_update(model, before, task_grads):
    # Positive dot means first-order loss decrease. Includes AdamW decay/momentum.
    after = torch.cat([p.detach().flatten() for p in model.backbone.parameters()])
    descent = before-after
    a,b = task_grads
    return dict(actual_dot_ts=torch.dot(a,descent), actual_dot_atm=torch.dot(b,descent),
                actual_hurts_either=((torch.dot(a,descent)<0)|(torch.dot(b,descent)<0)).float(),
                actual_step_norm=descent.norm())
