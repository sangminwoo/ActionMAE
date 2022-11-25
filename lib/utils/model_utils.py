import torch


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    output: (#items, #classes)
    target: int,
    """
    maxk = max(topk)
    num_items = output.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target)

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum()
        res.append(correct_k.mul_(100.0 / num_items))
    return res  # ([acc@1, acc@5])


@torch.no_grad()
def per_class_accuracy(output, target, topk=(1,), num_classes=60):
    """Computes the precision@k for the specified values of k
    output: (#items, #classes)
    target: int,
    """
    maxk = max(topk)
    list_of_classes = list(range(num_classes))

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    res = []
    for c in list_of_classes:
        acc = []
        num_items = max((target == c).sum(), 1)
        for k in topk:
            correct_k = ((pred == target) * (target == c))[:k].float().sum()
            acc.append(correct_k.mul_(100.0 / num_items))
        acc.append(num_items)
        res.append(acc)
    return res  # ([[[acc@1, acc@5], num_items] x #classes])


def count_parameters(model, verbose=True):
    """Count number of parameters in PyTorch model,
    References: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/7.

    from utils.utils import count_parameters
    count_parameters(model)
    import sys
    sys.exit(1)
    """
    # import pdb; pdb.set_trace()
    n_all = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    mem_params = sum([param.nelement()*param.element_size() for param in model.parameters()])
    mem_bufs = sum([buf.nelement()*buf.element_size() for buf in model.buffers()])
    mem = mem_params + mem_bufs # in bytes

    if verbose:
        print("Parameter Count: all {:,d}; trainable {:,d}".format(n_all, n_trainable))
        print("Memory: all {:,d}; params {:,d}; buffers {:,d}".format(mem, mem_params, mem_bufs))
    return n_all, n_trainable, mem, mem_params, mem_bufs


