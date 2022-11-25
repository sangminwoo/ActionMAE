import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_coefficient(label, outputs, alpha=1.):
    # Calculate discrepancy ratio and k
    softmax = nn.Softmax(dim=1)
    relu = nn.ReLU(inplace=True)
    tanh = nn.Tanh()

    scores = {
        mode: sum([softmax(output)[i][label[i]] for i in range(output.size(0))])
        for mode, output in outputs.items()
    }
    score_mean = sum(scores.values()) / len(scores)

    ratios = {
        mode: score / score_mean
        for mode, score in scores.items()
    }

    coeffs = {
        mode: 1 - tanh(alpha * (relu(ratio) - 1))
        for mode, ratio in ratios.items()
    }

    return coeffs


def update_model_with_OGM_GE(model, coeffs, generalization_enhancement=False):
    # Gradient Modulation begins before optimization, and with GE applied.
    for name, params in model.named_parameters():
        # sanity check
        if params.requires_grad and not params.grad == None and not torch.any(params.grad.isnan()):
            if 'rgb' in name:
                if generalization_enhancement:
                    params.grad = params.grad * coeffs['rgb'] + \
                                torch.zeros_like(params.grad).normal_(0, params.grad.std().item() + 1e-8)
                else:
                    params.grad = params.grad * coeffs['rgb']

            if 'depth' in name:
                if generalization_enhancement:
                    params.grad = params.grad * coeffs['depth'] + \
                                torch.zeros_like(params.grad).normal_(0, params.grad.std().item() + 1e-8)
                else:
                   params.grad = params.grad * coeffs['depth']

            if 'ir' in name:
                if generalization_enhancement:
                    params.grad = params.grad * coeffs['ir'] + \
                                torch.zeros_like(params.grad).normal_(0, params.grad.std().item() + 1e-8)
                else:
                   params.grad = params.grad * coeffs['ir']

            if 'skeleton' in name:
                if generalization_enhancement:
                    params.grad = params.grad * coeffs['skeleton'] + \
                                torch.zeros_like(params.grad).normal_(0, params.grad.std().item() + 1e-8)
                else:
                   params.grad = params.grad * coeffs['skeleton']