import torch

EPS = 1e-6


def tensors_about_equal(t1, t2, eps=EPS):
    result = torch.norm(t1 - t2) < EPS * torch.norm(t1 + t2)
    if not result:
        print("first tensor:\n", t1)
        print("second tensor:\n", t2)
    return result
