from typing import List, Union

import torch
from torch import nn
from torch.nn import functional as F
from pldm.models.utils import *


def build_projector(arch: str, embedding: int):
    if arch == "id":
        return nn.Identity(), embedding
    else:
        f = [embedding] + list(map(int, arch.split("-")))
        return build_mlp(f), f[-1]


def build_norm1d(norm: str, dim: int):
    if norm == "batch_norm":
        return torch.nn.BatchNorm1d(dim)
    elif norm == "layer_norm":
        return torch.nn.LayerNorm(dim)
    else:
        raise ValueError(f"Unknown norm {norm}")


def build_activation(activation: str):
    if activation == "relu":
        return nn.ReLU(True)
    elif activation == "mish":
        return nn.Mish(True)
    else:
        raise ValueError(f"Unknown activation {activation}")


class PartialAffineLayerNorm(nn.Module):
    def __init__(
        self,
        first_dim: int,
        second_dim: int,
        first_affine: bool = True,
        second_affine: bool = True,
    ):
        super().__init__()
        self.first_dim = first_dim
        self.second_dim = second_dim

        if first_affine:
            self.first_ln = nn.LayerNorm(first_dim, elementwise_affine=True)
        else:
            self.first_ln = nn.LayerNorm(first_dim, elementwise_affine=False)

        if second_affine:
            self.second_ln = nn.LayerNorm(second_dim, elementwise_affine=True)
        else:
            self.second_ln = nn.LayerNorm(second_dim, elementwise_affine=False)

    def forward(self, x):
        first = self.first_ln(x[..., : self.first_dim])
        second = self.second_ln(x[..., self.first_dim :])
        out = torch.cat([first, second], dim=-1)
        return out


def build_mlp(
    layers_dims: Union[List[int], str],
    input_dim: int = None,
    output_shape: int = None,
    norm="batch_norm",
    activation="relu",
    pre_actnorm=False,
    post_norm=False,
):
    if isinstance(layers_dims, str):
        layers_dims = (
            list(map(int, layers_dims.split("-"))) if layers_dims != "" else []
        )

    if input_dim is not None:
        layers_dims = [input_dim] + layers_dims

    if output_shape is not None:
        layers_dims = layers_dims + [output_shape]

    layers = []

    if pre_actnorm:
        if norm is not None:
            layers.append(build_norm1d(norm, layers_dims[0]))
        if activation is not None:
            layers.append(build_activation(activation))

    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        if norm is not None:
            layers.append(build_norm1d(norm, layers_dims[i + 1]))
        if activation is not None:
            layers.append(build_activation(activation))

    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))

    if post_norm:
        layers.append(build_norm1d(norm, layers_dims[-1]))

    return nn.Sequential(*layers)


class MLP(torch.nn.Module):
    def __init__(
        self,
        arch: str,
        input_dim: int = None,
        output_shape: int = None,
        norm=None,
        activation="relu",
    ):
        super().__init__()

        self.mlp = build_mlp(
            layers_dims=arch,
            input_dim=input_dim,
            output_shape=output_shape,
            norm=norm,
            activation=activation,
        )

    def forward(self, x):
        return self.mlp(x)


PROBER_CONV_LAYERS_CONFIG = {
    "a": [
        (-1, 16, 3, 1, 1),
        ("max_pool", 2, 2, 0),
        (16, 8, 1, 1, 0),
        ("max_pool", 2, 2, 0),
        ("fc", -1, 2),
    ],
    "b": [
        (-1, 32, 3, 1, 1),
        ("max_pool", 2, 2, 0),
        (32, 32, 3, 1, 1),
        ("max_pool", 2, 2, 0),
        (32, 32, 3, 1, 1),
        ("fc", -1, 2),
    ],
    "c": [
        (-1, 32, 3, 1, 1),
        ("max_pool", 2, 2, 0),
        (32, 32, 3, 1, 1),
        ("max_pool", 2, 2, 0),
        (32, 32, 3, 1, 1),
        (32, 32, 3, 1, 1),
        ("fc", -1, 2),
    ],
}


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: int,
        input_dim=None,
        arch_subclass: str = "a",
    ):
        super().__init__()
        self.output_shape = output_shape
        self.arch = arch

        if arch == "conv":
            self.prober = build_conv(
                PROBER_CONV_LAYERS_CONFIG[arch_subclass], input_dim=input_dim
            )
        else:
            arch_list = list(map(int, arch.split("-"))) if arch != "" else []
            f = [embedding] + arch_list + [self.output_shape]
            layers = []
            for i in range(len(f) - 2):
                layers.append(torch.nn.Linear(f[i], f[i + 1]))
                # layers.append(torch.nn.BatchNorm1d(f[i + 1]))
                layers.append(torch.nn.ReLU(True))
            layers.append(torch.nn.Linear(f[-2], f[-1]))
            self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        if self.arch == "conv":
            output = self.prober(e)
        else:
            e = flatten_conv_output(e)
            output = self.prober(e)

        # output = output.view(*output.shape[:-1], *self.output_shape)

        return output


class Projector(torch.nn.Module):
    def __init__(self, arch: str, embedding: int, random: bool = False):
        super().__init__()

        self.arch = arch
        self.embedding = embedding
        self.random = random

        self.model, self.output_dim = build_projector(arch, embedding)

        if self.random:
            for param in self.parameters():
                param.requires_grad = False

    def maybe_reinit(self):
        if self.random and self.arch != "id":
            for param in self.parameters():
                torch.nn.init.xavier_uniform_(param)
                print("initialized")

    def forward(self, x: torch.Tensor):
        return self.model(x)
