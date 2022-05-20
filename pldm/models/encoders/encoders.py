import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from pldm.models.encoders.resnet import resnet18, resnet18ID
from pldm.models.encoders import resnet
from pldm.models.misc import (
    build_mlp,
    Projector,
    MLP,
    build_norm1d,
    PartialAffineLayerNorm,
)
from pldm.models.utils import build_conv, Expander2D
from pldm.models.encoders.enums import BackboneConfig, BackboneOutput
from pldm.models.encoders.base_class import SequenceBackbone
from pldm.models.encoders.impala import ImpalaEncoder

ResNet18 = resnet18
ResNet18ID = resnet18ID

ENCODER_LAYERS_CONFIG = {
    # L1
    "a": [(2, 32, 5, 1, 0), (32, 32, 4, 2, 0), (32, 32, 3, 1, 0), (32, 16, 1, 1, 0)],
    "b": [(2, 16, 5, 1, 0), (16, 32, 4, 2, 0), (32, 32, 3, 1, 0), (32, 16, 1, 1, 0)],
    "c": [(2, 16, 5, 1, 0), (16, 16, 4, 2, 0), (16, 16, 3, 1, 0)],
    "f": [(2, 16, 5, 1, 0), (16, 16, 5, 2, 0), (16, 16, 5, 1, 2)],
    "g": [(2, 32, 5, 1, 0), (32, 32, 5, 2, 0), (32, 32, 5, 1, 2), (32, 16, 1, 1, 0)],
    "h": [(2, 16, 5, 1, 0), (16, 16, 5, 2, 0), (16, 16, 3, 1, 0)],
    "i": [(2, 16, 5, 1, 0), (16, 16, 5, 2, 0), (16, 16, 3, 1, 1)],
    "i_fc": [
        (2, 16, 5, 1, 0),
        (16, 16, 5, 2, 0),
        (16, 16, 3, 1, 1),
        ("fc", 13456, 512),
    ],
    "i_b": [(6, 16, 5, 1, 0), (16, 16, 5, 2, 0), (16, 16, 3, 1, 0)],
    "d4rl_a": [
        (6, 16, 5, 1, 0),
        (16, 32, 5, 2, 0),
        (32, 32, 3, 1, 0),
        (32, 32, 3, 1, 1),
        (32, 16, 1, 1, 0),
    ],
    "d4rl_b": [
        (6, 16, 5, 1, 0),
        (16, 32, 5, 2, 0),
        (32, 32, 3, 1, 0),
        (32, 32, 3, 1, 1),
        (32, 32, 3, 1, 1),
        (32, 16, 1, 1, 0),
    ],
    "d4rl_c": [
        (6, 16, 5, 1, 0),
        (16, 32, 5, 2, 0),
        (32, 32, 3, 1, 0),
        (32, 32, 3, 1, 1),
        (32, 32, 3, 1, 1),
    ],
    "j": [(2, 32, 5, 1, 0), (32, 32, 5, 2, 0), (32, 32, 3, 1, 1), (32, 16, 1, 1, 0)],
    "k": [(2, 16, 5, 1, 0), (16, 32, 5, 2, 0), (32, 32, 3, 1, 1), (32, 16, 1, 1, 0)],
    # L2
    "d": [(16, 16, 3, 1, 0), (16, 16, 3, 1, 0)],
    "e": [
        ("pad", (0, 1, 0, 1)),
        (16, 16, 3, 1, 0),
        ("avg_pool", 2, 2, 0),
        (16, 16, 3, 1, 0),
    ],
    "l2a": [(16, 16, 5, 1, 2), (16, 16, 5, 2, 2), (16, 16, 3, 1, 1)],  # (8, 16, 15, 15)
    "l2b": [(16, 16, 3, 1, 1), (16, 16, 3, 2, 1), (16, 16, 3, 1, 1)],  # (8, 16, 15, 15)
    "l2c": [(16, 32, 5, 1, 2), (32, 32, 5, 2, 2), (32, 32, 3, 1, 1)],  # (8, 32, 15, 15)
    "l2d": [(16, 32, 3, 1, 1), (32, 32, 3, 2, 1), (32, 32, 3, 1, 1)],  # (8, 32, 15, 15)
    "l2e": [(16, 16, 3, 2, 1), (16, 16, 3, 1, 1)],
}


class PassThrough(nn.Module):
    def forward(self, x):
        return x


class MLPNet(nn.Module):
    def __init__(self, output_dim: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, output_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        out = x.flatten(1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = BackboneOutput(encodings=out)
        return out


class MeNet5(SequenceBackbone):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 64,
        input_channels: int = 1,
        width_factor: int = 1,
        conv_out_dim: int = 9 * 32,
        backbone_norm: str = "batch_norm",
        backbone_pool: str = "backbone_pool",
        backbone_final_fc: bool = True,
    ):
        super().__init__()
        self.width_factor = width_factor
        self.conv_out_dim = conv_out_dim
        self.backbone_final_fc = backbone_final_fc
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                input_channels, 16 * width_factor, kernel_size=5, stride=2, padding=2
            ),
            nn.ReLU(),
            (
                nn.BatchNorm2d(16 * width_factor)
                if backbone_norm == "batch_norm"
                else nn.GroupNorm(4, 16 * width_factor)
            ),
            nn.Conv2d(
                16 * width_factor, 32 * width_factor, kernel_size=5, stride=2, padding=2
            ),
            nn.ReLU(),
            (
                nn.BatchNorm2d(32 * width_factor)
                if backbone_norm == "batch_norm"
                else nn.GroupNorm(4, 32 * width_factor)
            ),
            nn.Conv2d(
                32 * width_factor, 32 * width_factor, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            (
                nn.BatchNorm2d(32 * width_factor)
                if backbone_norm == "batch_norm"
                else nn.GroupNorm(4, 32 * width_factor)
            ),
        )
        if backbone_pool == "avg_pool":
            self.pool = nn.AvgPool2d(2, stride=2)
        else:
            self.pool = nn.Conv2d(
                in_channels=32 * width_factor, out_channels=32, kernel_size=1
            )
        sample_input = torch.randn(input_dim).unsqueeze(0)
        sample_output = self.pool(self.layer1(sample_input)).reshape(1, -1)
        if backbone_final_fc:
            self.fc = nn.Linear(sample_output.shape[-1], output_dim)

    def forward(self, x):
        out = self.layer1(x)  # [bs,64,16,16]
        out = self.pool(out)  # [bs, 32, 16, 16]
        if self.backbone_final_fc:
            out = out.reshape(out.size(0), -1)
            out = self.fc(out)
        out = BackboneOutput(encodings=out)
        return out


class MeNet6(SequenceBackbone):
    def __init__(
        self,
        config,
        input_dim: int,
    ):
        super().__init__()

        self.config = config
        subclass = config.backbone_subclass
        layers_config = ENCODER_LAYERS_CONFIG[subclass]

        if "l2" in subclass:
            # add prenormalization and relu layers?
            pre_conv = nn.Sequential(nn.GroupNorm(4, layers_config[0][0]), nn.ReLU())
        else:
            pre_conv = nn.Identity()
        conv_layers = build_conv(layers_config, (input_dim[0],))

        self.layers = nn.Sequential(pre_conv, conv_layers)

        if config.propio_dim:
            # infer output dim of encoder
            sample_input = torch.randn(input_dim).unsqueeze(0)
            sample_output = self.layers(sample_input)
            encoder_output_dim = tuple(sample_output.shape[1:])

            if (
                self.config.propio_encoder_arch
                and self.config.propio_encoder_arch != "id"
            ):
                layer_dims = [
                    int(x) for x in self.config.propio_encoder_arch.split("-")
                ]
                layers = []
                for i in range(len(layer_dims) - 1):
                    layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
                    layers.append(nn.ReLU())
                # remove last ReLU
                layers.pop()

                self.propio_encoder = nn.Sequential(
                    *layers,
                    Expander2D(w=encoder_output_dim[-2], h=encoder_output_dim[-1]),
                )
            else:
                self.propio_encoder = Expander2D(
                    w=encoder_output_dim[-2], h=encoder_output_dim[-1]
                )

    def forward(self, x, propio=None):
        """
        torch.Size([bs, 2, 64, 64])
        torch.Size([bs, 32, 60, 60])
        torch.Size([bs, 32, 29, 29])
        torch.Size([bs, 32, 27, 27])
        torch.Size([bs, 16, 27, 27])
        """
        obs = self.layers(x)

        if self.config.propio_dim and propio is not None:
            propio_states = self.propio_encoder(propio)
            encodings = torch.cat([obs, propio_states], dim=1)
        else:
            propio_states = None
            encodings = obs

        output = BackboneOutput(
            encodings=encodings,
            obs_component=obs,
            propio_component=propio_states,
        )

        return output


class ResizeConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        scale_factor,
        mode="nearest",
        groups=1,
        bias=False,
        padding=1,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        x = BackboneOutput(encodings=x)
        return x


class Canonical(nn.Module):
    def __init__(self, output_dim: int = 64):
        super().__init__()
        res = int(np.sqrt(output_dim / 64))
        assert (
            res * res * 64 == output_dim
        ), "canonical backbone resolution error: cant fit desired output_dim"

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((res, res)),
        )

    def forward(self, x):
        x = self.backbone(x).flatten(1)
        x = BackboneOutput(encodings=x)
        return x


class MLPEncoder(SequenceBackbone):
    def __init__(self, config, input_dim):
        super().__init__()
        self.encoder = MLP(
            arch=config.backbone_subclass,
            input_dim=input_dim,
            norm=config.backbone_norm,
        )

    def forward(self, x):
        x = self.encoder(x)
        x = BackboneOutput(encodings=x)
        return x


class ObPropioEncoder1(SequenceBackbone):
    """
    Fused encoder for observation and propio state.
    cat(obs, propio) --> encoder --> encodings
    """

    def __init__(self, config, obs_dim):
        super().__init__()
        self.config = config

        self.encoder = MLP(
            arch=config.backbone_subclass,
            input_dim=obs_dim + config.propio_dim,
            norm=config.backbone_norm,
            activation="mish",
        )

        out_dim = int(config.backbone_subclass.split("-")[-1])

        if config.final_ln:
            self.final_ln = build_norm1d(config.backbone_norm, out_dim)
        else:
            self.final_ln = nn.Identity()

    def forward(self, obs, propio):
        x = torch.cat([obs, propio], dim=-1)
        x = self.encoder(x)
        x = self.final_ln(x)

        return BackboneOutput(encodings=x)


class ObPropioEncoder2(SequenceBackbone):
    """
    Distangled encoder for observation and propio state.
    obs --> obs_encoder --> obs_out
    propio --> propio_encoder --> propio_out
    encodings = cat(obs_out, propio_out)
    return: encodings, obs_out, propio_out
    """

    def __init__(self, config, obs_dim):
        super().__init__()
        self.config = config

        obs_subclass, propio_subclass = config.backbone_subclass.split(",")

        if obs_subclass == "id":
            self.obs_encoder = nn.Identity()
            obs_out_dim = obs_dim
        else:
            self.obs_encoder = build_mlp(
                layers_dims=obs_subclass,
                input_dim=obs_dim,
                norm=config.backbone_norm,
                activation="mish",
            )
            obs_out_dim = int(obs_subclass.split("-")[-1])

        if propio_subclass == "id":
            self.propio_encoder = nn.Identity()
            propio_out_dim = config.propio_dim
        else:
            self.propio_encoder = build_mlp(
                layers_dims=propio_subclass,
                input_dim=config.propio_dim,
                norm=config.backbone_norm,
                activation="mish",
            )
            propio_out_dim = int(propio_subclass.split("-")[-1])

        if config.final_ln:
            self.final_ln = PartialAffineLayerNorm(
                first_dim=obs_out_dim,
                second_dim=propio_out_dim,
                first_affine=(obs_subclass != "id"),
                second_affine=(propio_subclass != "id"),
            )
        else:
            self.final_ln = nn.Identity()

    def forward(self, obs, propio):
        obs_out = self.obs_encoder(obs)
        propio_out = self.propio_encoder(propio)

        next_state = torch.cat([obs_out, propio_out], dim=1)
        next_state = self.final_ln(next_state)

        return BackboneOutput(
            encodings=next_state,
            obs_component=obs_out,
            propio_component=propio_out,
        )


def build_backbone(
    config: BackboneConfig,
    input_dim,
):
    backbone, embedding = None, None
    arch = config.arch

    if arch == "resnet18" or "resnet18s" in arch:
        backbone, embedding = resnet.__dict__[arch](
            zero_init_residual=True,
            num_channels=config.channels,
            final_ln=config.final_ln,
        )
    elif arch == "resnet18ID":
        backbone, embedding = resnet.__dict__[arch](
            zero_init_residual=False, num_channels=config.channels
        )
    elif arch == "impala":
        backbone = ImpalaEncoder(final_ln=config.final_ln)
    elif arch == "id":
        backbone = PassThrough()
        assert config.input_dim is not None
    elif arch == "menet5":
        backbone = MeNet5(
            output_dim=config.fc_output_dim,
            width_factor=config.backbone_width_factor,
            input_channels=config.channels,
            input_dim=input_dim,
            backbone_norm=config.backbone_norm,
            backbone_pool=config.backbone_pool,
            backbone_final_fc=config.backbone_final_fc,
        )
    elif arch == "menet6":
        backbone = MeNet6(
            config=config,
            input_dim=input_dim,
        )
    elif arch == "mlp":
        backbone = MLPEncoder(
            config=config,
            input_dim=input_dim,
        )
    elif arch == "canonical":
        backbone = Canonical(output_dim=config.fc_output_dm)
    elif arch == "ob_propio_enc_1":
        backbone = ObPropioEncoder1(config=config, obs_dim=input_dim)
    elif arch == "ob_propio_enc_2":
        backbone = ObPropioEncoder2(config=config, obs_dim=input_dim)
    elif config.input_dim is not None:
        # Used for the second-level HJEPA.
        if arch == "identity":
            backbone = nn.Identity()
        else:
            # We assume it's mlp with input that's not image.
            mlp_params = list(map(int, arch.split("-"))) if arch != "" else []
            backbone = build_mlp(
                [config.input_dim] + mlp_params + [config.fc_output_dim]
            )
    else:
        raise NotImplementedError(f"backbone arch {arch} is unknown")

    if config.backbone_mlp is not None:
        backbone_mlp = Projector(config.backbone_mlp, embedding)
        backbone = nn.Sequential(backbone, backbone_mlp)

    backbone.input_dim = input_dim
    sample_input = torch.randn(input_dim).unsqueeze(0)

    if config.propio_dim is not None:
        sample_propio_input = torch.randn(config.propio_dim).unsqueeze(0)
        sample_output = backbone(sample_input, propio=sample_propio_input)
    else:
        sample_output = backbone(sample_input)

    output_dim = tuple(sample_output.encodings.shape[1:])
    output_dim = output_dim[0] if len(output_dim) == 1 else output_dim
    backbone.output_dim = output_dim

    if sample_output.propio_component is not None:
        output_obs_dim = tuple(sample_output.obs_component.shape[1:])
        output_obs_dim = (
            output_obs_dim[0] if len(output_obs_dim) == 1 else output_obs_dim
        )
        output_propio_dim = tuple(sample_output.propio_component.shape[1:])
        output_propio_dim = (
            output_propio_dim[0] if len(output_propio_dim) == 1 else output_propio_dim
        )
    else:
        output_obs_dim = output_dim
        output_propio_dim = 0

    backbone.output_obs_dim = output_obs_dim
    backbone.output_propio_dim = output_propio_dim

    backbone.config = config

    return backbone
