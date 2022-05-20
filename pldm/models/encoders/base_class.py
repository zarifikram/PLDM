from torch import nn
from pldm.models.encoders.enums import BackboneOutput


class SequenceBackbone(nn.Module):
    def __init__(self):
        """
        collapse T and BS dimensions prior to passing to backbone
        afterwards reshape to original shape
        """
        super().__init__()
        self.output_propio_dim = 0

    def _remove_propio_component_for_spatial(self, embeddings):
        """
        remove the propio component from spatial embeddings

        Input:
            embeddings: tensor
            (T, BS, Ch, W, H) or
            (BS, Ch, W, H) or
            (T, BS, H) or
            (BS, H)
        """
        og_shape = tuple(embeddings.shape)
        flattened_input = len(og_shape) < 4

        # first reshape to spatial dimension if needed
        if flattened_input:
            spatial_shape = (*embeddings.shape[:-1], *self.output_dim)
            embeddings = embeddings.view(spatial_shape)

        propio_channels = self.output_propio_dim[0]

        # remove the propio dimensions
        if len(embeddings.shape) == 5:
            embeddings = embeddings[:, :, :-propio_channels]
        elif len(embeddings.shape) == 4:
            embeddings = embeddings[:, :-propio_channels]

        # reflatten tensor if needed
        if flattened_input:
            embeddings = embeddings.view(*og_shape[:-1], -1)

        return embeddings

    def remove_propio_component(self, embeddings):
        """
        remove the propio component from embeddings
        Input:
            embeddings: tensor
            (T, BS, Ch, W, H) or
            (BS, Ch, W, H) or
            (T, BS, H) or
            (BS, H)
        """
        if not self.output_propio_dim:
            return embeddings

        if isinstance(self.output_dim, int):
            return embeddings[..., : -self.output_propio_dim]
        else:
            return self._remove_propio_component_for_spatial(embeddings)

    def forward_multiple(self, x, propio=None):
        """
        input:
            x: [T, BS, *] or [BS, *]
        output:
            x: [T, BS, *] or [BS, *]
        """

        # if no time dimension, just feed it directly to backbone
        if x.dim() == 2 or x.dim() == 4:
            if propio is not None:
                output = self.forward(x, propio)
            else:
                output = self.forward(x)
            return output

        state = x.flatten(0, 1)

        if propio is not None:
            propio = propio.flatten(0, 1)
            output = self.forward(state, propio)
        else:
            output = self.forward(state)

        state = output.encodings
        new_shape = x.shape[:2] + state.shape[1:]
        state = state.reshape(new_shape)

        if output.obs_component is not None:
            obs_component = output.obs_component
            new_shape = x.shape[:2] + obs_component.shape[1:]
            obs_component = obs_component.reshape(new_shape)
        else:
            obs_component = None

        if output.propio_component is not None:
            propio_component = output.propio_component
            new_shape = x.shape[:2] + propio_component.shape[1:]
            propio_component = propio_component.reshape(new_shape)
        else:
            propio_component = None

        output = BackboneOutput(
            encodings=state,
            obs_component=obs_component,
            propio_component=propio_component,
        )

        return output
