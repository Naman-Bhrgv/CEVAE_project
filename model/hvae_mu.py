import torch
from torch import nn
import pyro
import pyro.distributions as dist
from model.networks_hm import Encoder, Decoder

torch.set_default_dtype(torch.float32)


class HierarchicalVAE(nn.Module):
    def __init__(self, categorical_features, continuous_features, z_dims, hidden_dim, hidden_layers, activation, cuda, beta: float = 1, binary: bool = False, binary_latents: bool = False):
        super(HierarchicalVAE, self).__init__()
        
        assert isinstance(z_dims, list) and len(z_dims) > 1, "z_dims must be a list with more than one layer size for hierarchical latents."
        
        self.encoders = nn.ModuleList([
            Encoder(len(categorical_features), len(continuous_features) if i == 0 else z_dims[i - 1], z_dims[i], hidden_dim, hidden_layers, activation, cuda)
            for i in range(len(z_dims))
        ])
        self.decoder = Decoder(
            len(categorical_features), len(continuous_features), z_dims[-1], hidden_dim, hidden_layers, activation, cuda
        )

        if cuda:
            self.cuda()
        self.cuda = cuda
        self.z_dims = z_dims
        self.categorical = categorical_features
        self.continuous = continuous_features
        self.beta = beta
        self.binary = binary
        self.binary_latents = binary_latents

    def model(self, data):
        pyro.module("decoder", self.decoder)
        x_observation = data[0].double()  # Convert to Double
        continuous_x_observation = x_observation[:, :len(self.continuous)].double()
        binary_x_observation = x_observation[:, len(self.continuous):].double()
        t_observation = data[1].double()
        y_observation = data[2].double()

        with pyro.plate("data", x_observation.shape[0]):
            z_samples = []
            for i, z_dim in enumerate(self.z_dims):
                z_loc = x_observation.new_zeros((x_observation.shape[0], z_dim))
                z_scale = x_observation.new_ones((x_observation.shape[0], z_dim))

                z_name = f"latent_{i}"
                if i == 0:  # First latent level
                    z = pyro.sample(z_name, dist.Normal(z_loc, z_scale).to_event(1))
                else:  # Hierarchical dependency
                    z_parent = z_samples[-1]
                    z = pyro.sample(z_name, dist.Normal(z_loc + z_parent, z_scale).to_event(1))
                z_samples.append(z)

            # Pass the deepest latent variable to the decoder
            (x_logits, x_loc, x_scale), (t_logits), (y_loc_t0, y_loc_t1, y_scale) = self.decoder.forward(z_samples[-1])

            pyro.sample('x_bin', dist.Bernoulli(x_logits).to_event(1), obs=binary_x_observation)
            pyro.sample('x_cont', dist.Normal(x_loc, x_scale).to_event(1), obs=continuous_x_observation)
            t = pyro.sample('t1', dist.Bernoulli(t_logits).to_event(1), obs=t_observation.view(-1, 1))
            y_loc = t * y_loc_t1 + (1. - t) * y_loc_t0
            if not self.binary:
                pyro.sample('y', dist.Normal(y_loc, y_scale).to_event(1), obs=y_observation.view(-1, 1))
            else:
                pyro.sample('y', dist.Bernoulli(torch.sigmoid(y_loc)).to_event(1), obs=y_observation.view(-1, 1))


    def guide(self, data):
        pyro.module("encoder", self.encoders)
        x_observation = data[0].double()  # Convert to Double

        z_samples = []
        with pyro.plate("data", x_observation.shape[0]):
            for i, encoder in enumerate(self.encoders):
                prefix = f"encoder_{i}_"
                if i == 0:  # First latent level
                    z_loc, z_scale = encoder.forward(x_observation, prefix=prefix)  # Inputs are already Double
                else:  # Hierarchical dependency
                    previous_z = z_samples[-1]
                    if previous_z.size(1) != encoder.input_dim:
                        # Add linear layer to match dimensions
                        projection = nn.Linear(previous_z.size(1), encoder.input_dim).to(previous_z.device).double()
                        previous_z = projection(previous_z)
                    z_loc, z_scale = encoder.forward(previous_z, prefix=prefix)

                z_name = f"latent_{i}"
                z = pyro.sample(z_name, dist.Normal(z_loc, z_scale).to_event(1))
                z_samples.append(z)


    def predict_y(self, x, L=1):
        assert L >= 1
        z_samples = []
        for i, encoder in enumerate(self.encoders):
            if i == 0:
                z_loc, _ = encoder.forward(x)
            else:
                previous_z = z_samples[-1]
                if previous_z.size(1) != encoder.input_dim:
                    # Add a projection layer if dimensions mismatch
                    projection = nn.Linear(previous_z.size(1), encoder.input_dim).to(previous_z.device).double()
                    previous_z = projection(previous_z)
                z_loc, _ = encoder.forward(previous_z)
            z_samples.append(z_loc)

        y_loc_t0, y_scale_t0 = self.decoder.forward_y(z_samples[-1], False)
        y0 = dist.Normal(y_loc_t0, y_scale_t0).sample() / L if not self.binary else \
            dist.Bernoulli(torch.sigmoid(y_loc_t0)).sample() / L

        y_loc_t1, y_scale_t1 = self.decoder.forward_y(z_samples[-1], True)
        y1 = dist.Normal(y_loc_t1, y_scale_t1).sample() / L if not self.binary else \
            dist.Bernoulli(torch.sigmoid(y_loc_t1)).sample() / L

        for _ in range(L - 1):
            y_loc_t0, y_scale_t0 = self.decoder.forward_y(z_samples[-1], False)
            y0 += dist.Normal(y_loc_t0, y_scale_t0).sample() / L if not self.binary else \
                dist.Bernoulli(torch.sigmoid(y_loc_t0)).sample() / L

            y_loc_t1, y_scale_t1 = self.decoder.forward_y(z_samples[-1], True)
            y1 += dist.Normal(y_loc_t1, y_scale_t1).sample() / L if not self.binary else \
                dist.Bernoulli(torch.sigmoid(y_loc_t1)).sample() / L

        return y0, y1
