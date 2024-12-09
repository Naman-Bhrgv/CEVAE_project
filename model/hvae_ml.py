import torch
from torch import nn
import pyro
import pyro.distributions as dist
from model.networks import Encoder, Decoder

class HierarchicalVAE(nn.Module):
    def __init__(self, categorical_features, continuous_features, z_dim, num_latent_layers, hidden_dim, hidden_layers, activation, cuda, beta: float = 1, binary: bool = False, binary_latents: bool = False):
        super(HierarchicalVAE, self).__init__()
        self.num_latent_layers = num_latent_layers
        self.encoders = nn.ModuleList([
            Encoder(
                len(categorical_features) + z_dim * i, z_dim, hidden_dim, hidden_layers, activation, cuda
            ) for i in range(num_latent_layers)
        ])
        self.decoder = Decoder(
            len(categorical_features), len(continuous_features), z_dim, hidden_dim, hidden_layers, activation, cuda
        )
        if cuda:
            self.cuda()
        self.cuda = cuda
        self.z_dim = z_dim
        self.categorical = categorical_features
        self.continuous = continuous_features
        self.beta = beta
        self.binary = binary
        self.binary_latents = binary_latents

    def model(self, data):
        pyro.module("decoder", self.decoder)
        x_observation = data[0]
        continuous_x_observation = x_observation[:, :len(self.continuous)]
        binary_x_observation = x_observation[:, len(self.continuous):]
        t_observation = data[1]
        y_observation = data[2]
        
        with pyro.plate("data", x_observation.shape[0]):
            with pyro.poutine.scale(scale=self.beta):
                latent_vars = []
                z_loc = x_observation.new_zeros((x_observation.shape[0], self.z_dim))
                z_scale = x_observation.new_ones((x_observation.shape[0], self.z_dim))
                
                for i in range(self.num_latent_layers):
                    if not self.binary_latents:
                        z = pyro.sample(f"latent_{i}", dist.Normal(z_loc, z_scale).to_event(1))
                    else:
                        z = pyro.sample(f"latent_{i}", dist.Bernoulli(torch.sigmoid(z_loc)).to_event(1))
                    latent_vars.append(z)
                    # Update z_loc and z_scale based on current latent layer
                    z_loc, z_scale = self.decoder.update_latent_distribution(z)

            (x_logits, x_loc, x_scale), (t_logits), (y_loc_t0, y_loc_t1, y_scale) = self.decoder.forward(latent_vars)

            pyro.sample('x_bin', dist.Bernoulli(x_logits).to_event(1), obs=binary_x_observation)
            pyro.sample('x_cont', dist.Normal(x_loc, x_scale).to_event(1), obs=continuous_x_observation)
            t = pyro.sample('t', dist.Bernoulli(t_logits).to_event(1), obs=t_observation.view(-1, 1))
            y_loc = t * y_loc_t1 + (1. - t) * y_loc_t0
            if not self.binary:
                pyro.sample('y', dist.Normal(y_loc, y_scale).to_event(1), obs=y_observation.view(-1, 1))
            else:
                pyro.sample('y', dist.Bernoulli(torch.sigmoid(y_loc)).to_event(1), obs=y_observation.view(-1, 1))

    def guide(self, data):
        pyro.module("encoder", self.encoders)
        x_observation = data[0]
        latent_vars = []
        with pyro.plate("data", x_observation.shape[0]):
            with pyro.poutine.scale(scale=self.beta):
                for i, encoder in enumerate(self.encoders):
                    if i == 0:
                        z_loc, z_scale = encoder.forward(x_observation)
                    else:
                        z_loc, z_scale = encoder.forward(torch.cat([x_observation] + latent_vars, dim=1))
                    if not self.binary_latents:
                        latent_vars.append(pyro.sample(f'latent_{i}', dist.Normal(z_loc, z_scale).to_event(1)))
                    else:
                        latent_vars.append(pyro.sample(f'latent_{i}', dist.Bernoulli(torch.sigmoid(z_loc)).to_event(1)))

    def predict_y(self, x, L=1):
        assert L >= 1
        z_loc, z_scale = self.encoders[0].forward(x)
        latent_vars = [dist.Normal(z_loc, z_scale).sample() for _ in range(self.num_latent_layers)]
        y_loc_t0, y_scale_t0 = self.decoder.forward_y(latent_vars, False)
        y_loc_t1, y_scale_t1 = self.decoder.forward_y(latent_vars, True)
        y0 = dist.Normal(y_loc_t0, y_scale_t0).sample() / L
        y1 = dist.Normal(y_loc_t1, y_scale_t1).sample() / L
        return y0, y1
