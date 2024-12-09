import torch
from torch import nn
import pyro
import pyro.distributions as dist
from model.networks import Encoder, Decoder

class HierarchicalVAE(nn.Module):
    def __init__(self, categorical_features, continuous_features, z_dim, hidden_dim, hidden_layers, activation, cuda, beta: float = 1, binary: bool = False, binary_latents: bool = False):
        super(HierarchicalVAE, self).__init__()
        self.encoder = Encoder(
            len(categorical_features), len(continuous_features), z_dim, hidden_dim, hidden_layers, activation, cuda
        )
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
                z_loc = x_observation.new_zeros((x_observation.shape[0], self.z_dim))
                z_scale = x_observation.new_ones((x_observation.shape[0], self.z_dim))
                if not self.binary_latents:
                    z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
                else:
                    z = pyro.sample("latent", dist.Bernoulli(torch.sigmoid(z_loc)).to_event(1))

            (x_logits, x_loc, x_scale), (t_logits), (y_loc_t0, y_loc_t1, y_scale) = self.decoder.forward(z)

            pyro.sample('x_bin', dist.Bernoulli(x_logits).to_event(1), obs=binary_x_observation)
            pyro.sample('x_cont', dist.Normal(x_loc, x_scale).to_event(1), obs=continuous_x_observation)
            t = pyro.sample('t', dist.Bernoulli(t_logits).to_event(1), obs=t_observation.view(-1, 1))
            y_loc = t * y_loc_t1 + (1. - t) * y_loc_t0
            if not self.binary:
                pyro.sample('y', dist.Normal(y_loc, y_scale).to_event(1), obs=y_observation.view(-1, 1))
            else:
                pyro.sample('y', dist.Bernoulli(torch.sigmoid(y_loc)).to_event(1), obs=y_observation.view(-1, 1))

    def guide(self, data):
        pyro.module("encoder", self.encoder)
        x_observation = data[0]
        with pyro.plate("data", x_observation.shape[0]):
            with pyro.poutine.scale(scale=self.beta):
                z_loc, z_scale = self.encoder.forward(x_observation)
                if not self.binary_latents:
                    pyro.sample('latent', dist.Normal(z_loc, z_scale).to_event(1))
                else:
                    pyro.sample('latent', dist.Bernoulli(torch.sigmoid(z_loc)).to_event(1))

    def predict_y(self, x, L=1):
        assert L >= 1
        z_loc_t0, z_loc_t1 = self.encoder.forward_z(x)

        y_loc_t0, y_scale_t0 = self.decoder.forward_y(z_loc_t0, False)
        y0 = dist.Normal(y_loc_t0, y_scale_t0).sample() / L if not self.binary else \
             dist.Bernoulli(torch.sigmoid(y_loc_t0)).sample() / L

        y_loc_t1, y_scale_t1 = self.decoder.forward_y(z_loc_t1, True)
        y1 = dist.Normal(y_loc_t1, y_scale_t1).sample() / L if not self.binary else \
             dist.Bernoulli(torch.sigmoid(y_loc_t1)).sample() / L

        for _ in range(L - 1):
            y_loc_t0, y_scale_t0 = self.decoder.forward_y(z_loc_t0, False)
            y0 += dist.Normal(y_loc_t0, y_scale_t0).sample() / L if not self.binary else \
                  dist.Bernoulli(torch.sigmoid(y_loc_t0)).sample() / L

            y_loc_t1, y_scale_t1 = self.decoder.forward_y(z_loc_t1, True)
            y1 += dist.Normal(y_loc_t1, y_scale_t1).sample() / L if not self.binary else \
                  dist.Bernoulli(torch.sigmoid(y_loc_t1)).sample() / L

        return y0, y1
