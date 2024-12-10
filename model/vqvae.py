import torch
from torch import nn
import pyro
import pyro.distributions as dist
from torch.nn import functional as F
from model.networks import Encoder, Decoder

class VQVAE(nn.Module):
    def __init__(self, categorical_features, continuous_features, z_dim, hidden_dim, hidden_layers, activation, cuda,
                 num_embeddings, embedding_dim, commitment_cost):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(
            len(categorical_features), len(continuous_features), z_dim, hidden_dim, hidden_layers, activation, cuda)
        self.decoder = Decoder(
            len(categorical_features), len(continuous_features), embedding_dim, hidden_dim, hidden_layers, activation, cuda)

        self.codebook = VQEmbedding(num_embeddings, embedding_dim)
        self.commitment_cost = commitment_cost

        if cuda:
            self.cuda()
        self.cuda = cuda
        self.z_dim = z_dim
        self.categorical = categorical_features
        self.continuous = continuous_features

    def model(self, data):
        pyro.module("decoder", self.decoder)
        x_observation = data[0]
        continuous_x_observation = x_observation[:, :len(self.continuous)]
        binary_x_observation = x_observation[:, len(self.continuous):]
        t_observation = data[1]
        y_observation = data[2]

        with pyro.plate("data", x_observation.shape[0]):
            z = self.encoder(x_observation)
            
            z_quantized, vq_loss = self.codebook(z)
            
            (x_logits, x_loc, x_scale), t_logits, (y_loc_t0, y_loc_t1, y_scale) = self.decoder(z_quantized)

            # P(x|z) for categorical x
            pyro.sample('x_bin', dist.Bernoulli(x_logits).to_event(1), obs=binary_x_observation)
            pyro.sample('x_cont', dist.Normal(x_loc, x_scale).to_event(1), obs=continuous_x_observation)

            # P(t|z)
            t = pyro.sample('t1', dist.Bernoulli(t_logits).to_event(1), obs=t_observation.view(-1, 1))

            # P(y|z, t)
            y_loc = t * y_loc_t1 + (1. - t) * y_loc_t0
            pyro.sample('y1', dist.Normal(y_loc, y_scale).to_event(1), obs=y_observation.view(-1, 1))

        return vq_loss

    def guide(self, data):
        pyro.module("encoder", self.encoder)
        x_observation = data[0]
        with pyro.plate("data", x_observation.shape[0]):
            z = self.encoder(x_observation)
            z=z[0]
            self.codebook(z)  # Codebook quantization happens here, but no sampling in the guide

    def predict_y(self, x):
        z = self.encoder(x)
        z_quantized, _ = self.codebook(z)
        y_loc_t0, y_scale_t0 = self.decoder.forward_y(z_quantized, False)
        y_loc_t1, y_scale_t1 = self.decoder.forward_y(z_quantized, True)
        return y_loc_t0, y_loc_t1


class VQEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VQEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)
        self.commitment_cost=0.1

    def forward(self, z):

        z=z[0]
        z_flattened = z.view(-1, z.size(-1))

        # Compute distances
        distances = (torch.sum(z_flattened**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(z_flattened, self.embedding.weight.t()))

        # Get the closest embedding indices
        indices = torch.argmin(distances, dim=1).unsqueeze(1)
        z_quantized = self.embedding(indices).view(z.shape)

        # Compute VQ Loss
        commitment_loss = F.mse_loss(z_quantized.detach(), z)
        vq_loss = F.mse_loss(z, z_quantized.detach()) + self.commitment_cost * commitment_loss

        # Straight-through estimator for backpropagation
        z_quantized = z + (z_quantized - z).detach()
        return z_quantized, vq_loss