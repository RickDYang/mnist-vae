import torch
from torch import nn


# visual example: https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73
class VAE(nn.Module):
    def __init__(self, n_classes, input_dim, hidden_dims, latent_dim):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        # promotes embedding
        self.promotes_embedding = nn.Sequential(nn.Embedding(n_classes, latent_dim))

        # hidden layers
        self.hidden_fc = self._build_fc_layers([input_dim] + hidden_dims)
        # mu and log variance layers to the latent space
        self.mu_fc = nn.Sequential(nn.Linear(hidden_dims[-1], latent_dim), nn.ReLU())
        self.logvar_fc = nn.Sequential(
            nn.Linear(hidden_dims[-1], latent_dim), nn.ReLU()
        )

        # decoder layers
        self.decoder = self._build_fc_layers([latent_dim] + hidden_dims[::-1])
        self.decoder.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder.append(nn.Sigmoid())

    def _build_fc_layers(self, dims):
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def encode(self, x):
        # to a hidden layer
        hidden_layer = self.hidden_fc(x)
        # fork the hidden layer to the mean ang log variance latent space
        mu = self.mu_fc(hidden_layer)
        log_var = self.logvar_fc(hidden_layer)

        std = torch.exp(0.5 * log_var)
        # sample from a normal distribution of the latent space
        eps = torch.randn_like(std)
        noise = mu + eps * std
        return noise, mu, log_var

    def decode(self, noise, promotes):
        # add the input and the embedding promotes
        # which will have the promotes information
        # and help the decoder to generate the right promotes.
        promotes_emb = self.promotes_embedding(promotes)
        noise = noise + promotes_emb
        return self.decoder(noise)

    def forward(self, x, promotes):
        x = x.view(-1, self.input_dim)
        noise, mu, log_var = self.encode(x)
        return self.decode(noise, promotes), mu, log_var

    def infer(self, promotes, device):
        # make a latent noise
        latent_noise = torch.randn(promotes.shape[0], self.latent_dim).to(device)
        return self.decode(latent_noise, promotes)
