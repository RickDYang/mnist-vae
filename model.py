import os

from tqdm import tqdm
import numpy as np
import torch
from torchvision.utils import save_image

from huggingface_hub import upload_file, create_repo, hf_hub_download

from vae import VAE


class MnistVaeModel:
    def __init__(self):
        # latent space dim of VAE
        self.n_classes = 10
        self.image_resolution = (1, 28, 28)
        self.input_dim = np.prod(self.image_resolution)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = VAE(
            n_classes=self.n_classes,
            input_dim=self.input_dim,
            hidden_dims=[512, 384, 256],
            latent_dim=128,
        )
        self.model = self._init_parameters(model)
        self.model_fn = "mnist_vae_model.pt"
        self.hf_modle_fn = "mnist_vae"

    def _init_parameters(self, model):
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
        return model.to(self.device)

    def load(self, model_fn: str = None):
        model_fn = model_fn or self.model_fn
        self.model.load_state_dict(torch.load(model_fn))

    def train(self, train_loader, validate_loader, epochs, lr: float = 1e-3):
        # optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        best_val_loss = float("inf")
        for epoch in range(epochs):
            # Train the model
            train_loss = self.train_epoch(
                optimizer, train_loader, f"Epoch {epoch+1}/{epochs} Train"
            )

            val_loss = self.evaluate(validate_loader, f"Epoch {epoch+1}/{epochs} Eval")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.model_fn)

            print(
                f"Epoch {epoch+1}/{epochs}, Train loss {train_loss}, Val loss {val_loss}"
            )

    def train_epoch(self, optimizer, train_loader, desc):
        self.model.train()
        train_loss = 0
        total = len(train_loader.dataset)
        for data, promotes in tqdm(train_loader, desc=desc):
            promotes = promotes.to(self.device)
            data = data.to(self.device)
            optimizer.zero_grad()
            # pylint: disable=not-callable
            recon_data, mu, log_var = self.model(data, promotes)
            loss = self.criterion(recon_data, data, mu, log_var)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        return train_loss / total

    def evaluate(self, validate_loader, desc):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, promotes in tqdm(validate_loader, desc=desc):
                data = data.to(self.device)
                promotes = promotes.to(self.device)
                # pylint: disable=not-callable
                recon, mu, log_var = self.model(data, promotes)
                val_loss += self.criterion(recon, data, mu, log_var).item()

        return val_loss / len(validate_loader.dataset)

    def infer(self, promotes: list[int]):
        self.model.eval()
        with torch.no_grad():
            # generate a random latent space vector
            promotes_tensor = torch.tensor(promotes).to(self.device)
            images_gen = self.model.infer(promotes_tensor, self.device)

            if not os.path.exists("test"):
                os.makedirs("test")

            images = []
            for i, promote in enumerate(promotes):
                image = images_gen[i].cpu().view(*self.image_resolution)
                save_image(image, f"test/mnist_vae_sample_{promote}.png")
                images.append(image)
            return images

    def upload(self, model_fn: str = None):
        model_fn = model_fn or self.model_fn
        token = os.getenv("HUGGINGFACE_TOKEN")
        repo_id = os.getenv("HUGGINGFACE_REPO")
        create_repo(
            repo_id,
            token=token,
            private=False,
            repo_type="model",
            exist_ok=True,
        )

        upload_file(
            repo_id=repo_id,
            path_or_fileobj=model_fn,
            path_in_repo=self.hf_modle_fn,
            token=token,
        )

    def from_pretrain(self):
        repo_id = os.getenv("HUGGINGFACE_REPO")
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=self.hf_modle_fn,
            cache_dir="./cache",
        )
        # model_path = try_to_load_from_cache(repo_id=repo_id, filename=self.hf_modle_fn)
        self.load(model_path)

    def criterion(self, recon_x, x, mu, log_var):
        # reconstruction loss
        loss = torch.nn.functional.binary_cross_entropy(
            recon_x, x.view(-1, self.input_dim), reduction="sum"
        )
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # regularization loss: Kullback-Leibler divergence
        kld = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return loss + kld
