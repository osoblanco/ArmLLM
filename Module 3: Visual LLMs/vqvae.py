import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import numpy as np


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc
        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=self.inv_freq.dtype)
        pos_y = torch.arange(y, device=tensor.device, dtype=self.inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb(sin_inp_x).unsqueeze(1)
        emb_y = get_emb(sin_inp_y)
        emb = torch.zeros(
            (x, y, self.channels * 2),
            device=tensor.device,
            dtype=tensor.dtype,
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y
        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc


class ViTEncoder(nn.Module):
    def __init__(
        self, in_channels, num_heads, num_layers, embedding_dim, patch_size, image_size
    ):
        super().__init__()
        self.patch_embed = nn.Conv2d(
            in_channels, embedding_dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_encoding = PositionalEncoding2D(embedding_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads),
            num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        pos_encoding = self.pos_encoding(x)
        x = x + pos_encoding
        x = x.view(x.size(0), -1, x.size(-1))  # (B, H*W, C)
        x = self.transformer(x)
        x = self.norm(x)
        return x.view(
            x.size(0), int(x.size(1) ** 0.5), int(x.size(1) ** 0.5), x.size(2)
        ).permute(0, 3, 1, 2)


class ViTDecoder(nn.Module):
    def __init__(
        self, out_channels, num_heads, num_layers, embedding_dim, patch_size, image_size
    ):
        super().__init__()
        self.pos_encoding = PositionalEncoding2D(embedding_dim)
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads),
            num_layers=num_layers,
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.unpatch = nn.ConvTranspose2d(
            embedding_dim, out_channels, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        pos_encoding = self.pos_encoding(x)
        x = x + pos_encoding
        x = x.view(x.size(0), -1, x.size(-1))  # (B, H*W, C)
        x = self.transformer(x, x)
        x = self.norm(x)
        x = x.view(
            x.size(0), int(x.size(1) ** 0.5), int(x.size(1) ** 0.5), x.size(2)
        ).permute(0, 3, 1, 2)
        x = self.unpatch(x)
        return torch.sigmoid(x)


class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta):
        super(VectorQuantizer, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)

        # Add a buffer to track codebook usage
        self.register_buffer("usage", torch.zeros(self.n_e))

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.e_dim)

        # TODO: Implement the distance metric d between points.
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e).to(
            z.device
        )
        min_encodings.scatter_(1, min_encoding_indices, 1)

        # Update usage
        self.usage += min_encodings.sum(0)

        z_q = torch.matmul(min_encodings, self.embedding.weight).view(z.shape)

        # TODO: loss = ...

        z_q = z + (z_q - z).detach()

        e_mean = torch.mean(min_encodings, dim=0)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + 1e-10)))

        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return loss, z_q, perplexity, min_encodings, min_encoding_indices


class ViT_VQVAE(nn.Module):
    def __init__(
        self,
        in_channels,
        latent_dim,
        num_embeddings,
        num_heads,
        num_layers,
        patch_size,
        image_size,
        beta=0.25,
    ):
        super().__init__()
        self.encoder = ViTEncoder(
            in_channels, num_heads, num_layers, latent_dim, patch_size, image_size
        )
        self.vq = VectorQuantizer(num_embeddings, latent_dim, beta)
        self.decoder = ViTDecoder(
            in_channels, num_heads, num_layers, latent_dim, patch_size, image_size
        )

    def forward(self, x):
        z = self.encoder(x)
        vq_loss, quantized, perplexity, _, _ = self.vq(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, z, quantized, perplexity

    def calculate_loss(self, x, x_recon, vq_loss):
        recon_loss = F.mse_loss(x_recon, x)
        total_loss = recon_loss + vq_loss
        return total_loss, recon_loss, vq_loss

    def print_codebook_utilization(self):
        total_usage = self.vq.usage.sum().item()
        used_codes = (self.vq.usage > 0).sum().item()
        utilization = used_codes / self.vq.n_e * 100
        print(
            f"Codebook utilization: {utilization:.2f}% ({used_codes}/{self.vq.n_e} codes used)"
        )

        # Calculate and print histogram of usage
        usage_counts, _ = torch.histogram(self.vq.usage, bins=10)
        print("Usage histogram:")
        for i, count in enumerate(usage_counts):
            print(f"Bin {i+1}: {count.item()}")


def load_laion_art_dataset(batch_size, resolution):
    dataset = load_dataset("fantasyfish/laion-art", split="train[:100]")

    def preprocess_image(example):
        image = example["image"].convert("RGB")
        image = transforms.Resize((resolution, resolution))(image)
        image = transforms.ToTensor()(image)
        # Normalize the image to have pixel values between 0 and 1
        # This step is achieved by ToTensor() which scales the image to [0, 1] by dividing by 255
        return {"image": image}

    dataset = dataset.map(preprocess_image, remove_columns=dataset.column_names)
    dataset.set_format(type="torch", columns=["image"])

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


def train(model, dataloader, optimizer, num_epochs, device):
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            x = batch["image"].to(device)
            optimizer.zero_grad()
            x_recon, vq_loss, _, _, _ = model(x)
            loss, recon_loss, _ = model.calculate_loss(x, x_recon, vq_loss)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        # Print codebook utilization every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.print_codebook_utilization()
            visualize_results(model, x, x_recon, epoch + 1)


def visualize_results(model, original, reconstructed, epoch):
    model.eval()
    with torch.no_grad():
        idx = torch.randint(0, original.size(0), (5,))
        original = original[idx].cpu()
        reconstructed = reconstructed[idx].cpu()

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for i in range(5):
            axes[0, i].imshow(original[i].permute(1, 2, 0))
            axes[0, i].axis("off")
            axes[1, i].imshow(reconstructed[i].permute(1, 2, 0))
            axes[1, i].axis("off")

        axes[0, 0].set_title("Original")
        axes[1, 0].set_title("Reconstructed")
        plt.tight_layout()
        plt.savefig(f"reconstruction_epoch_{epoch}.png")
        plt.close()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloader = load_laion_art_dataset(args.batch_size, args.resolution)

    model = ViT_VQVAE(
        in_channels=3,
        latent_dim=args.latent_dim,
        num_embeddings=args.num_embeddings,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        patch_size=args.patch_size,
        image_size=args.resolution,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train(model, dataloader, optimizer, args.num_epochs, device)

    torch.save(
        model.state_dict(),
        f"vit_vqvae_laion_art_{args.resolution}x{args.resolution}.pth",
    )
    print("Training completed and model saved.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT-VQVAE on LAION-Art dataset")
    parser.add_argument("--resolution", type=int, default=64, help="Image resolution")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument("--latent_dim", type=int, default=256, help="Latent dimension")
    parser.add_argument(
        "--num_embeddings", type=int, default=512, help="Number of embeddings for VQ"
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_layers", type=int, default=6, help="Number of transformer layers"
    )
    parser.add_argument("--patch_size", type=int, default=8, help="Patch size for ViT")

    args = parser.parse_args()
    main(args)
