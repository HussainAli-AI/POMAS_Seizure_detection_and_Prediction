"""Conditional WGAN-GP for 1D EEG Data Augmentation.

Phase 2: Generates synthetic preictal EEG windows to balance training set.
Falls back to traditional augmentation (time-warp, noise injection, mixup) if OOM.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import SAMPLE_RATE, WINDOW_SIZE


class Generator1D(nn.Module):
    """Conditional Generator: 1D transposed CNN for EEG signals.

    Input: latent vector z + condition embedding -> synthetic EEG window
    """

    def __init__(
        self,
        latent_dim: int = 100,
        n_channels: int = 18,
        window_size: int = WINDOW_SIZE,
        n_classes: int = 3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_channels = n_channels

        self.label_embedding = nn.Embedding(n_classes, latent_dim)

        self.model = nn.Sequential(
            nn.ConvTranspose1d(latent_dim, 512, kernel_size=16, stride=2, padding=4),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.ConvTranspose1d(512, 256, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, n_channels, kernel_size=8, stride=2, padding=3),
            nn.Tanh(),
        )

        self._init_output_size()

    def _init_output_size(self):
        dummy = torch.zeros(1, self.latent_dim, 1)
        out = self.model(dummy)
        self.output_length = out.shape[-1]

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        c = self.label_embedding(labels)
        z = z * c
        z = z.unsqueeze(-1)
        return self.model(z)


class Critic1D(nn.Module):
    """Critic (Discriminator) for WGAN-GP.

    Outputs a single scalar (critic score) per sample.
    """

    def __init__(self, n_channels: int = 18, window_size: int = WINDOW_SIZE, n_classes: int = 3):
        super().__init__()
        self.label_embedding = nn.Embedding(n_classes, window_size)

        self.model = nn.Sequential(
            nn.Conv1d(n_channels + 1, 64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        c = self.label_embedding(labels).unsqueeze(1)
        x = torch.cat([x, c], dim=1)
        return self.model(x)


class ConditionalWGAN_GP:
    """Conditional WGAN with Gradient Penalty for EEG generation."""

    def __init__(
        self,
        n_channels: int = 18,
        window_size: int = WINDOW_SIZE,
        latent_dim: int = 100,
        n_classes: int = 3,
        lr: float = 1e-4,
        lambda_gp: float = 10,
        n_critic: int = 5,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.generator = Generator1D(latent_dim, n_channels, window_size, n_classes).to(self.device)
        self.critic = Critic1D(n_channels, window_size, n_classes).to(self.device)
        self.latent_dim = latent_dim
        self.lambda_gp = lambda_gp
        self.n_critic = n_critic

        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(0.5, 0.9))
        self.c_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr, betas=(0.5, 0.9))

    def _gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = real.size(0)
        epsilon = torch.rand(batch_size, 1, 1, device=self.device)
        interpolated = epsilon * real + (1 - epsilon) * fake
        interpolated.requires_grad_(True)
        critic_interp = self.critic(interpolated, labels)
        gradients = torch.autograd.grad(
            outputs=critic_interp,
            inputs=interpolated,
            grad_outputs=torch.ones_like(critic_interp),
            create_graph=True,
            retain_graph=True,
        )[0]
        gradients = gradients.view(batch_size, -1)
        gradient_norm = gradients.norm(2, dim=1)
        return self.lambda_gp * ((gradient_norm - 1) ** 2).mean()

    def train_step(self, real: torch.Tensor, labels: torch.Tensor) -> dict:
        batch_size = real.size(0)

        for _ in range(self.n_critic):
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            fake = self.generator(z, labels)
            critic_real = self.critic(real, labels)
            critic_fake = self.critic(fake.detach(), labels)
            gp = self._gradient_penalty(real, fake.detach(), labels)
            c_loss = critic_fake.mean() - critic_real.mean() + gp
            self.c_optimizer.zero_grad()
            c_loss.backward()
            self.c_optimizer.step()

        z = torch.randn(batch_size, self.latent_dim, device=self.device)
        fake = self.generator(z, labels)
        critic_fake = self.critic(fake, labels)
        g_loss = -critic_fake.mean()
        self.g_optimizer.zero_grad()
        g_loss.backward()
        self.g_optimizer.step()

        return {"g_loss": g_loss.item(), "c_loss": c_loss.item()}

    def generate(self, n_samples: int, labels: torch.Tensor) -> torch.Tensor:
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, device=self.device)
            samples = self.generator(z, labels.to(self.device))
        return samples.cpu()


def traditional_augmentation(windows: np.ndarray, labels: np.ndarray, factor: float = 0.5) -> tuple:
    """Traditional data augmentation fallback.

    Applies: time-warp, noise injection, mixup.
    """
    n = len(windows)
    n_aug = int(n * factor)
    indices = np.random.choice(n, n_aug, replace=False)
    aug_windows = []
    aug_labels = []

    for idx in indices:
        w = windows[idx].copy()
        aug_type = np.random.choice(["noise", "time_warp", "mixup"])
        if aug_type == "noise":
            noise = np.random.randn(*w.shape) * 0.02
            w = w + noise
        elif aug_type == "time_warp":
            shift = np.random.randint(-10, 10)
            w = np.roll(w, shift, axis=-1)
        elif aug_type == "mixup":
            other_idx = np.random.choice(n)
            lam = np.random.beta(0.5, 0.5)
            w = lam * w + (1 - lam) * windows[other_idx]

        aug_windows.append(w)
        aug_labels.append(labels[idx])

    if aug_windows:
        return (
            np.concatenate([windows, np.array(aug_windows)], axis=0),
            np.concatenate([labels, np.array(aug_labels)], axis=0),
        )
    return windows, labels
