import os
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)

def train_gan(config, G, D, X_train: np.ndarray, y_train: np.ndarray, device: torch.device) -> dict:

    X_anomaly = X_train[y_train == 1]
    logger.info(" GAN Training | Anomaly samples: %d | Epochs: %d",
                len(X_anomaly), config.GAN_EPOCHS)

    tensor_X = torch.tensor(X_anomaly, dtype=torch.float32)
    loader   = DataLoader(
        TensorDataset(tensor_X),
        batch_size=config.GAN_BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    opt_G = torch.optim.Adam(G.parameters(), lr=config.GAN_LR_G, betas=config.GAN_BETAS)
    opt_D = torch.optim.Adam(D.parameters(), lr=config.GAN_LR_D, betas=config.GAN_BETAS)
    criterion = nn.BCELoss()

    sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(opt_G, T_max=config.GAN_EPOCHS)
    sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(opt_D, T_max=config.GAN_EPOCHS)

    g_losses, d_losses = [], []

    G.train(); D.train()

    for epoch in range(1, config.GAN_EPOCHS + 1):
        g_epoch, d_epoch = 0.0, 0.0
        n_batches = 0
        pbar = tqdm(loader, desc=f"GAN Epoch {epoch}/{config.GAN_EPOCHS}")
        for (real_batch,) in pbar:
            real_batch = real_batch.to(device)
            bsz        = real_batch.size(0)

            real_labels = torch.ones(bsz, 1, device=device)
            fake_labels = torch.zeros(bsz, 1, device=device)

            opt_D.zero_grad()
            z    = torch.randn(bsz, config.LATENT_DIM, device=device)
            fake = G(z).detach()

            loss_real = criterion(D(real_batch), real_labels)
            loss_fake = criterion(D(fake),       fake_labels)
            loss_D    = (loss_real + loss_fake) * 0.5
            loss_D.backward()
            opt_D.step()

            opt_G.zero_grad()
            z    = torch.randn(bsz, config.LATENT_DIM, device=device)
            fake = G(z)
            loss_G = criterion(D(fake), real_labels)
            loss_G.backward()
            opt_G.step()

            g_epoch += loss_G.item()
            d_epoch += loss_D.item()
            n_batches += 1

        g_losses.append(g_epoch / n_batches)
        d_losses.append(d_epoch / n_batches)
        sched_G.step(); sched_D.step()

        if epoch % 20 == 0 or epoch == 1:
            logger.info("  [Epoch %3d/%d]  G_loss=%.4f  D_loss=%.4f",
                        epoch, config.GAN_EPOCHS, g_losses[-1], d_losses[-1])

    G.eval()
    with torch.no_grad():
        z_synth  = torch.randn(config.GAN_SYNTHETIC_SAMPLES, config.LATENT_DIM, device=device)
        synth_X  = G(z_synth).cpu().numpy()

    logger.info(" GAN done. Generated %d synthetic anomaly samples.", config.GAN_SYNTHETIC_SAMPLES)

    os.makedirs(config.MODELS_DIR, exist_ok=True)
    torch.save(G.state_dict(), os.path.join(config.MODELS_DIR, "generator.pt"))
    torch.save(D.state_dict(), os.path.join(config.MODELS_DIR, "discriminator.pt"))

    return {
        "G":          G,
        "D":          D,
        "g_losses":   g_losses,
        "d_losses":   d_losses,
        "synth_X":    synth_X,
    }