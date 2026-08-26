import torch


def save_checkpoint(
    model,
    optimizer,
    epoch,
    batch_idx,
    loss,
    train_losses,
    val_losses,
    filename="checkpoint.pth"
):

    checkpoint = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    torch.save(
        checkpoint,
        filename
    )

    print(
        f"Checkpoint saved: {filename}"
    )


def load_checkpoint(
    model,
    optimizer,
    filename="checkpoint.pth"
):

    checkpoint = torch.load(
        filename,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    epoch = checkpoint["epoch"]
    batch_idx = checkpoint.get("batch_idx", 0)
    loss = checkpoint["loss"]
    train_losses = checkpoint.get(
        "train_losses", []
    )
    val_losses = checkpoint.get(
        "val_losses", []
    )

    print(
        f"Checkpoint loaded: epoch={epoch}, "
        f"batch={batch_idx}, loss={loss:.4f}"
    )

    return epoch, batch_idx, loss, train_losses, val_losses