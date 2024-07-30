from torch.amp.autocast_mode import autocast
from torch.nn import Module
from torch.amp.grad_scaler import GradScaler
from torch.optim.optimizer import Optimizer
from torch import Tensor
import torch

def reshape_imgs_masks(imgs, masks):
    """
    Reshapes the input images and masks tensors.

    Args:
        imgs (torch.Tensor): Input images tensor of shape [batch_size, num_tiles, channels, height, width].
        masks (torch.Tensor): Input masks tensor of shape [batch_size, num_tiles, height, width].

    Returns:
        torch.Tensor: Reshaped images tensor of shape [batch_size * num_tiles, channels, height, width].
        torch.Tensor: Reshaped masks tensor of shape [batch_size * num_tiles, height, width].
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    imgs, masks = imgs.to(device), masks.to(device)

    # Reshape images: [batch_size, num_tiles, channels, height, width] -> [batch_size * num_tiles, channels, height, width]
    imgs = imgs.view(-1, imgs.shape[2], imgs.shape[3], imgs.shape[4])
    # Reshape masks: [batch_size, num_tiles, height, width] -> [batch_size * num_tiles, height, width]
    masks = masks.view(-1, masks.shape[2], masks.shape[3])

    # Convert masks to Long() type
    masks = masks.to(torch.long)

    return imgs, masks

def train(
    train_loss: Tensor,
    imgs: Tensor,
    masks: Tensor,
    model: Module,
    scaler: GradScaler,
    optimizer: Optimizer,
    criterion: Module,
    iteration: int,
    accumulation_steps: int = 1,
    use_amp: bool = True,
    tiles: bool = False
) -> Tensor:
    """
    Trains the model using the given images and masks.

    Args:
        train_loss (torch.Tensor): The current training loss.
        imgs (torch.Tensor): The input images.
        masks (torch.Tensor): The target masks.
        model (torch.nn.Module): The model to be trained.
        scaler (torch.cuda.amp.GradScaler): The gradient scaler for mixed precision training.
        optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
        criterion (torch.nn.Module): The loss function.
        iteration (int): The current iteration number.
        accumulation_steps (int, optional): The number of steps to accumulate gradients before performing optimization. Defaults to 1.
        use_amp (bool, optional): Whether to use automatic mixed precision training. Defaults to True.
        tiles (bool, optional): Whether the images and masks are tiled by the Dataset. Defaults to False.

    Returns:
        torch.Tensor: The updated training loss.
    """

    device='cuda' if torch.cuda.is_available() else 'cpu'

    # Only reshape images and masks if tiles are being computed by the Dataset class
    # Else the source is the already tiled images and masks
    if tiles:
        imgs, masks = reshape_imgs_masks(imgs, masks)
    else:
        with torch.profiler.record_function("Data to device"):
            # TODO - Check if non_blocking=True is necessary
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, torch.int, non_blocking=True)
            masks = masks.to(torch.long)

    optimizer.zero_grad()

    if use_amp:
        with autocast(device_type='cuda'):
            outputs = model(imgs)
            loss = criterion(outputs, masks) / accumulation_steps
        scaler.scale(loss).backward()
    else:
        outputs = model(imgs)
        loss = criterion(outputs, masks) / accumulation_steps
        loss.backward()

    if (iteration + 1) % accumulation_steps == 0:
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    train_loss += loss.detach() * accumulation_steps  # Adjust for scaled loss

    if (iteration + 1) % 64 == 0:
        print(f'Train loss at iteration {iteration + 1}: {train_loss.item() / iteration :.3f}')

    return train_loss

def validate(
    val_loss: Tensor,
    imgs: Tensor,
    masks: Tensor,
    model: Module,
    criterion: Module,
    use_amp: bool = True,
    tiles: bool = False
) -> tuple[Tensor, Tensor]:
    """
    Perform validation on the given images and masks using the provided model and criterion.

    Args:
        val_loss (float): Current validation loss.
        imgs (torch.Tensor): Input images.
        masks (torch.Tensor): Ground truth masks.
        model (torch.nn.Module): Model to be used for validation.
        criterion (torch.nn.Module): Loss criterion.
        use_amp (bool, optional): Whether to use automatic mixed precision. Defaults to True.
        tiles (bool, optional): Whether the images and masks are tiled. Defaults to False.

    Returns:
        tuple: A tuple containing the updated validation loss and predicted masks.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        if tiles:
            imgs, masks = reshape_imgs_masks(imgs, masks)
        else:
            imgs, masks = imgs.to(device), masks.to(device)
            masks = masks.to(torch.long)

        if use_amp:
            with autocast(device_type='cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.detach()
        else:
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            val_loss += loss.detach()

        preds = torch.argmax(outputs, dim=1)

    return val_loss, preds

