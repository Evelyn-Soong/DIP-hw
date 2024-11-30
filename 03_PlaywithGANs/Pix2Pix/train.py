import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from FCN_network import Generator, Discriminator
from torch.optim.lr_scheduler import StepLR

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)

def train_one_epoch(gen, dis, dataloader, gen_optimizer, dis_optimizer, criterion,
                    l1_criterion, device, epoch, num_epochs, lambda_l1=100):
    """
    Train the generator and discriminator for one epoch using cGAN loss.

    Args:
        gen (nn.Module): The generator network.
        dis (nn.Module): The discriminator network.
        dataloader (DataLoader): DataLoader for the training data.
        gen_optimizer (Optimizer): Optimizer for updating generator parameters.
        dis_optimizer (Optimizer): Optimizer for updating discriminator parameters.
        criterion (Loss): Loss function for adversarial loss (BCEWithLogitsLoss).
        l1_criterion (Loss): Loss function for L1 loss.
        device (torch.device): Device to run the training on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
        lambda_l1 (float): Weight for L1 loss.
    """
    gen.train()
    dis.train()
    gen_running_loss = 0.0
    dis_running_loss = 0.0

    for i, (image_rgb, image_semantic) in enumerate(dataloader):
        # Move data to the device
        image_rgb = image_rgb.to(device)
        image_semantic = image_semantic.to(device)

        # Generate fake images
        fake_images = gen(image_semantic)

        # Real and Fake labels
        real_labels = torch.ones(image_semantic.size(0), 1, 60, 60).to(device)
        fake_labels = torch.zeros(image_semantic.size(0), 1, 60, 60).to(device)

        # ---------------------
        #  Train Discriminator
        # ---------------------

        # Zero the gradients
        dis_optimizer.zero_grad()

        # Real images
        real_outputs = dis(image_semantic, image_rgb)
        d_real_loss = criterion(real_outputs, real_labels)

        # Fake images
        fake_outputs = dis(image_semantic, fake_images.detach())
        d_fake_loss = criterion(fake_outputs, fake_labels)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.backward()
        dis_optimizer.step()

        # -----------------
        #  Train Generator
        # -----------------

        # Zero the gradients
        gen_optimizer.zero_grad()

        # Adversarial loss
        g_adv_loss = criterion(dis(image_semantic, fake_images), real_labels)

        # L1 loss
        g_l1_loss = l1_criterion(fake_images, image_rgb)

        # Total generator loss
        g_loss = g_adv_loss + lambda_l1 * g_l1_loss
        g_loss.backward()
        gen_optimizer.step()

        # Update running losses
        gen_running_loss += g_loss.item()
        dis_running_loss += d_loss.item()

        # Print loss information
        print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}],'
              f'Gen Loss: {g_loss.item():.4f}, Dis Loss: {d_loss.item():.4f}')

        # Save sample images every 5 epochs
        if epoch % 5 == 0 and i == 0:
            save_images(image_semantic, image_rgb, fake_images, 'train_results', epoch)

def validate(gen, dis, dataloader, criterion, device, epoch, num_epochs):
    """
    Validate the generator and discriminator on the validation dataset using cGAN loss.

    Args:
        gen (nn.Module): The generator network.
        dis (nn.Module): The discriminator network.
        dataloader (DataLoader): DataLoader for the validation data.
        criterion (Loss): Loss function for L1 loss.
        device (torch.device): Device to run the validation on.
        epoch (int): Current epoch number.
        num_epochs (int): Total number of epochs.
    """
    gen.eval()
    dis.eval()
    val_gen_loss = 0.0
    val_dis_loss = 0.0

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            # Move data to the device
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            # Generate fake images
            fake_images = gen(image_semantic)

            # Real and Fake labels
            real_labels = torch.ones(image_semantic.size(0), 1, 60, 60).to(device)
            fake_labels = torch.zeros(image_semantic.size(0), 1, 60, 60).to(device)

            # Discriminator loss
            real_outputs = dis(image_semantic, image_rgb)
            d_real_loss = criterion(real_outputs, real_labels)

            fake_outputs = dis(image_semantic, fake_images.detach())
            d_fake_loss = criterion(fake_outputs, fake_labels)

            d_loss = (d_real_loss + d_fake_loss) / 2

            # Generator loss
            g_loss = criterion(dis(image_semantic, fake_images), real_labels)

            # Update validation losses
            val_gen_loss += g_loss.item()
            val_dis_loss += d_loss.item()

            # Save sample images every 5 epochs
            if epoch % 5 == 0 and i == 0:
                save_images(image_semantic, image_rgb, fake_images, 'val_results', epoch)

    # Calculate average validation losses
    avg_val_gen_loss = val_gen_loss / len(dataloader)
    avg_val_dis_loss = val_dis_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Gen Loss: {avg_val_gen_loss:.4f}, Validation Dis Loss: {avg_val_dis_loss:.4f}')

def main():
    """
    Main function to set up the training and validation processes.
    """
    # Set device to GPU if available
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

    # Initialize datasets and dataloaders
    train_dataset = FacadesDataset(list_file='train_list.txt')
    val_dataset = FacadesDataset(list_file='val_list.txt')

    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=4)

    # Initialize model, loss function, and optimizer
    gen = Generator().to(device)
    dis = Discriminator().to(device)
    criterion = nn.BCELoss()
    l1_criterion = nn.L1Loss()
    gen_optimizer = optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
    dis_optimizer = optim.Adam(dis.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Add a learning rate scheduler for decay
    gen_scheduler = StepLR(gen_optimizer, step_size=100, gamma=0.1)
    dis_scheduler = StepLR(dis_optimizer, step_size=100, gamma=0.1)

    # Training loop
    num_epochs = 800
    for epoch in range(num_epochs):
        train_one_epoch(gen, dis, train_loader, gen_optimizer, dis_optimizer,
                        criterion, l1_criterion, device, epoch, num_epochs)
        validate(gen, dis, val_loader, criterion, device, epoch, num_epochs)

        # Step the schedulers after each epoch
        gen_scheduler.step()
        dis_scheduler.step()

        # Save model checkpoints every 20 epochs
        if (epoch + 1) % 20 == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(gen.state_dict(), f'checkpoints/pix2pix_gen_epoch_{epoch + 1}.pth')
            torch.save(dis.state_dict(), f'checkpoints/pix2pix_dis_epoch_{epoch + 1}.pth')

if __name__ == '__main__':
    main()
