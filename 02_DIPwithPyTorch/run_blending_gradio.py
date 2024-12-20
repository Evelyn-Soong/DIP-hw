import gradio as gr
from PIL import Image, ImageDraw
import numpy as np
import torch


# Initialize the polygon state
def initialize_polygon():
    """
    Initializes the polygon state.

    Returns:
        dict: A dictionary with 'points' and 'closed' status.
    """

    return {'points': [], 'closed': False}


# Add a point to the polygon when the user clicks on the image
def add_point(img_original, polygon_state, evt: gr.SelectData):
    """
    Adds a point to the polygon based on user click event.

    Args:
        img_original (PIL.Image): The original image.
        polygon_state (dict): The current state of the polygon.
        evt (gr.SelectData): The click event data.

    Returns:
        tuple: Updated image with polygon and updated polygon state.
    """
    if polygon_state['closed']:
        return img_original, polygon_state  # Do not add points if polygon is closed

    x, y = evt.index
    polygon_state['points'].append((x, y))

    img_with_poly = img_original.copy()
    draw = ImageDraw.Draw(img_with_poly)

    # Draw lines between points
    if len(polygon_state['points']) > 1:
        draw.line(polygon_state['points'], fill='red', width=2)

    # Draw points
    for point in polygon_state['points']:
        draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill='blue')

    return img_with_poly, polygon_state


# Close the polygon when the user clicks the "Close Polygon" button
def close_polygon(img_original, polygon_state):
    """
    Closes the polygon if there are at least three points.

    Args:
        img_original (PIL.Image): The original image.
        polygon_state (dict): The current state of the polygon.

    Returns:
        tuple: Updated image with closed polygon and updated polygon state.
    """
    if not polygon_state['closed'] and len(polygon_state['points']) > 2:
        polygon_state['closed'] = True
        img_with_poly = img_original.copy()
        draw = ImageDraw.Draw(img_with_poly)
        draw.polygon(polygon_state['points'], outline='red')
        return img_with_poly, polygon_state
    else:
        return img_original, polygon_state


# Update the background image by drawing the shifted polygon on it
def update_background(background_image_original, polygon_state, dx, dy):
    """
    Updates the background image by drawing the shifted polygon on it.

    Args:
        background_image_original (PIL.Image): The original background image.
        polygon_state (dict): The current state of the polygon.
        dx (int): Horizontal offset.
        dy (int): Vertical offset.

    Returns:
        PIL.Image: The updated background image with the polygon overlay.
    """
    if background_image_original is None:
        return None

    if polygon_state['closed']:
        img_with_poly = background_image_original.copy()
        draw = ImageDraw.Draw(img_with_poly)
        shifted_points = [(x + dx, y + dy) for x, y in polygon_state['points']]
        draw.polygon(shifted_points, outline='red')
        return img_with_poly
    else:
        return background_image_original


# Create a binary mask from polygon points
def create_mask_from_points(points, img_h, img_w):
    """
    Creates a binary mask from the given polygon points.

    Args:
        points (np.ndarray): Polygon points of shape (n, 2).
        img_h (int): Image height.
        img_w (int): Image width.

    Returns:
        np.ndarray: Binary mask of shape (img_h, img_w).
    """
    mask = Image.new('L', (img_w, img_h), 0)
    ### FILL: Obtain Mask from Polygon Points. 
    ### 0 indicates outside the Polygon.
    ### 255 indicates inside the Polygon.

    # Draw the polygon directly with a fill value of 255
    ImageDraw.Draw(mask).polygon(list(points.flatten()), fill=255)

    # Convert the PIL image to a NumPy array
    mask = np.array(mask, dtype=np.uint8)

    return mask


# Calculate the Laplacian loss between the foreground and blended image
def cal_laplacian_loss(foreground_img, foreground_mask, blended_img, background_mask, flag=0):
    """
    Computes the Laplacian loss between the foreground and blended images within the masks.

    Args:
        flag: Divergence
        foreground_img (torch.Tensor): Foreground image tensor.
        foreground_mask (torch.Tensor): Foreground mask tensor.
        blended_img (torch.Tensor): Blended image tensor.
        background_mask (torch.Tensor): Background mask tensor.

    Returns:
        torch.Tensor: The computed Laplacian loss.
    """

    ### FILL: Compute Laplacian Loss with https://pytorch.org/docs/stable/generated/torch.nn.functional.conv2d.html.
    ### Note: The loss is computed within the masks.
    if foreground_img.size(2) != blended_img.size(2) or foreground_img.size(3) != blended_img.size(3):
        height_diff = (blended_img.size(2) - foreground_img.size(2)) // 2
        width_diff = (blended_img.size(3) - foreground_img.size(3)) // 2
        padding = (width_diff, blended_img.size(3) - foreground_img.size(3) - width_diff,
                   height_diff, blended_img.size(2) - foreground_img.size(2) - height_diff)
        foreground_img = torch.nn.functional.pad(foreground_img, padding)
        foreground_mask = torch.nn.functional.pad(foreground_mask, padding)

    if flag == 0:
        # Define the laplacian kernel
        laplace_kernel_single_channel = torch.tensor([[0, 1, 0],
                                                      [1, -4, 1],
                                                      [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3).to(
            foreground_img.device)

        # Expand the kernel for each channel
        num_channels = foreground_img.shape[1]
        laplace_kernel = torch.cat([laplace_kernel_single_channel] * num_channels, dim=0)
        laplace_kernel = laplace_kernel.expand(num_channels, -1, -1, -1)

        # Apply the kernel to the foreground and blended images
        fg_laplacian = torch.nn.functional.conv2d(foreground_img, laplace_kernel, padding=1, groups=num_channels)
        bg_laplacian = torch.nn.functional.conv2d(blended_img, laplace_kernel, padding=1, groups=num_channels)

        # Expand the foreground mask to match the number of channels in the image
        foreground_mask_expanded = foreground_mask.expand(-1, num_channels, -1, -1).bool()
        background_mask_expanded = background_mask.expand(-1, num_channels, -1, -1).bool()

        # Apply the masks to the Laplacian results
        fg_laplacian_masked = fg_laplacian[foreground_mask_expanded]
        bg_laplacian_masked = bg_laplacian[background_mask_expanded]

        # Compute the loss only in the region where the foreground mask is active
        # loss = torch.mean((fg_laplacian - bg_laplacian) ** 2 * foreground_mask_expanded)
        # Compute the loss using MSELoss
        mse_loss = torch.nn.MSELoss()
        loss = mse_loss(fg_laplacian_masked, bg_laplacian_masked)

    else:
        # Define kernels for gradient computation in four directions
        kernel_x_pos = torch.tensor([[0, -1, 1]], dtype=torch.float32).view(1, 1, 1, 3).to(foreground_img.device)
        kernel_x_neg = torch.tensor([[1, -1, 0]], dtype=torch.float32).view(1, 1, 1, 3).to(foreground_img.device)
        kernel_y_pos = torch.tensor([[0], [-1], [1]], dtype=torch.float32).view(1, 1, 3, 1).to(foreground_img.device)
        kernel_y_neg = torch.tensor([[1], [-1], [0]], dtype=torch.float32).view(1, 1, 3, 1).to(foreground_img.device)

        # Extend kernels to match the number of channels in the image
        num_channels = foreground_img.shape[1]
        kernel_x_pos = kernel_x_pos.repeat(num_channels, 1, 1, 1)
        kernel_x_neg = kernel_x_neg.repeat(num_channels, 1, 1, 1)
        kernel_y_pos = kernel_y_pos.repeat(num_channels, 1, 1, 1)
        kernel_y_neg = kernel_y_neg.repeat(num_channels, 1, 1, 1)

        # Apply the mask
        foreground_mask_expanded = foreground_mask.expand(-1, num_channels, -1, -1).bool()
        background_mask_expanded = background_mask.expand(-1, num_channels, -1, -1).bool()

        # Compute gradients in four directions
        grads = []
        for kernel in [kernel_x_pos, kernel_x_neg, kernel_y_pos, kernel_y_neg]:
            src_grad = torch.nn.functional.conv2d(foreground_img, kernel,
                                                  padding=(kernel.size(2) // 2, kernel.size(3) // 2),
                                                  groups=num_channels)
            dst_grad = torch.nn.functional.conv2d(blended_img, kernel,
                                                  padding=(kernel.size(2) // 2, kernel.size(3) // 2),
                                                  groups=num_channels)

            src_grad = src_grad[foreground_mask_expanded]
            dst_grad = dst_grad[background_mask_expanded]
            grads.append(torch.where(torch.abs(src_grad) > torch.abs(dst_grad), src_grad ** 2, dst_grad ** 2))

        # Sum up the gradients from all directions
        final_grad = torch.sum(torch.stack(grads), dim=0)

        # Sum up the gradients in both directions
        loss = torch.mean(final_grad)

    return loss


# Perform Poisson image blending
def blending(foreground_image_original, background_image_original, dx, dy, polygon_state):
    """
    Blends the foreground polygon area onto the background image using Poisson blending.

    Args:
        foreground_image_original (PIL.Image): The original foreground image.
        background_image_original (PIL.Image): The original background image.
        dx (int): Horizontal offset.
        dy (int): Vertical offset.
        polygon_state (dict): The current state of the polygon.

    Returns:
        np.ndarray: The blended image as a numpy array.
    """
    if not polygon_state['closed'] or background_image_original is None or foreground_image_original is None:
        return background_image_original  # Return original background if conditions are not met

    # Convert images to numpy arrays
    foreground_np = np.array(foreground_image_original)
    background_np = np.array(background_image_original)

    # Get polygon points and shift them by dx and dy
    foreground_polygon_points = np.array(polygon_state['points']).astype(np.int64)
    background_polygon_points = foreground_polygon_points + np.array([int(dx), int(dy)]).reshape(1, 2)

    # Create masks from polygon points
    foreground_mask = create_mask_from_points(foreground_polygon_points, foreground_np.shape[0], foreground_np.shape[1])
    background_mask = create_mask_from_points(background_polygon_points, background_np.shape[0], background_np.shape[1])

    # Convert numpy arrays to torch tensors
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # Using CPU will be slow
    fg_img_tensor = torch.from_numpy(foreground_np).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.
    bg_img_tensor = torch.from_numpy(background_np).to(device).permute(2, 0, 1).unsqueeze(0).float() / 255.
    fg_mask_tensor = torch.from_numpy(foreground_mask).to(device).unsqueeze(0).unsqueeze(0).float() / 255.
    bg_mask_tensor = torch.from_numpy(background_mask).to(device).unsqueeze(0).unsqueeze(0).float() / 255.

    # Initialize blended image
    blended_img = bg_img_tensor.clone()
    mask_expanded = bg_mask_tensor.bool().expand(-1, 3, -1, -1)
    blended_img[mask_expanded] = blended_img[mask_expanded] * 0.9 + fg_img_tensor[
        fg_mask_tensor.bool().expand(-1, 3, -1, -1)] * 0.1
    blended_img.requires_grad = True

    # Set up optimizer
    optimizer = torch.optim.Adam([blended_img], lr=1e-3)

    # Optimization loop
    iter_num = 10000
    for step in range(iter_num):
        blended_img_for_loss = blended_img.detach() * (
                1. - bg_mask_tensor) + blended_img * bg_mask_tensor  # Only blending in the mask region

        loss = cal_laplacian_loss(fg_img_tensor, fg_mask_tensor, blended_img_for_loss, bg_mask_tensor)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print(f'Optimize step: {step}, Laplacian distance loss: {loss.item()}')

        if step == (iter_num // 2):  ### decrease learning rate at the half step
            optimizer.param_groups[0]['lr'] *= 0.1

    # Convert result back to numpy array
    result = torch.clamp(blended_img.detach(), 0, 1).cpu().permute(0, 2, 3, 1).squeeze().numpy() * 255
    result = result.astype(np.uint8)
    return result


# Helper function to close the polygon and reset dx
def close_polygon_and_reset_dx(img_original, polygon_state, dx, dy, background_image_original):
    """
    Closes the polygon, resets dx to 0, and updates the background image.

    Args:
        img_original (PIL.Image): The original image.
        polygon_state (dict): The current state of the polygon.
        dx (int): Horizontal offset.
        dy (int): Vertical offset.
        background_image_original (PIL.Image): The original background image.

    Returns:
        tuple: Updated image with polygon, updated polygon state, updated background image, and reset dx value.
    """
    # Close polygon
    img_with_poly, updated_polygon_state = close_polygon(img_original, polygon_state)

    # Reset dx value to 0
    new_dx = gr.update(value=0)

    # Update background image
    updated_background = update_background(background_image_original, updated_polygon_state, 0, dy)
    return img_with_poly, updated_polygon_state, updated_background, new_dx


# Gradio Interface
with gr.Blocks(title="Poisson Image Blending", css="""
    body {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .gr-button {
        font-size: 1em;
        padding: 0.75em 1.5em;
        border-radius: 8px;
        background-color: #6200ee;
        color: #ffffff;
        border: none;
    }
    .gr-button:hover {
        background-color: #3700b3;
    }
    .gr-slider input[type=range] {
        accent-color: #03dac6;
    }
    .gr-text, .gr-markdown {
        font-size: 1.1em;
    }
    .gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
        color: #bb86fc;
    }
    .gr-input, .gr-output {
        background-color: #2c2c2c;
        border: 1px solid #3c3c3c;
    }
""") as demo:
    # Initialize states
    polygon_state = gr.State(initialize_polygon())
    background_image_original = gr.State(value=None)

    # Title and description
    gr.Markdown("<h1 style='text-align: center;'>Poisson Image Blending</h1>")
    gr.Markdown(
        "<p style='text-align: center; font-size: 1.2em;'>Blend a selected area from a foreground image onto a background image with adjustable positions.</p>")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Foreground Image")
            foreground_image_original = gr.Image(
                label="", type="pil", interactive=True, height=300
            )
            gr.Markdown(
                "<p style='font-size: 0.9em;'>Upload the foreground image where the polygon will be selected.</p>"
            )
            gr.Markdown("### Foreground Image with Polygon")
            foreground_image_with_polygon = gr.Image(
                label="", type="pil", interactive=True, height=300
            )
            gr.Markdown(
                "<p style='font-size: 0.9em;'>Click on the image to define the polygon area. After selecting at least three points, click <strong>Close Polygon</strong>.</p>"
            )
            close_polygon_button = gr.Button("Close Polygon")
        with gr.Column():
            gr.Markdown("### Background Image")
            background_image = gr.Image(
                label="", type="pil", interactive=True, height=300
            )
            gr.Markdown(
                "<p style='font-size: 0.9em;'>Upload the background image where the polygon will be placed.</p>")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Background Image with Polygon Overlay")
            background_image_with_polygon = gr.Image(
                label="", type="pil", height=500
            )
            gr.Markdown("<p style='font-size: 0.9em;'>Adjust the position of the polygon using the sliders below.</p>")
        with gr.Column():
            gr.Markdown("### Blended Image")
            output_image = gr.Image(
                label="", type="pil", height=500  # Increased height for larger display
            )

    with gr.Row():
        with gr.Column():
            dx = gr.Slider(
                label="Horizontal Offset", minimum=-500, maximum=500, step=1, value=0
            )
        with gr.Column():
            dy = gr.Slider(
                label="Vertical Offset", minimum=-500, maximum=500, step=1, value=0
            )
        blend_button = gr.Button("Blend Images")

    # Interactions

    # Copy the original image to the interactive image when uploaded
    foreground_image_original.change(
        fn=lambda img: img,
        inputs=foreground_image_original,
        outputs=foreground_image_with_polygon,
    )

    # User interacts with the image with polygon
    foreground_image_with_polygon.select(
        add_point,
        inputs=[foreground_image_original, polygon_state],
        outputs=[foreground_image_with_polygon, polygon_state],
    )

    close_polygon_button.click(
        fn=close_polygon_and_reset_dx,
        inputs=[foreground_image_original, polygon_state, dx, dy, background_image_original],
        outputs=[foreground_image_with_polygon, polygon_state, background_image_with_polygon, dx],
    )

    background_image.change(
        fn=lambda img: img,
        inputs=background_image,
        outputs=background_image_original,
    )

    # Update background image when dx or dy changes
    dx.change(
        fn=update_background,
        inputs=[background_image_original, polygon_state, dx, dy],
        outputs=background_image_with_polygon,
    )
    dy.change(
        fn=update_background,
        inputs=[background_image_original, polygon_state, dx, dy],
        outputs=background_image_with_polygon,
    )

    # Blend images when button is clicked
    blend_button.click(
        fn=blending,
        inputs=[foreground_image_original, background_image_original, dx, dy, polygon_state],
        outputs=output_image,
    )

# Launch the Gradio app
demo.launch()
