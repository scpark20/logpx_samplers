import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image



class StateCollector:
    """Callable class to collect intermediate states from a sampling loop."""
    def __init__(self):
        self.states = []
        self.gradients = []  # gradient들을 저장

    def __call__(self, step: int, x_t: torch.Tensor, v_t: torch.Tensor = None) -> None:
        state = x_t.clone().detach()
        self.states.append(state)
        
        # v_t가 제공되면 gradient로 저장
        if v_t is not None:
            gradient = v_t.clone().detach()
            self.gradients.append(gradient)
            #print(f"[step {step}] x_t: {tuple(state.shape)}, gradient: {tuple( grdient.shape)}")
        else:
            print(f"[step {step}] {tuple(state.shape)}")


def to_image_array(x: torch.Tensor) -> np.ndarray:
    """Convert a tensor to a HWC numpy uint8 array for image display."""
    arr = x.detach().cpu()
    # [B,C,H,W] -> [H,W,C]
    if arr.ndim == 4 and arr.shape[1] in (1, 3):
        arr = arr[0].permute(1, 2, 0)
    # [B,H,W,C] -> [H,W,C]
    elif arr.ndim == 4 and arr.shape[3] in (1, 3):
        arr = arr[0]
    # [C,H,W] -> [H,W,C]
    elif arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.permute(1, 2, 0)
    # [H,W,C] or [H,W] stay as is
    elif arr.ndim == 2 or (arr.ndim == 3 and arr.shape[2] in (1, 3)):
        pass
    else:
        raise ValueError(f"Unexpected tensor shape: {arr.shape}")
    np_arr = arr.numpy()
    if np_arr.dtype in (np.float32, np.float64):
        np_arr = np.clip(np_arr, 0.0, 1.0)
        np_arr = (np_arr * 255).round()
    return np_arr.astype(np.uint8)


def visualize_latent(latent: torch.Tensor, channel: int = 0) -> Image.Image:
    """Visualize a latent tensor by selecting first batch and specified channel."""
    lt = latent
    if isinstance(latent, dict) and 'samples' in latent:
        lt = latent['samples']
    # Expect [B,C,H,W] or [C,H,W]
    if lt.ndim == 4:
        lt = lt[0, channel]  # [0,channel] - 지정된 채널 선택
    elif lt.ndim == 3:
        lt = lt[channel]  # [channel] - 지정된 채널 선택
    else:
        raise ValueError(f"Bad latent shape: {lt.shape}")
    # Convert bfloat16
    if lt.dtype == torch.bfloat16:
        lt = lt.to(torch.float32)
    # Normalize to [0,1]
    arr = lt.detach().cpu().numpy().astype(np.float32)
    arr -= arr.min()
    denom = arr.max() if arr.max() > 0 else 1
    arr /= denom
    img_arr = (arr * 255).round().astype(np.uint8)
    return Image.fromarray(img_arr, mode='L')


def visualize_decoded(decoded) -> Image.Image:
    """Visualize a decoded tensor or PIL.Image using to_image_array."""
    # If already a PIL Image, return directly
    if isinstance(decoded, Image.Image):
        return decoded
    # Otherwise assume torch.Tensor or array-like
    arr = to_image_array(decoded)
    return Image.fromarray(arr)


def show_compare(
    img1,
    img2,
    name1: str = "Image 1",
    name2: str = "Image 2",
    viz1 = visualize_latent,
    viz2 = visualize_decoded,
    figsize: tuple = (6, 4),
    channel: int = 0
) -> None:
    """Display two images side by side in color."""
    # channel 파라미터를 viz1에 전달 (visualize_latent인 경우)
    if viz1 == visualize_latent:
        pil1 = viz1(img1, channel)
    else:
        pil1 = viz1(img1)
    
    pil2 = viz2(img2)
    arr1 = np.asarray(pil1)
    arr2 = np.asarray(pil2)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(arr1)
    axes[0].axis('off')
    axes[0].set_title(name1)
    axes[1].imshow(arr2)
    axes[1].axis('off')
    axes[1].set_title(name2)
    plt.tight_layout()
    plt.show()


def show_latent_compare(
    latent1,
    latent2,
    name1: str = "Latent 1",
    name2: str = "Latent 2",
    figsize: tuple = (6, 4),
    channel: int = 0
) -> None:
    """Compare two latent tensors side by side."""
    show_compare(
        latent1,
        latent2,
        name1=name1,
        name2=name2,
        viz1=visualize_latent,
        viz2=visualize_latent,
        figsize=figsize,
        channel=channel
    )


def show_decoded_compare(
    dec1,
    dec2,
    name1: str = "Decoded 1",
    name2: str = "Decoded 2",
    figsize: tuple = (6, 4)
) -> None:
    """Compare two decoded images side by side."""
    show_compare(
        dec1,
        dec2,
        name1=name1,
        name2=name2,
        viz1=visualize_decoded,
        viz2=visualize_decoded,
        figsize=figsize
    )


def show_latent_vs_decoded(
    latent,
    decoded,
    name1: str = "Latent",
    name2: str = "Decoded",
    figsize: tuple = (6, 4),
    channel: int = 24
) -> None:
    """Compare a latent tensor with its decoded image side by side."""
    show_compare(
        latent,
        decoded,
        name1=name1,
        name2=name2,
        viz1=visualize_latent,
        viz2=visualize_decoded,
        figsize=figsize,
        channel=channel
    )

def show_three_decoded_compare(
    dec1,
    dec2,
    dec3,
    name1: str = "Decoded 1",
    name2: str = "Decoded 2",
    name3: str = "Decoded 3",
    figsize: tuple = (18, 6)
) -> None:
    """Compare three decoded images side by side."""
    # Convert to PIL Images
    pil1 = visualize_decoded(dec1)
    pil2 = visualize_decoded(dec2)
    pil3 = visualize_decoded(dec3)
    # Convert to numpy arrays
    arr1 = np.asarray(pil1)
    arr2 = np.asarray(pil2)
    arr3 = np.asarray(pil3)
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].imshow(arr1)
    axes[0].axis('off')
    axes[0].set_title(name1)
    axes[1].imshow(arr2)
    axes[1].axis('off')
    axes[1].set_title(name2)
    axes[2].imshow(arr3)
    axes[2].axis('off')
    axes[2].set_title(name3)
    plt.tight_layout()
    plt.show()


def plot_single_state_pixel(
    states, pixel_x, pixel_y, channel=0
):
    """
    Plot the evolution of a single (channel, y, x) pixel across all states.

    Args:
      states      : list of torch.Tensor, each of shape [1, C, H, W]
      pixel_x     : the W coordinate (0 <= x < width)
      pixel_y     : the H coordinate (0 <= y < height)
      channel     : which channel to sample (default 0)
    """
    n_steps = len(states)
    print(f"Total number of steps: {n_steps}")

    steps = list(range(n_steps))
    values = []

    for i, state in enumerate(states):
        # state is [1, C, H, W]
        t = state.squeeze(0)                  # → [C, H, W]
        if t.dtype == torch.bfloat16:
            t = t.to(torch.float32)
        val = t[channel, pixel_y, pixel_x].item()
        values.append(val)

    # plot
    plt.figure(figsize=(12, 4))
    plt.plot(steps, values, '-o', color='tab:blue')
    plt.title(f"Pixel Evolution @ (x={pixel_x}, y={pixel_y}, ch={channel})")
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # stats
    print("\nStatistics:")
    print(f"  Initial : {values[0]:.4f}")
    print(f"  Final   : {values[-1]:.4f}")
    print(f"  Min/Max : {min(values):.4f} / {max(values):.4f}")
    print(f"  Delta   : {values[-1] - values[0]:.4f}")


def plot_multiple_state_pixels(
    states,
    pixel_positions,
    channel: int = 0
):
    """
    Plot the evolution of multiple pixel locations across a list of latent states,
    automatically skipping any (x,y) that lies outside the [0..W-1]×[0..H-1] grid.
    """
    # figure out H,W from the very first state
    sample = states[0].squeeze(0)            # [C,H,W]
    C, H, W = sample.shape

    # filter valid positions
    valid = []
    for x,y in pixel_positions:
        if 0 <= x < W and 0 <= y < H:
            valid.append((x,y))
        else:
            print(f"⚠️ skipping ({x},{y}) — out of bounds (W={W}, H={H})")
    if not valid:
        raise ValueError("No valid pixel positions to plot!")

    n_steps = len(states)
    steps = list(range(n_steps))

    plt.figure(figsize=(12, 4))
    for (x, y) in valid:
        values = []
        for st in states:
            t = st.squeeze(0)               # [C,H,W]
            if t.dtype == torch.bfloat16:
                t = t.to(torch.float32)
            values.append(t[channel, y, x].item())
        plt.plot(steps, values, '-o', label=f'({x},{y})')

    plt.title(f'Pixel Evolution for Channel {channel}')
    plt.xlabel('Step')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend(title='Pixels')
    plt.tight_layout()
    plt.show()


def visualize_intermediate_results(
    states,
    decode_fn,
    viz_latent=visualize_latent,
    viz_decoded=visualize_decoded,
    figsize=(8, 3),
    channel: int = 0
) -> None:
    """
    Visualize latent states and their decoded images side by side for each step.

    Args:
        states   : list of torch.Tensor of shape [1, C, H, W]
        decode_fn: function mapping a batch tensor -> list of decoded images
        viz_latent: function to convert latent tensor to PIL Image
        viz_decoded: function to convert decoded output to PIL Image
        figsize   : figure size for each row
        channel   : which channel to visualize (default 0)
    """
    import math
    for i, state in enumerate(states):
        # prepare latent image with specified channel
        lat_img = viz_latent(state, channel)
        # decode (expects batch), take first sample
        decoded_list = decode_fn(state)
        dec_img = viz_decoded(decoded_list[0])
        # plot
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].imshow(np.asarray(lat_img))
        axes[0].axis('off')
        axes[0].set_title(f"Step {i+1} Latent (ch{channel})")
        axes[1].imshow(np.asarray(dec_img))
        axes[1].axis('off')
        axes[1].set_title(f"Step {i+1} Decoded")
        plt.tight_layout()
        plt.show()


def callback_func(step, x_t, v_i=None):
    """Simple callback function that works with both old and new signatures."""
    if v_i is not None:
        print(f"Step {step}: x_t shape={x_t.shape}, gradient shape={v_i.shape}")
        print(f"gradient norm: {torch.norm(v_i).item():.4f}")
    else:
        print(f"Step {step}: x_t shape={x_t.shape}")
