"""Shared, opt-in image resizing for training and policy serving.

The default keeps the existing torchvision path. The OpenCV path uses only
classical Lanczos interpolation and treats arrays as RGB without channel swaps.
"""

import numpy as np


def _image_size(size):
    if len(size) != 2 or any(int(value) != value or value <= 0 for value in size):
        raise ValueError(f"Expected positive integer (height, width), got {size}")
    return tuple(int(value) for value in size)


def resize_rgb_uint8(image, size):
    """Resize HWC RGB bytes using cv2.INTER_LANCZOS4; size is (height, width)."""
    import cv2

    height, width = _image_size(size)
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError("OpenCV resize expects an HWC uint8 RGB image")
    # uint8 output saturates Lanczos ringing to [0, 255]. No AI or color conversion.
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LANCZOS4)


class OpenCVResize:
    """LeRobot transform for CPU CHW/TCHW uint8 or float [0, 1] tensors.

    Normalize float PNG decodes to bytes before interpolation, matching the
    uint8 video / policy inputs. Restore the input dtype and range afterwards.
    """

    def __init__(self, size):
        self.size = _image_size(size)

    def __call__(self, image):
        import torch

        if not isinstance(image, torch.Tensor) or image.ndim not in (3, 4):
            raise ValueError("OpenCVResize expects a CHW or TCHW torch tensor")
        if image.device.type != "cpu" or image.shape[-3] != 3:
            raise ValueError("OpenCVResize expects CPU RGB tensors")
        is_float = image.is_floating_point()
        if image.dtype != torch.uint8 and not is_float:
            raise ValueError("OpenCVResize expects uint8 or float [0, 1] inputs")
        if is_float:
            if not bool(torch.isfinite(image).all()) or image.min() < 0 or image.max() > 1:
                raise ValueError("Float resize input must be in [0, 1]; use uint8 for [0, 255]")
            pixels = (image.float() * 255).round().clamp(0, 255).to(torch.uint8)
        else:
            pixels = image
        batched = pixels.unsqueeze(0) if image.ndim == 3 else pixels
        frames = batched.permute(0, 2, 3, 1).contiguous().numpy()
        output = np.stack([resize_rgb_uint8(frame, self.size) for frame in frames])
        output = torch.from_numpy(output).permute(0, 3, 1, 2).contiguous()
        if is_float:
            output = output.to(dtype=image.dtype) / 255
        return output[0] if image.ndim == 3 else output


def build_image_resize(size, mode="torchvision_bilinear"):
    """Construct the same resize transform for dataset and inference inputs."""
    if mode == "opencv_lanczos4":
        return OpenCVResize(size)
    if mode == "torchvision_bilinear":
        from torchvision.transforms.v2 import Resize

        return Resize(size)
    raise ValueError(f"Unknown image_resize_mode: {mode!r}")


def fixed_image_grid(size, patch_size, merge_size):
    height, width = _image_size(size)
    factor = int(patch_size) * int(merge_size)
    if factor <= 0 or height % factor or width % factor:
        raise ValueError(f"Image size {size} must be divisible by patch_size * merge_size ({factor})")
    return np.array([[1, height // int(patch_size), width // int(patch_size)]], dtype=np.int64)


def process_image_at_fixed_size(image_processor, image, size):
    """Bypass a processor's min/max resize and verify the actual patch grid."""
    if image_processor is None:
        raise ValueError("preserve_image_resolution requires a Qwen image processor")
    expected = fixed_image_grid(size, image_processor.patch_size, image_processor.merge_size)
    if tuple(image.shape) != (3, *tuple(size)):
        raise ValueError(f"Expected RGB image (3, {size[0]}, {size[1]}), got {tuple(image.shape)}")
    processed = image_processor(image, do_resize=False)
    grid = processed.get("image_grid_thw")
    if grid is None:
        raise ValueError("Image processor did not return image_grid_thw")
    if hasattr(grid, "detach"):
        grid = grid.detach().cpu().numpy()
    actual = np.asarray(grid).reshape(-1, 3)
    if not np.array_equal(actual, expected):
        raise ValueError(f"Image processor changed the token grid: expected {expected.tolist()}, got {actual.tolist()}")
    return processed
