from typing import Any

import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """Load image as float32 RGB numpy array.

    Supports PNG (8/16-bit) and EXR (float32).
    """
    path_lower = image_path.lower()

    if path_lower.endswith(".exr"):
        try:
            import pyexr
            img = pyexr.read(image_path).astype(np.float32)
        except ImportError:
            try:
                import OpenImageIO as oiio
                inp = oiio.ImageInput.open(image_path)
                spec = inp.spec()
                img = np.frombuffer(inp.read_image(oiio.FLOAT), dtype=np.float32)
                img = img.reshape(spec.height, spec.width, spec.nchannels)
                inp.close()
            except ImportError:
                raise ImportError("Install pyexr or OpenImageIO to read EXR files")
    else:
        from PIL import Image
        pil_img = Image.open(image_path).convert("RGB")
        img = np.array(pil_img).astype(np.float32) / 255.0

    if img.shape[-1] == 4:
        img = img[:, :, :3]

    return img


def save_image(output_path: str, img: np.ndarray):
    """Save image as PNG (8-bit) or EXR (float32). Expects RGB input."""
    path_lower = output_path.lower()

    if path_lower.endswith(".exr"):
        import pyexr
        pyexr.write(output_path, img.astype(np.float32))
    elif path_lower.endswith(".png"):
        from PIL import Image
        if img.dtype != np.uint8:
            img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        else:
            img_u8 = img
        Image.fromarray(img_u8).save(output_path)
    else:
        from PIL import Image
        if img.dtype != np.uint8:
            img_u8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        else:
            img_u8 = img
        Image.fromarray(img_u8).save(output_path)
