import numpy as np
import torch
from PIL import Image
import io
import base64


def tensor_to_base64_png(tensor: torch.Tensor) -> str:
    """
    Convert a 1x28x28 or 28x28 float tensor [0, 1] to base64 data URI PNG string.
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)

    arr = (tensor.detach().cpu().numpy().clip(0.0, 1.0) * 255.0).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def base64_png_to_tensor(data_uri: str) -> torch.Tensor:
    """
    Convert a base64 data URI or raw base64 string from canvas to a 1x1x28x28 float tensor [0, 1].
    """
    if "," in data_uri:
        data_uri = data_uri.split(",", 1)[1]
    raw_bytes = base64.b64decode(data_uri)
    img = Image.open(io.BytesIO(raw_bytes)).convert("L")
    img = img.resize((28, 28), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return tensor
