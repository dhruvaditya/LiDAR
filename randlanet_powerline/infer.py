import argparse
import os

import laspy
import numpy as np
import torch

from randlanet_powerline.models.randlanet import RandLANet


def chunk_indices(total: int, chunk_size: int):
    for start in range(0, total, chunk_size):
        yield start, min(start + chunk_size, total)


def main():
    parser = argparse.ArgumentParser(description="Run inference for power-line detection on a LAS file")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_las", type=str, required=True)
    parser.add_argument("--output_las", type=str, default="predicted_powerline.las")
    parser.add_argument("--chunk_size", type=int, default=8192)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    center = ckpt.get("center", np.zeros(3, dtype=np.float32))
    scale = float(ckpt.get("scale", 1.0))

    model = RandLANet(num_classes=3)
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    las = laspy.read(args.input_las)
    xyz = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
    xyz_norm = (xyz - center) / max(scale, 1e-8)

    preds = np.zeros((xyz.shape[0],), dtype=np.uint8)

    with torch.no_grad():
        for s, e in chunk_indices(xyz_norm.shape[0], args.chunk_size):
            chunk = xyz_norm[s:e]
            if chunk.shape[0] < args.chunk_size:
                pad_count = args.chunk_size - chunk.shape[0]
                pad_idx = np.random.choice(chunk.shape[0], pad_count, replace=True)
                padded = np.concatenate([chunk, chunk[pad_idx]], axis=0)
            else:
                padded = chunk

            inp = torch.from_numpy(padded).unsqueeze(0).to(device)
            logits = model(inp)
            out = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            preds[s:e] = out[: e - s]

    out_las = laspy.LasData(las.header)
    out_las.x = las.x
    out_las.y = las.y
    out_las.z = las.z

    for dim_name in las.point_format.dimension_names:
        if dim_name in {"X", "Y", "Z"}:
            continue
        out_las[dim_name] = las[dim_name]

    out_las.classification = preds

    out_dir = os.path.dirname(args.output_las)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_las.write(args.output_las)

    pct = 100.0 * float((preds == 1).sum()) / max(len(preds), 1)
    print(f"Saved predictions to: {args.output_las}")
    print(f"Predicted power-line points: {pct:.2f}%")


if __name__ == "__main__":
    main()
