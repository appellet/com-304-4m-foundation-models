#!/usr/bin/env python3
import os
import json
import argparse
import time

import numpy as np
import torch
from PIL import Image
from scripts.inference_evaluate import load_model_from_config

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prétokenisation VidTok par vidéo avec découpage en chunks"
    )
    parser.add_argument("--keyframes_root", type=str, required=True,
                        help="Chemin vers dataset/keyframes (dossiers par vidéo)")
    parser.add_argument("--out_root",       type=str, required=True,
                        help="Racine où créer train/val/test/{rgb_tok,pose_tok,caption}")
    parser.add_argument("--cfg",            type=str, required=True,
                        help="Config YAML VidTok")
    parser.add_argument("--ckpt",           type=str, required=True,
                        help="Checkpoint .ckpt VidTok")
    parser.add_argument("--splits",         type=str, nargs="+",
                        default=["train","val","test"],
                        help="Splits à traiter")
    parser.add_argument("--num_frames",     type=int, default=17,
                        help="Nb de key‐frames à prendre par clip")
    parser.add_argument("--chunk_size",     type=int, default=8,
                        help="Taille des chunks temporels (frames) pour le modèle")
    return parser.parse_args()

def make_tensor(frames, device):
    """
    frames: liste de PIL.Image length T
    retourne torch.Tensor float16 shape (1,3,T,H,W) sur device
    """
    arrs = []
    for img in frames:
        a = np.array(img)
        t = torch.from_numpy(a).permute(2,0,1).float()  # (3,H,W)
        t = (t/127.5) - 1.0
        arrs.append(t)
    # (3, T, H, W) puis add batch dim
    return torch.stack(arrs, dim=1).unsqueeze(0).to(device).half()

def forward_chunks(model, clip_tensor, chunk_size):
    """
    clip_tensor: torch.Tensor shape (1,3,T,H,W)
    découpe en chunks temporels, forward, puis reconcat
    retourne tensor idx shape (T', H', W')
    """
    _, _, T, _, _ = clip_tensor.shape
    pieces = []
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        sub = clip_tensor[:, :, start:end]  # (1,3,<=chunk, H, W)
        with torch.no_grad(), torch.autocast("cuda", torch.float16):
            tokens, _, _ = model(sub)
        # one‐hot → argmax si besoin
        if tokens.dim() == 5:
            idx = tokens.argmax(dim=1)[0]  # (T', H', W')
        else:
            idx = tokens[0]                # (T', H', W')
        pieces.append(idx)
    # concat sur dim time (0)
    return torch.cat(pieces, dim=0)  # (T', H', W')

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model_from_config(args.cfg, args.ckpt)
    model  = model.to(device).eval().half()

    for split in args.splits:
        print(f"\n=== Split: {split} ===")
        split_out = os.path.join(args.out_root, split)
        os.makedirs(split_out, exist_ok=True)
        rgb_dir_out  = os.path.join(split_out, "rgb_tok");  os.makedirs(rgb_dir_out, exist_ok=True)
        pose_dir_out = os.path.join(split_out, "pose_tok"); os.makedirs(pose_dir_out, exist_ok=True)
        cap_dir_out  = os.path.join(split_out, "caption");   os.makedirs(cap_dir_out, exist_ok=True)

        for vid in sorted(os.listdir(args.keyframes_root)):
            vid_dir = os.path.join(args.keyframes_root, vid)
            if not os.path.isdir(vid_dir):
                continue

            # rgb & pose keyframes
            rgb_dir  = os.path.join(vid_dir, "rgb_keyframes")
            pose_dir = os.path.join(vid_dir, "pose_keyframes")
            if not os.path.isdir(rgb_dir) or not os.path.isdir(pose_dir):
                print(f" ⚠️ Skip {vid}: pas de dossier rgb_keyframes/pose_keyframes")
                continue

            # lister et tronquer
            rgb_files  = sorted([f for f in os.listdir(rgb_dir)  if f.lower().endswith((".png",".jpg"))])[:args.num_frames]
            pose_files = sorted([f for f in os.listdir(pose_dir) if f.lower().endswith((".png",".jpg"))])[:args.num_frames]
            if len(rgb_files) < args.num_frames or len(pose_files) < args.num_frames:
                print(f" ⚠️ Skip {vid}: trop peu d'images ({len(rgb_files)}/{len(pose_files)})")
                continue

            # préparer tensors
            rgb_imgs  = [Image.open(os.path.join(rgb_dir,f)).convert("RGB")  for f in rgb_files]
            pose_imgs = [Image.open(os.path.join(pose_dir,f)).convert("RGB") for f in pose_files]
            rgb_tensor  = make_tensor(rgb_imgs,  device)  # (1,3,T,H,W)
            pose_tensor = make_tensor(pose_imgs, device)  # (1,3,T,H,W)

            # découpage + forward
            t0 = time.time()
            rgb_idx  = forward_chunks(model, rgb_tensor,  args.chunk_size).cpu().numpy().astype(np.int64)  # (T',H',W')
            pose_idx = forward_chunks(model, pose_tensor, args.chunk_size).cpu().numpy().astype(np.int64)
            dt = time.time() - t0

            # sauvegarde
            np.save(os.path.join(rgb_dir_out,  f"{vid}.npy"), rgb_idx)
            np.save(os.path.join(pose_dir_out, f"{vid}.npy"), pose_idx)

            txt = os.path.join(vid_dir, "caption.txt")
            if os.path.exists(txt):
                with open(txt) as f:
                    caps = [l.strip() for l in f if l.strip()]
            else:
                caps = [""]
            with open(os.path.join(cap_dir_out, f"{vid}.json"), "w") as jf:
                json.dump(caps, jf, ensure_ascii=False)

            print(f" • {vid} → tokenisé en {dt:.1f}s, shapes rgb {rgb_idx.shape}, pose {pose_idx.shape}")

    print("\n[✓] Prétokenisation terminée.")

if __name__ == "__main__":
    main()
