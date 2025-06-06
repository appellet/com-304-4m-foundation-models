

import os
import cv2
import numpy as np

def extract_segmented_keyframes(video_path, output_dir, max_duration=4.0, total_K=17):
    """
    Extrait jusqu’à total_K key-frames uniques par vidéo, réparties sur 4 segments d’1s :
      - 0→1s  :  5 frames
      - 1→2s  :  4 frames
      - 2→3s  :  4 frames
      - 3→4s  :  4 frames
    Si la vidéo est plus courte, on tronque à max_duration et on adapte,
    puis on complète avec les meilleurs scores globaux sans dupliquer les frames.
    Sauvegarde les images dans output_dir.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # 1) Charger et tronquer à max_duration
    frames = []
    max_frames = int(round(fps * max_duration))
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    n = len(frames)
    if n == 0:
        print(f"[!] Vidéo vide ou illisible : {video_path}")
        return
    if n < total_K:
        total_K = n

    # 2) Score global
    global_scores = []
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    for i in range(1, n):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        global_scores.append((i, float(np.sum(diff))))
        prev_gray = gray
    global_scores.sort(key=lambda x: x[1], reverse=True)

    # 3) Segments et quotas
    segments = [(0,1,5),(1,2,4),(2,3,4),(3,4,4)]
    selected = []

    for start_s, end_s, quota in segments:
        if len(selected) >= total_K:
            break
        start_f = min(int(round(start_s * fps)), n-1)
        end_f   = min(int(round(end_s * fps)), n)
        if end_f <= start_f:
            continue

        seg_scores = []
        prev_gray = cv2.cvtColor(frames[start_f], cv2.COLOR_BGR2GRAY)
        for i in range(start_f+1, end_f):
            gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            diff = cv2.absdiff(gray, prev_gray)
            seg_scores.append((i, float(np.sum(diff))))
            prev_gray = gray

        seg_scores.sort(key=lambda x: x[1], reverse=True)
        candidates = [idx for idx,_ in seg_scores[:quota]]
        if len(candidates) < quota and start_f not in candidates:
            candidates.insert(0, start_f)
        candidates = candidates[:quota]

        for idx in candidates:
            if len(selected) >= total_K:
                break
            if idx not in selected:
                selected.append(idx)

    # 4) Compléter avec global_scores
    for idx,_ in global_scores:
        if len(selected) >= total_K:
            break
        if idx not in selected:
            selected.append(idx)

    selected.sort()

    # 5) Sauvegarde
    os.makedirs(output_dir, exist_ok=True)
    for rank, idx in enumerate(selected, start=1):
        out_path = os.path.join(output_dir, f"keyframe_{rank:02d}_idx{idx}.jpg")
        cv2.imwrite(out_path, frames[idx])
    print(f"[+] {len(selected)} key-frames extraites pour {os.path.basename(video_path)}")

# ==== Configuration des dossiers ====
# Chemins à adapter
rgb_input_dir  = r"C:\Users\User\Desktop\content\raw_videos"
pose_input_dir = r"C:\Users\User\Desktop\content\openpose_output\video"
output_root     = r"C:\Users\User\Desktop\content\Keyframes"


# Paramètres
max_duration = 4.0   # secondes
total_K      = 17    # keyframes à extraire

# Création du dossier de sortie
os.makedirs(output_root, exist_ok=True)

# Parcours des vidéos RGB avec correspondance dans pose
for fname in os.listdir(rgb_input_dir):
    if not fname.lower().endswith('.mp4'):
        continue
    clip_name, _ = os.path.splitext(fname)
    rgb_path  = os.path.join(rgb_input_dir, fname)
    pose_path = os.path.join(pose_input_dir, fname)  # même nom de fichier

    # Vérification existence
    if not os.path.isfile(pose_path):
        print(f"⚠️ Clip pose introuvable pour {clip_name}, on passe.")
        continue

    # Dossiers de sortie pour ce clip
    clip_out_dir = os.path.join(output_root, clip_name)
    rgb_out  = os.path.join(clip_out_dir, 'rgb_keyframes')
    pose_out = os.path.join(clip_out_dir, 'pose_keyframes')

    # Extraction pour RGB puis pour Pose
    extract_segmented_keyframes(rgb_path, rgb_out, max_duration, total_K)
    extract_segmented_keyframes(pose_path, pose_out, max_duration, total_K)

print("✅ Extraction terminée pour tous les clips.")


