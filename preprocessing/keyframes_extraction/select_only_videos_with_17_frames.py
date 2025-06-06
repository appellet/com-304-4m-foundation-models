import os
import shutil

# À adapter : dossier racine où sont tous les clips
output_root = r"C:\Users\User\Desktop\content\Keyframes"
total_K     = 17

def clean_clips_missing_pose_frames(root_dir, total_K):
    """
    Pour chaque dossier de clip dans root_dir :
      - compte les images dans pose_keyframes
      - si < total_K, supprime le dossier entier du clip
    """
    for clip_name in os.listdir(root_dir):
        clip_dir = os.path.join(root_dir, clip_name)
        if not os.path.isdir(clip_dir):
            continue

        pose_dir = os.path.join(clip_dir, "pose_keyframes")
        if not os.path.isdir(pose_dir):
            print(f"⚠️  Pas de dossier pose_keyframes pour «{clip_name}», on supprime.")
            shutil.rmtree(clip_dir)
            continue

        # Compte des images .jpg/.png
        imgs = [f for f in os.listdir(pose_dir)
                if f.lower().endswith((".jpg", ".png"))]
        if len(imgs) < total_K:
            print(f"❌ Clip «{clip_name}» supprimé : {len(imgs)} < {total_K} frames dans pose_keyframes")
            shutil.rmtree(clip_dir)
        else:
            print(f"✅ Clip «{clip_name}» conservé : {len(imgs)} frames")

if __name__ == "__main__":
    clean_clips_missing_pose_frames(output_root, total_K)
