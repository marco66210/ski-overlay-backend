# backend/video_processing.py
from pathlib import Path
import subprocess
import shlex

import cv2
import numpy as np

# Limite vidéo
MAX_DURATION_DEFAULT = 40.0  # secondes
# Nombre max de frames utilisées pour ANALYSER (pas pour le rendu)
MAX_ANALYSIS_FRAMES = 80
# Nombre de segments temporels pour les homographies
NUM_SEGMENTS = 4


def _sample_frames(cap, max_duration_s=MAX_DURATION_DEFAULT, max_frames=MAX_ANALYSIS_FRAMES):
    """
    Échantillonne au plus `max_frames` frames sur la durée max `max_duration_s`.
    Retourne une liste de tuples (frame_index, time_seconds).
    """
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total_frames <= 0:
        return []

    total_duration = total_frames / fps
    effective_duration = min(max_duration_s, total_duration)

    if effective_duration <= 0 or max_frames <= 0:
        return []

    step_t = effective_duration / max_frames
    frames = []

    for i in range(max_frames):
        t = i * step_t
        if t > effective_duration:
            break
        idx = int(min(t * fps, total_frames - 1))
        frames.append((idx, t))

    return frames


def _read_frame(cap, index: int):
    """
    Lit une frame à un index donné.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, frame = cap.read()
    return frame if ret else None


def _compute_segmented_homographies(
    video_path_ref: Path,
    video_path_to_align: Path,
    num_segments: int = NUM_SEGMENTS,
    max_duration_s: float = MAX_DURATION_DEFAULT,
):
    """
    Calcule des homographies moyennes par segments de temps.
    Retourne (liste_H_par_segment, H_globale).
    """
    cap_ref = cv2.VideoCapture(str(video_path_ref))
    cap_al = cv2.VideoCapture(str(video_path_to_align))

    if not cap_ref.isOpened() or not cap_al.isOpened():
        cap_ref.release()
        cap_al.release()
        identity = np.eye(3)
        return [identity for _ in range(num_segments)], identity

    samples_ref = _sample_frames(cap_ref, max_duration_s=max_duration_s)
    samples_al = _sample_frames(cap_al, max_duration_s=max_duration_s)

    orb = cv2.ORB_create(300)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    seg_homos = [[] for _ in range(num_segments)]
    all_homos = []

    min_len = min(len(samples_ref), len(samples_al))

    for i in range(min_len):
        idx_ref, t_ref = samples_ref[i]
        idx_al, t_al = samples_al[i]

        fr1 = _read_frame(cap_ref, idx_ref)
        fr2 = _read_frame(cap_al, idx_al)
        if fr1 is None or fr2 is None:
            continue

        # downscale pour la détection de features (rapide)
        target_size = (640, 360)
        fr1s = cv2.resize(fr1, target_size)
        fr2s = cv2.resize(fr2, target_size)

        kp1, d1 = orb.detectAndCompute(fr1s, None)
        kp2, d2 = orb.detectAndCompute(fr2s, None)
        if d1 is None or d2 is None:
            continue

        matches = bf.match(d1, d2)
        if len(matches) < 12:
            continue

        # garder seulement les meilleurs matches
        matches = sorted(matches, key=lambda m: m.distance)[:60]

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)
        if H is None:
            continue

        all_homos.append(H)

        # Attribution à un segment temporel
        seg_idx = int((t_ref / max_duration_s) * num_segments)
        if seg_idx >= num_segments:
            seg_idx = num_segments - 1
        seg_homos[seg_idx].append(H)

    cap_ref.release()
    cap_al.release()

    if not all_homos:
        identity = np.eye(3)
        return [identity for _ in range(num_segments)], identity

    # Homographie globale moyenne
    H_global = np.mean(np.stack(all_homos), axis=0)
    if H_global[2, 2] != 0:
        H_global /= H_global[2, 2]

    # Homographies par segment
    seg_H_final = []
    for h_list in seg_homos:
        if not h_list:
            seg_H_final.append(H_global)
        else:
            Hm = np.mean(np.stack(h_list), axis=0)
            if Hm[2, 2] != 0:
                Hm /= Hm[2, 2]
            seg_H_final.append(Hm)

    return seg_H_final, H_global


def _encode_with_ffmpeg(tmp_path: Path, final_path: Path, fps: int):
    """
    Encode le fichier temporaire (MJPEG AVI) en MP4 H.264 via ffmpeg.
    Si ffmpeg n'est pas dispo, on garde le fichier tmp tel quel.
    """
    cmd = (
        f'ffmpeg -y -loglevel error '
        f'-i "{tmp_path}" '
        f'-c:v libx264 -preset veryfast -crf 23 '
        f'-r {fps} '
        f'"{final_path}"'
    )

    try:
        subprocess.run(shlex.split(cmd), check=True)
        # Si tout s'est bien passé, on supprime le tmp
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
    except FileNotFoundError:
        # ffmpeg pas installé -> fallback : on renomme juste
        tmp_path.replace(final_path)
    except subprocess.CalledProcessError:
        # erreur ffmpeg -> fallback
        if not final_path.exists():
            tmp_path.replace(final_path)


def process_videos(
    video1_path: Path,
    video2_path: Path,
    output_path: Path,
    max_duration_s: float = MAX_DURATION_DEFAULT,
    output_width: int = 1280,
    output_height: int = 720,
    output_fps: int = 30,
):
    """
    Crée une vidéo superposée à partir de deux vidéos :
      - video1 = référence
      - video2 = vidéo à aligner

    Sortie : MP4 720p / 30 fps (par défaut).
    """
    video1_path = Path(video1_path)
    video2_path = Path(video2_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    max_duration_s = float(max_duration_s)

    cap1 = cv2.VideoCapture(str(video1_path))
    cap2 = cv2.VideoCapture(str(video2_path))

    if not cap1.isOpened() or not cap2.isOpened():
        cap1.release()
        cap2.release()
        raise RuntimeError("Impossible d'ouvrir une des vidéos.")

    fps1 = cap1.get(cv2.CAP_PROP_FPS) or 30.0
    fps2 = cap2.get(cv2.CAP_PROP_FPS) or 30.0

    total1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    total2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

    # Homographies par segments (plus stable qu'une seule matrice globale)
    seg_H_list, H_global = _compute_segmented_homographies(
        video1_path, video2_path, num_segments=NUM_SEGMENTS, max_duration_s=max_duration_s
    )

    max_out_frames = int(max_duration_s * output_fps)

    # Fichier vidéo temporaire (MJPEG) avant passage dans ffmpeg
    tmp_path = output_path.with_suffix(".avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(
        str(tmp_path),
        fourcc,
        output_fps,
        (output_width, output_height),
    )

    if not writer.isOpened():
        cap1.release()
        cap2.release()
        raise RuntimeError("Impossible d'ouvrir le writer vidéo.")

    for i in range(max_out_frames):
        t = i / output_fps
        if t > max_duration_s:
            break

        # Segment temporel courant
        seg_idx = int((t / max_duration_s) * NUM_SEGMENTS)
        if seg_idx >= NUM_SEGMENTS:
            seg_idx = NUM_SEGMENTS - 1
        H = seg_H_list[seg_idx]

        idx1 = int(min(t * fps1, max(total1 - 1, 0)))
        idx2 = int(min(t * fps2, max(total2 - 1, 0)))

        cap1.set(cv2.CAP_PROP_POS_FRAMES, idx1)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, idx2)

        ret1, f1 = cap1.read()
        ret2, f2 = cap2.read()
        if not ret1 or not ret2:
            break

        f1r = cv2.resize(f1, (output_width, output_height))
        f2r = cv2.resize(f2, (output_width, output_height))

        warped = cv2.warpPerspective(f2r, H, (output_width, output_height))
        blended = cv2.addWeighted(f1r, 0.5, warped, 0.5, 0.0)

        writer.write(blended)

    cap1.release()
    cap2.release()
    writer.release()

    # Encodage final via ffmpeg (avec fallback)
    _encode_with_ffmpeg(tmp_path, output_path, output_fps)

    return output_path
