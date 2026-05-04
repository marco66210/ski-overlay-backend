"""
video_processing.py  –  Ski Shadow Overlay (v2)
================================================
Stratégie corrigée :

  1. ALIGNEMENT robuste des 2 vidéos via ORB feature matching multi-frames
     (détection automatique des points fixes : sapins, neige, portes…).
  2. Calcul d'une homographie globale → vidéo 2 est warpée pour épouser
     PARFAITEMENT le repère de la vidéo 1.
  3. Extraction de la silhouette du skieur 2 (soustraction de fond).
  4. Superposition uniquement de cette silhouette, teintée en bleu/cyan,
     sur la vidéo 1 → on voit le décor net + les 2 skieurs.
  5. Crop dynamique centré sur les 2 skieurs + ralenti.
"""

from pathlib import Path
import cv2
import numpy as np
from typing import Optional

# ─────────────────────────────────────────────────────────────────
#  PARAMÈTRES (modifiables via process_videos)
# ─────────────────────────────────────────────────────────────────
MAX_REAL_DURATION_S = 40.0
OUTPUT_WIDTH        = 1280
OUTPUT_HEIGHT       = 720
OUTPUT_FPS          = 30
SPEED_FACTOR        = 0.6      # < 1 = ralenti

CROP_REL_W          = 0.65
CROP_REL_H          = 0.65

# Alignement
N_ALIGN_SAMPLES     = 12        # nb de frames échantillonnées pour ORB
ORB_FEATURES        = 4000
RANSAC_THRESH       = 4.0

# Shadow
SHADOW_OPACITY      = 0.65      # 0 → invisible, 1 → opaque
SHADOW_TINT_BGR     = (255, 120, 0)   # cyan-bleu (BGR)


# ─────────────────────────────────────────────────────────────────
#  1. ALIGNEMENT VIA ORB FEATURE MATCHING
# ─────────────────────────────────────────────────────────────────
def compute_alignment_homography(
    cap1: cv2.VideoCapture,
    cap2: cv2.VideoCapture,
    n_samples: int = N_ALIGN_SAMPLES,
) -> Optional[np.ndarray]:
    """
    Calcule l'homographie H telle que   pt_video1 ≈ H @ pt_video2
    en agrégeant les keypoints ORB sur n_samples frames réparties dans
    chaque vidéo. Les points "skieur" finissent éliminés par RANSAC.
    """
    total1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    total2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
    if total1 < 2 or total2 < 2:
        return None

    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    bf  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    pts1_all, pts2_all = [], []

    for k in range(n_samples):
        ratio = (k + 1) / (n_samples + 1)
        idx1 = int(total1 * ratio)
        idx2 = int(total2 * ratio)

        cap1.set(cv2.CAP_PROP_POS_FRAMES, idx1)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, idx2)
        r1, f1 = cap1.read()
        r2, f2 = cap2.read()
        if not (r1 and r2):
            continue

        g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)
        # Égalisation d'histogramme : gomme les écarts de lumière
        g1 = cv2.equalizeHist(g1)
        g2 = cv2.equalizeHist(g2)

        kp1, des1 = orb.detectAndCompute(g1, None)
        kp2, des2 = orb.detectAndCompute(g2, None)
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            continue

        matches = bf.match(des1, des2)
        if not matches:
            continue
        matches = sorted(matches, key=lambda m: m.distance)[:300]

        for m in matches:
            pts1_all.append(kp1[m.queryIdx].pt)
            pts2_all.append(kp2[m.trainIdx].pt)

    if len(pts1_all) < 20:
        print(f"[ALIGN] Pas assez de matches ({len(pts1_all)}). Abandon.")
        return None

    pts1 = np.array(pts1_all, dtype=np.float32)
    pts2 = np.array(pts2_all, dtype=np.float32)

    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, RANSAC_THRESH)
    if H is None:
        print("[ALIGN] findHomography a echoue.")
        return None

    inliers = int(mask.sum()) if mask is not None else 0
    print(f"[ALIGN] {len(pts1)} matches -> {inliers} inliers RANSAC. OK.")
    return H


# ─────────────────────────────────────────────────────────────────
#  2. EXTRACTION DE LA SILHOUETTE DU SKIEUR
# ─────────────────────────────────────────────────────────────────
def extract_skier_mask(bg_subtractor, frame_bgr: np.ndarray) -> np.ndarray:
    """
    Retourne un masque binaire (uint8, 0/255) de la silhouette du skieur
    (le plus gros blob en mouvement), à la même résolution que frame_bgr.
    """
    fgmask = bg_subtractor.apply(frame_bgr)
    _, m = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=3)
    m = cv2.dilate(m, k, iterations=2)

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(m)

    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < (frame_bgr.shape[0] * frame_bgr.shape[1]) * 0.0005:
        return np.zeros_like(m)

    clean = np.zeros_like(m)
    cv2.drawContours(clean, [biggest], -1, 255, thickness=cv2.FILLED)
    clean = cv2.GaussianBlur(clean, (15, 15), 0)
    return clean


def skier_centroid(mask: np.ndarray) -> Optional[tuple]:
    """Retourne (cx, cy) du centroïde du masque, ou None."""
    M = cv2.moments(mask)
    if M["m00"] < 1.0:
        return None
    return M["m10"] / M["m00"], M["m01"] / M["m00"]


# ─────────────────────────────────────────────────────────────────
#  3. SUPERPOSITION SHADOW
# ─────────────────────────────────────────────────────────────────
def apply_shadow(
    base_bgr: np.ndarray,
    ghost_bgr: np.ndarray,
    ghost_mask: np.ndarray,
    opacity: float = SHADOW_OPACITY,
    tint: tuple = SHADOW_TINT_BGR,
) -> np.ndarray:
    """
    Superpose la silhouette `ghost` (extraite via `ghost_mask`) sur `base`,
    en la teintant pour la distinguer du skieur principal.
    """
    tint_layer = np.full_like(ghost_bgr, tint, dtype=np.uint8)
    tinted = cv2.addWeighted(ghost_bgr, 0.35, tint_layer, 0.65, 0)

    m = (ghost_mask.astype(np.float32) / 255.0) * opacity
    m3 = cv2.merge([m, m, m])

    out = base_bgr.astype(np.float32) * (1.0 - m3) + \
          tinted.astype(np.float32)   * m3
    return out.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────
#  4. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────
def process_videos(
    video1_path,
    video2_path,
    output_path,
    max_duration_s: float = MAX_REAL_DURATION_S,
    output_width:   int   = OUTPUT_WIDTH,
    output_height:  int   = OUTPUT_HEIGHT,
    output_fps:     int   = OUTPUT_FPS,
    speed_factor:   float = SPEED_FACTOR,
    shadow_opacity: float = SHADOW_OPACITY,
) -> Path:
    """
    Genere une video "shadow" : decor de la video 1 + silhouette du skieur 2
    teintee en bleu, alignee precisement via homographie ORB.
    """
    video1_path = Path(video1_path)
    video2_path = Path(video2_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap1 = cv2.VideoCapture(str(video1_path))
    cap2 = cv2.VideoCapture(str(video2_path))
    if not cap1.isOpened() or not cap2.isOpened():
        cap1.release(); cap2.release()
        raise RuntimeError("Impossible d'ouvrir une des videos.")

    fps1 = cap1.get(cv2.CAP_PROP_FPS) or 30.0
    fps2 = cap2.get(cv2.CAP_PROP_FPS) or 30.0
    total1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    total2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

    real_duration  = min(total1 / fps1, total2 / fps2, max_duration_s)
    out_duration   = real_duration / speed_factor
    max_frames_out = int(out_duration * output_fps)

    # Dimensions de reference (video 1)
    cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, first1 = cap1.read()
    if not ok:
        cap1.release(); cap2.release()
        raise RuntimeError("Video 1 illisible.")
    H_REF, W_REF = first1.shape[:2]

    # 1) HOMOGRAPHIE
    print("[1/3] Calcul de l'homographie ORB...")
    H = compute_alignment_homography(cap1, cap2)
    if H is None:
        print("[WARN] Pas d'homographie -> fallback resize simple.")

    cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 2) Crop & writer
    crop_w = max(64, min(W_REF, int(W_REF * CROP_REL_W)))
    crop_h = max(64, min(H_REF, int(H_REF * CROP_REL_H)))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, output_fps,
                             (output_width, output_height))
    if not writer.isOpened():
        cap1.release(); cap2.release()
        raise RuntimeError("Writer video KO.")

    bg1 = cv2.createBackgroundSubtractorMOG2(history=400,
                                             varThreshold=20,
                                             detectShadows=False)
    bg2 = cv2.createBackgroundSubtractorMOG2(history=400,
                                             varThreshold=20,
                                             detectShadows=False)

    cx_smooth, cy_smooth = W_REF / 2.0, H_REF / 2.0
    SMOOTH = 0.85

    print(f"[2/3] Rendu de {max_frames_out} frames "
          f"({out_duration:.1f}s @ {output_fps}fps)...")

    for i in range(max_frames_out):
        t_real = (i / output_fps) * speed_factor
        if t_real > real_duration:
            break

        idx1 = int(min(t_real * fps1, total1 - 1))
        idx2 = int(min(t_real * fps2, total2 - 1))

        cap1.set(cv2.CAP_PROP_POS_FRAMES, idx1)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, idx2)
        r1, f1 = cap1.read()
        r2, f2 = cap2.read()
        if not (r1 and r2):
            break

        # Aligner f2 sur le repere de f1
        if H is not None:
            f2a = cv2.warpPerspective(
                f2, H, (W_REF, H_REF),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        else:
            f2a = cv2.resize(f2, (W_REF, H_REF))

        # Masques skieurs
        mask1 = extract_skier_mask(bg1, f1)
        mask2 = extract_skier_mask(bg2, f2a)

        c1 = skier_centroid(mask1)
        c2 = skier_centroid(mask2)
        if c1 and c2:
            cx = (c1[0] + c2[0]) / 2
            cy = (c1[1] + c2[1]) / 2
        elif c1:
            cx, cy = c1
        elif c2:
            cx, cy = c2
        else:
            cx, cy = cx_smooth, cy_smooth

        cx_smooth = SMOOTH * cx_smooth + (1 - SMOOTH) * cx
        cy_smooth = SMOOTH * cy_smooth + (1 - SMOOTH) * cy

        # Crop
        x0 = int(cx_smooth - crop_w / 2)
        y0 = int(cy_smooth - crop_h / 2)
        x0 = max(0, min(W_REF - crop_w, x0))
        y0 = max(0, min(H_REF - crop_h, y0))
        x1 = x0 + crop_w
        y1 = y0 + crop_h

        crop_base  = f1[y0:y1, x0:x1]
        crop_ghost = f2a[y0:y1, x0:x1]
        crop_mask  = mask2[y0:y1, x0:x1]

        if crop_base.size == 0:
            continue

        base_out  = cv2.resize(crop_base,  (output_width, output_height))
        ghost_out = cv2.resize(crop_ghost, (output_width, output_height))
        mask_out  = cv2.resize(crop_mask,  (output_width, output_height))

        # Composition finale
        frame_out = apply_shadow(base_out, ghost_out, mask_out,
                                 opacity=shadow_opacity)

        # HUD
        cv2.putText(frame_out, f"t = {t_real:5.2f}s",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_out,
                    "Skieur 1 (couleur)  /  Skieur 2 (shadow bleu)",
                    (20, output_height - 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(frame_out)

        if i % 60 == 0:
            print(f"  -> {int(100 * i / max_frames_out)}% "
                  f"(frame {i}/{max_frames_out})")

    cap1.release()
    cap2.release()
    writer.release()
    print(f"[3/3] OK -> {output_path}")
    return output_path
