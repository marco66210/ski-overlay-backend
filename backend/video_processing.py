"""
video_processing.py  -  Ski Shadow Overlay (v4)
===============================================
Corrections demandees :
  - Garder la resolution NATIVE de la video 1 (pas de crop, pas de resize)
  - Rendre le SHADOW vraiment visible (silhouette pleine + contour blanc)
  - Extraction de silhouette FIABLE via image de fond mediane
    (plus robuste que MOG2, qui rate le skieur quand le decor a des bords
    noirs apres warp)

Pipeline :
  1. Calcul d'une image de fond MEDIANE par video (skieur median = background).
  2. Pour chaque frame :
     - Recalcul d'homographie tous les KEYFRAME_STEP (decor fixe via masque
       de mouvement).
     - Warp de la video 2 vers le repere de la video 1.
     - Silhouette skieur 2 = |frame2_warped - background2_warped| seuille.
     - Composition : decor video 1 + silhouette teintee + contour blanc.
  3. Ralenti via SPEED_FACTOR.
"""

from pathlib import Path
import cv2
import numpy as np
from typing import Optional, Tuple

# ─────────────────────────────────────────────────────────────────
#  PARAMETRES
# ─────────────────────────────────────────────────────────────────
MAX_REAL_DURATION_S = 40.0
OUTPUT_FPS          = 30
SPEED_FACTOR        = 0.6

# Alignement par frame
KEYFRAME_STEP       = 3
ORB_FEATURES        = 3500
RANSAC_THRESH       = 5.0
MIN_INLIERS         = 25
H_SMOOTH            = 0.75

# Masque de mouvement
MOTION_DIFF_THRESH  = 22
MOTION_DILATE_PX    = 25

# Background median
BG_SAMPLES          = 25         # nb frames pour calculer le fond median
BG_DIFF_THRESH      = 28         # seuil de detection du skieur

# Shadow rendering
SHADOW_OPACITY      = 0.85       # tres visible
SHADOW_TINT_BGR     = (255, 100, 0)    # cyan-bleu vif
SHADOW_OUTLINE_BGR  = (255, 255, 255)  # contour blanc
SHADOW_OUTLINE_W    = 3


# ─────────────────────────────────────────────────────────────────
#  1. BACKGROUND MEDIAN  (decor sans skieur)
# ─────────────────────────────────────────────────────────────────
def compute_background_median(
    cap: cv2.VideoCapture,
    duration_s: float,
    fps: float,
    n_samples: int = BG_SAMPLES,
) -> Optional[np.ndarray]:
    """
    Image de fond = mediane temporelle de n_samples frames reparties.
    Comme le skieur change de position a chaque frame, la mediane
    produit le decor stable sans skieur.
    """
    total_frames = int(min(duration_s * fps, cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    if total_frames < 5:
        return None

    frames = []
    for k in range(n_samples):
        idx = int((k + 0.5) * total_frames / n_samples)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if ok:
            frames.append(f)
    if len(frames) < 3:
        return None

    stack = np.stack(frames, axis=0)
    bg = np.median(stack, axis=0).astype(np.uint8)
    return bg


# ─────────────────────────────────────────────────────────────────
#  2. MASQUE DE MOUVEMENT (entre 2 frames consecutives)
# ─────────────────────────────────────────────────────────────────
def motion_mask(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """255 = decor stable, 0 = zone en mouvement (skieur)."""
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, m = cv2.threshold(diff, MOTION_DIFF_THRESH, 255, cv2.THRESH_BINARY)
    k = np.ones((MOTION_DILATE_PX, MOTION_DILATE_PX), np.uint8)
    m = cv2.dilate(m, k, iterations=1)
    return cv2.bitwise_not(m)


# ─────────────────────────────────────────────────────────────────
#  3. HOMOGRAPHIE PAR FRAME
# ─────────────────────────────────────────────────────────────────
def compute_frame_homography(
    f1_bgr, f2_bgr, mask1, mask2, orb, matcher,
    prev_H: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    g1 = cv2.equalizeHist(cv2.cvtColor(f1_bgr, cv2.COLOR_BGR2GRAY))
    g2 = cv2.equalizeHist(cv2.cvtColor(f2_bgr, cv2.COLOR_BGR2GRAY))

    kp1, des1 = orb.detectAndCompute(g1, mask1)
    kp2, des2 = orb.detectAndCompute(g2, mask2)
    if des1 is None or des2 is None or len(kp1) < 30 or len(kp2) < 30:
        return prev_H

    matches = matcher.match(des1, des2)
    if len(matches) < 20:
        return prev_H
    matches = sorted(matches, key=lambda x: x.distance)[:250]

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    H, inliers = cv2.findHomography(pts2, pts1, cv2.RANSAC, RANSAC_THRESH)
    if H is None or inliers is None or int(inliers.sum()) < MIN_INLIERS:
        return prev_H

    if prev_H is not None:
        H = H_SMOOTH * prev_H + (1.0 - H_SMOOTH) * H
    return H


# ─────────────────────────────────────────────────────────────────
#  4. EXTRACTION SILHOUETTE PAR DIFFERENCE AVEC LE FOND MEDIAN
# ─────────────────────────────────────────────────────────────────
def extract_skier_mask_from_bg(
    frame_bgr: np.ndarray,
    bg_bgr: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    threshold: int = BG_DIFF_THRESH,
) -> np.ndarray:
    """
    Silhouette du skieur = pixels qui different significativement du fond.
    valid_mask : zones valides apres warp (0 sur les bords noirs).
    """
    if frame_bgr.shape != bg_bgr.shape:
        bg_bgr = cv2.resize(bg_bgr, (frame_bgr.shape[1], frame_bgr.shape[0]))

    diff = cv2.absdiff(frame_bgr, bg_bgr)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)

    if valid_mask is not None:
        m = cv2.bitwise_and(m, valid_mask)

    # Nettoyage
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=4)
    m = cv2.dilate(m, k, iterations=1)

    # Conserver le plus gros blob
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(m)

    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < (frame_bgr.shape[0] * frame_bgr.shape[1]) * 0.0003:
        return np.zeros_like(m)

    clean = np.zeros_like(m)
    cv2.drawContours(clean, [biggest], -1, 255, thickness=cv2.FILLED)
    return clean


# ─────────────────────────────────────────────────────────────────
#  5. RENDU SHADOW (silhouette pleine + contour blanc)
# ─────────────────────────────────────────────────────────────────
def render_shadow(
    base: np.ndarray,
    mask: np.ndarray,
    opacity: float = SHADOW_OPACITY,
    tint: Tuple[int, int, int] = SHADOW_TINT_BGR,
    outline: Tuple[int, int, int] = SHADOW_OUTLINE_BGR,
    outline_w: int = SHADOW_OUTLINE_W,
) -> np.ndarray:
    """
    Affiche la silhouette du skieur 2 comme un aplat de couleur tres visible
    (pas de melange avec la texture du skieur 2 -> on voit clairement l'ombre).
    """
    out = base.copy()

    # 1) Silhouette pleine teintee
    if mask.any():
        # Calque uniforme de la couleur du shadow
        color_layer = np.full_like(out, tint, dtype=np.uint8)

        # Mask flouter legerement les bords pour rendu propre
        soft_mask = cv2.GaussianBlur(mask, (7, 7), 0)
        a = (soft_mask.astype(np.float32) / 255.0) * opacity
        a3 = cv2.merge([a, a, a])

        out = (out.astype(np.float32) * (1.0 - a3) +
               color_layer.astype(np.float32) * a3).astype(np.uint8)

        # 2) Contour blanc bien net
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, outline, outline_w, cv2.LINE_AA)

    return out


# ─────────────────────────────────────────────────────────────────
#  6. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────
def process_videos(
    video1_path,
    video2_path,
    output_path,
    max_duration_s: float = MAX_REAL_DURATION_S,
    output_fps:     int   = OUTPUT_FPS,
    speed_factor:   float = SPEED_FACTOR,
    shadow_opacity: float = SHADOW_OPACITY,
    keyframe_step:  int   = KEYFRAME_STEP,
) -> Path:
    """
    Genere une video shadow a la resolution NATIVE de la video 1.
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

    fps1   = cap1.get(cv2.CAP_PROP_FPS) or 30.0
    fps2   = cap2.get(cv2.CAP_PROP_FPS) or 30.0
    total1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
    total2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))

    real_duration  = min(total1 / fps1, total2 / fps2, max_duration_s)
    out_duration   = real_duration / speed_factor
    max_frames_out = int(out_duration * output_fps)

    # ── Resolution NATIVE video 1 ──
    cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, first1 = cap1.read()
    if not ok:
        cap1.release(); cap2.release()
        raise RuntimeError("Video 1 illisible.")
    H_REF, W_REF = first1.shape[:2]
    print(f"[INFO] Resolution de sortie native: {W_REF}x{H_REF}")

    # ── Image de fond mediane pour CHAQUE video ──
    print("[1/3] Calcul des images de fond medianes...")
    bg1 = compute_background_median(cap1, real_duration, fps1)
    bg2 = compute_background_median(cap2, real_duration, fps2)
    if bg1 is None or bg2 is None:
        cap1.release(); cap2.release()
        raise RuntimeError("Echec calcul background.")
    print("    -> backgrounds OK")

    # Reset
    cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Adapter bg2 a la taille de bg1 si necessaire
    if bg2.shape[:2] != bg1.shape[:2]:
        bg2 = cv2.resize(bg2, (W_REF, H_REF))

    # ── Writer (resolution native) ──
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, output_fps,
                             (W_REF, H_REF))
    if not writer.isOpened():
        cap1.release(); cap2.release()
        raise RuntimeError("Writer video KO.")

    orb     = cv2.ORB_create(nfeatures=ORB_FEATURES)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    H_curr     = None
    prev_gray1 = None
    prev_gray2 = None
    n_align_ok = 0
    n_align_fail = 0

    print(f"[2/3] Rendu {max_frames_out} frames "
          f"({out_duration:.1f}s @ {output_fps}fps) keyframe={keyframe_step}")

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

        # Adapter f2 a f1 si tailles differentes
        if f2.shape[:2] != (H_REF, W_REF):
            f2 = cv2.resize(f2, (W_REF, H_REF))

        gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

        # ── Recalcul homographie tous les keyframe_step ──
        if i % keyframe_step == 0:
            m1 = motion_mask(prev_gray1, gray1) if prev_gray1 is not None else None
            m2 = motion_mask(prev_gray2, gray2) if prev_gray2 is not None else None
            new_H = compute_frame_homography(
                f1, f2, m1, m2, orb, matcher, prev_H=H_curr,
            )
            if new_H is not None:
                if H_curr is None or not np.allclose(new_H, H_curr):
                    H_curr = new_H
                    n_align_ok += 1
            else:
                n_align_fail += 1

        prev_gray1 = gray1
        prev_gray2 = gray2

        # ── Warp f2 et bg2 dans le repere f1 ──
        if H_curr is not None:
            f2_aligned = cv2.warpPerspective(
                f2, H_curr, (W_REF, H_REF),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            bg2_aligned = cv2.warpPerspective(
                bg2, H_curr, (W_REF, H_REF),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            # Masque des pixels valides apres warp (1 = donnee reelle)
            ones = np.full((H_REF, W_REF), 255, dtype=np.uint8)
            valid = cv2.warpPerspective(
                ones, H_curr, (W_REF, H_REF),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        else:
            f2_aligned  = f2
            bg2_aligned = bg2
            valid       = np.full((H_REF, W_REF), 255, dtype=np.uint8)

        # ── Silhouette skieur 2 par diff avec son fond aligne ──
        skier2_mask = extract_skier_mask_from_bg(
            f2_aligned, bg2_aligned, valid_mask=valid,
            threshold=BG_DIFF_THRESH,
        )

        # ── Composition : video 1 brute + shadow visible ──
        frame_out = render_shadow(
            f1, skier2_mask, opacity=shadow_opacity,
            tint=SHADOW_TINT_BGR, outline=SHADOW_OUTLINE_BGR,
            outline_w=SHADOW_OUTLINE_W,
        )

        # HUD discret
        cv2.putText(frame_out, f"t = {t_real:5.2f}s",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(frame_out)

        if i % 60 == 0 and i > 0:
            print(f"  -> {int(100 * i / max_frames_out)}% "
                  f"(frame {i}/{max_frames_out}) "
                  f"align ok={n_align_ok} fail={n_align_fail}")

    cap1.release()
    cap2.release()
    writer.release()

    print(f"[3/3] OK -> {output_path} "
          f"(align ok={n_align_ok}, fail={n_align_fail})")
    return output_path
