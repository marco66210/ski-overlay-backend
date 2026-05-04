"""
video_processing.py  -  Ski Shadow Overlay (v3 - cameras a main levee)
======================================================================
Probleme : les videos sont prises a l'epaule -> chaque image a son
propre tremblement. Une homographie globale est donc INSUFFISANTE.

Solution :
  1. Pour chaque frame, on calcule une NOUVELLE homographie entre f1 et f2
     en utilisant les features ORB du DECOR FIXE uniquement.
  2. Les zones en mouvement (skieur) sont MASQUEES avant le matching
     pour ne pas polluer l'alignement.
  3. L'homographie est ensuite LISSEE temporellement (filtre passe-bas)
     pour eviter les sauts d'une frame a l'autre.
  4. Le skieur 2 est extrait, teinte, et superpose sur la video 1.

Optimisation : on calcule l'homographie toutes les N frames (KEYFRAME_STEP)
et on interpole entre les keyframes -> ~5x plus rapide.
"""

from pathlib import Path
import cv2
import numpy as np
from typing import Optional, Tuple

# ─────────────────────────────────────────────────────────────────
#  PARAMETRES
# ─────────────────────────────────────────────────────────────────
MAX_REAL_DURATION_S = 40.0
OUTPUT_WIDTH        = 1280
OUTPUT_HEIGHT       = 720
OUTPUT_FPS          = 30
SPEED_FACTOR        = 0.6

# Crop dynamique
CROP_REL_W          = 0.70
CROP_REL_H          = 0.70

# Alignement par frame
KEYFRAME_STEP       = 3        # recalcule l'homographie toutes les N frames
ORB_FEATURES        = 3500
RANSAC_THRESH       = 5.0
MIN_INLIERS         = 25
H_SMOOTH            = 0.75     # 0 = pas de lissage, 1 = fige

# Detection mouvement (pour masquer les zones mobiles)
MOTION_DIFF_THRESH  = 22
MOTION_DILATE_PX    = 25       # on elargit le masque autour du mouvement

# Shadow
SHADOW_OPACITY      = 0.65
SHADOW_TINT_BGR     = (255, 130, 0)   # cyan-bleu (BGR)


# ─────────────────────────────────────────────────────────────────
#  1. MASQUE DE MOUVEMENT (pour exclure le skieur du matching)
# ─────────────────────────────────────────────────────────────────
def motion_mask(prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
    """
    Masque binaire (uint8): 255 = pixel STABLE (decor), 0 = pixel MOBILE
    (skieur). On utilisera ce masque comme `mask` pour ORB pour qu'il ne
    cherche des features que dans le decor.
    """
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, m = cv2.threshold(diff, MOTION_DIFF_THRESH, 255, cv2.THRESH_BINARY)
    k = np.ones((MOTION_DILATE_PX, MOTION_DILATE_PX), np.uint8)
    m = cv2.dilate(m, k, iterations=1)
    return cv2.bitwise_not(m)   # inverse: 255 sur le decor


# ─────────────────────────────────────────────────────────────────
#  2. HOMOGRAPHIE PAR FRAME (avec masques de mouvement)
# ─────────────────────────────────────────────────────────────────
def compute_frame_homography(
    f1_bgr: np.ndarray,
    f2_bgr: np.ndarray,
    mask1: Optional[np.ndarray],
    mask2: Optional[np.ndarray],
    orb: cv2.ORB,
    matcher: cv2.BFMatcher,
    prev_H: Optional[np.ndarray] = None,
) -> Optional[np.ndarray]:
    """
    Homographie H telle que   pt_dans_f1 ~= H * pt_dans_f2
    Retourne `prev_H` en cas d'echec (continuite garantie).
    """
    g1 = cv2.cvtColor(f1_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(f2_bgr, cv2.COLOR_BGR2GRAY)
    g1 = cv2.equalizeHist(g1)
    g2 = cv2.equalizeHist(g2)

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

    # Lissage temporel (mix avec H precedente)
    if prev_H is not None:
        H = H_SMOOTH * prev_H + (1.0 - H_SMOOTH) * H

    return H


# ─────────────────────────────────────────────────────────────────
#  3. EXTRACTION SILHOUETTE SKIEUR
# ─────────────────────────────────────────────────────────────────
def extract_skier_mask(bg_subtractor, frame_bgr: np.ndarray) -> np.ndarray:
    fg = bg_subtractor.apply(frame_bgr)
    _, m = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
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


def mask_centroid(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    M = cv2.moments(mask)
    if M["m00"] < 1.0:
        return None
    return M["m10"] / M["m00"], M["m01"] / M["m00"]


# ─────────────────────────────────────────────────────────────────
#  4. SUPERPOSITION SHADOW
# ─────────────────────────────────────────────────────────────────
def apply_shadow(
    base: np.ndarray,
    ghost: np.ndarray,
    ghost_mask: np.ndarray,
    opacity: float,
    tint: Tuple[int, int, int],
) -> np.ndarray:
    tint_layer = np.full_like(ghost, tint, dtype=np.uint8)
    tinted = cv2.addWeighted(ghost, 0.35, tint_layer, 0.65, 0)

    a = (ghost_mask.astype(np.float32) / 255.0) * opacity
    a3 = cv2.merge([a, a, a])

    out = base.astype(np.float32) * (1.0 - a3) + tinted.astype(np.float32) * a3
    return out.astype(np.uint8)


# ─────────────────────────────────────────────────────────────────
#  5. PIPELINE PRINCIPAL
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
    keyframe_step:  int   = KEYFRAME_STEP,
) -> Path:
    """
    Pipeline complet pour videos a main levee :
      - Homographie recalculee toutes les `keyframe_step` frames sur le decor
      - Skieur 2 superpose en mode shadow teinte sur le scene de la video 1
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

    # Dimensions de reference (video 1)
    cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ok, first1 = cap1.read()
    if not ok:
        cap1.release(); cap2.release()
        raise RuntimeError("Video 1 illisible.")
    H_REF, W_REF = first1.shape[:2]
    cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)

    crop_w = max(64, min(W_REF, int(W_REF * CROP_REL_W)))
    crop_h = max(64, min(H_REF, int(H_REF * CROP_REL_H)))

    # Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, output_fps,
                             (output_width, output_height))
    if not writer.isOpened():
        cap1.release(); cap2.release()
        raise RuntimeError("Writer video KO.")

    # Detecteurs
    orb     = cv2.ORB_create(nfeatures=ORB_FEATURES)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    bg1 = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=20,
                                             detectShadows=False)
    bg2 = cv2.createBackgroundSubtractorMOG2(history=400, varThreshold=20,
                                             detectShadows=False)

    # Etat
    H_curr            = None    # derniere homographie connue
    prev_gray1        = None    # frame precedente video 1 (pour motion mask)
    prev_gray2        = None    # frame precedente video 2
    cx_smooth, cy_smooth = W_REF / 2.0, H_REF / 2.0
    SMOOTH_CROP       = 0.85

    n_align_ok   = 0
    n_align_fail = 0

    print(f"[INFO] Rendu {max_frames_out} frames "
          f"({out_duration:.1f}s @ {output_fps}fps) "
          f"keyframe={keyframe_step}")

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

        # Adapter f2 a la taille de f1 si different (avant tout calcul)
        if f2.shape[:2] != f1.shape[:2]:
            f2 = cv2.resize(f2, (W_REF, H_REF))

        gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

        # ── Recalcul de l'homographie tous les `keyframe_step` ──
        if i % keyframe_step == 0:
            # Masques de mouvement (= masques pour ORB : ne chercher
            # que les features stables)
            if prev_gray1 is not None:
                m1 = motion_mask(prev_gray1, gray1)
            else:
                m1 = None
            if prev_gray2 is not None:
                m2 = motion_mask(prev_gray2, gray2)
            else:
                m2 = None

            new_H = compute_frame_homography(
                f1, f2, m1, m2, orb, matcher, prev_H=H_curr
            )
            if new_H is not None and not np.array_equal(new_H, H_curr):
                H_curr = new_H
                n_align_ok += 1
            else:
                n_align_fail += 1

        prev_gray1 = gray1
        prev_gray2 = gray2

        # ── Warp f2 -> repere f1 ──
        if H_curr is not None:
            f2a = cv2.warpPerspective(
                f2, H_curr, (W_REF, H_REF),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
        else:
            f2a = f2   # pas encore d'alignement -> on prend f2 brut

        # ── Silhouettes skieurs ──
        sk_mask1 = extract_skier_mask(bg1, f1)
        sk_mask2 = extract_skier_mask(bg2, f2a)

        c1 = mask_centroid(sk_mask1)
        c2 = mask_centroid(sk_mask2)
        if c1 and c2:
            cx, cy = (c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2
        elif c1:
            cx, cy = c1
        elif c2:
            cx, cy = c2
        else:
            cx, cy = cx_smooth, cy_smooth

        cx_smooth = SMOOTH_CROP * cx_smooth + (1 - SMOOTH_CROP) * cx
        cy_smooth = SMOOTH_CROP * cy_smooth + (1 - SMOOTH_CROP) * cy

        # ── Crop dynamique ──
        x0 = int(cx_smooth - crop_w / 2)
        y0 = int(cy_smooth - crop_h / 2)
        x0 = max(0, min(W_REF - crop_w, x0))
        y0 = max(0, min(H_REF - crop_h, y0))
        x1 = x0 + crop_w
        y1 = y0 + crop_h

        crop_base  = f1[y0:y1, x0:x1]
        crop_ghost = f2a[y0:y1, x0:x1]
        crop_mask  = sk_mask2[y0:y1, x0:x1]

        if crop_base.size == 0:
            continue

        base_out  = cv2.resize(crop_base,  (output_width, output_height))
        ghost_out = cv2.resize(crop_ghost, (output_width, output_height))
        mask_out  = cv2.resize(crop_mask,  (output_width, output_height))

        # ── Composition shadow ──
        frame_out = apply_shadow(
            base_out, ghost_out, mask_out,
            opacity=shadow_opacity, tint=SHADOW_TINT_BGR,
        )

        # HUD
        cv2.putText(frame_out, f"t = {t_real:5.2f}s",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame_out,
                    "Video 1 (couleur)  +  Video 2 (shadow)",
                    (20, output_height - 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(frame_out)

        if i % 60 == 0 and i > 0:
            print(f"  -> {int(100 * i / max_frames_out)}% "
                  f"(frame {i}/{max_frames_out}) "
                  f"align ok={n_align_ok} fail={n_align_fail}")

    cap1.release()
    cap2.release()
    writer.release()

    print(f"[OK] {output_path}  "
          f"alignements ok={n_align_ok} / fail={n_align_fail}")
    return output_path
