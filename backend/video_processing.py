"""
video_processing.py  -  Ski Shadow Overlay (v6)
================================================
Correctifs v6 (le shadow etait invisible) :

  1. Le masque du skieur est desormais calcule dans le repere PROPRE de
     la video 2 (avant tout warp), contre son fond median PROPRE.
     On warpe ENSUITE le masque vers le repere de la video 1.
     -> plus de pollution par les bords noirs du warp.

  2. Fallback automatique : si la diff avec le fond est trop faible
     (skieur qui passe vite, lumiere qui change), on bascule sur du
     frame-differencing entre 2 frames consecutives de la video 2.

  3. Seuil de detection plus permissif (BG_DIFF_THRESH 28 -> 18) et
     filtre de surface minimum reduit (-> on garde meme les petits
     skieurs eloignes).

  4. Shadow plus visible : couleur ROUGE saturee + contour blanc 4 px.

  5. Logs : compteur de frames avec mask non-vide (`mask_ok` /
     `mask_empty`) pour diagnostiquer si jamais le shadow disparait.
"""

from pathlib import Path
import cv2
import numpy as np
import traceback
from typing import Optional, Tuple

# ─────────────────────────────────────────────────────────────────
#  PARAMETRES
# ─────────────────────────────────────────────────────────────────
MAX_REAL_DURATION_S = 40.0
OUTPUT_FPS_DEFAULT  = 30
SPEED_FACTOR        = 0.6

PROC_MAX_W          = 1280

# Alignement
KEYFRAME_STEP       = 3
ORB_FEATURES        = 2500
RANSAC_THRESH       = 5.0
MIN_INLIERS         = 20
H_SMOOTH            = 0.75

MOTION_DIFF_THRESH  = 22
MOTION_DILATE_PX    = 25

# Background median
BG_SAMPLES          = 25
BG_DIFF_THRESH      = 18         # PLUS permissif (etait 28)
MIN_AREA_RATIO      = 0.0001     # PLUS permissif (etait 0.0003)

# Frame-diff fallback
FRAMEDIFF_THRESH    = 18

# Shadow rendering — TRES visible
SHADOW_OPACITY      = 0.85
SHADOW_TINT_BGR     = (0, 0, 255)        # ROUGE pur
SHADOW_OUTLINE_BGR  = (255, 255, 255)    # contour blanc
SHADOW_OUTLINE_W    = 4


# ─────────────────────────────────────────────────────────────────
#  HELPERS RESIZE
# ─────────────────────────────────────────────────────────────────
def proc_size(w: int, h: int, max_w: int = PROC_MAX_W) -> Tuple[int, int, float]:
    if w <= max_w:
        return w, h, 1.0
    s = max_w / float(w)
    return int(round(w * s)), int(round(h * s)), s


def to_proc(frame: np.ndarray, proc_w: int, proc_h: int) -> np.ndarray:
    if frame.shape[1] == proc_w and frame.shape[0] == proc_h:
        return frame
    return cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)


# ─────────────────────────────────────────────────────────────────
#  1. BACKGROUND MEDIAN
# ─────────────────────────────────────────────────────────────────
def compute_background_median(
    cap: cv2.VideoCapture, duration_s: float, fps: float,
    proc_w: int, proc_h: int, n_samples: int = BG_SAMPLES,
) -> Optional[np.ndarray]:
    total_frames = int(min(duration_s * fps, cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    if total_frames < 5:
        return None
    frames = []
    for k in range(n_samples):
        idx = int((k + 0.5) * total_frames / n_samples)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        if not ok:
            continue
        frames.append(to_proc(f, proc_w, proc_h))
    if len(frames) < 3:
        return None
    stack = np.stack(frames, axis=0)
    bg = np.median(stack, axis=0).astype(np.uint8)
    del stack, frames
    return bg


# ─────────────────────────────────────────────────────────────────
#  2. MASQUE DE MOUVEMENT (pour ORB)
# ─────────────────────────────────────────────────────────────────
def motion_mask(prev_gray, curr_gray):
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, m = cv2.threshold(diff, MOTION_DIFF_THRESH, 255, cv2.THRESH_BINARY)
    k = np.ones((MOTION_DILATE_PX, MOTION_DILATE_PX), np.uint8)
    m = cv2.dilate(m, k, iterations=1)
    return cv2.bitwise_not(m)


# ─────────────────────────────────────────────────────────────────
#  3. HOMOGRAPHIE
# ─────────────────────────────────────────────────────────────────
def compute_frame_homography(f1, f2, mask1, mask2, orb, matcher, prev_H=None):
    g1 = cv2.equalizeHist(cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY))
    g2 = cv2.equalizeHist(cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY))
    kp1, des1 = orb.detectAndCompute(g1, mask1)
    kp2, des2 = orb.detectAndCompute(g2, mask2)
    if des1 is None or des2 is None or len(kp1) < 30 or len(kp2) < 30:
        return prev_H
    matches = matcher.match(des1, des2)
    if len(matches) < 20:
        return prev_H
    matches = sorted(matches, key=lambda x: x.distance)[:200]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    H, inliers = cv2.findHomography(pts2, pts1, cv2.RANSAC, RANSAC_THRESH)
    if H is None or inliers is None or int(inliers.sum()) < MIN_INLIERS:
        return prev_H
    if prev_H is not None:
        H = H_SMOOTH * prev_H + (1.0 - H_SMOOTH) * H
    return H


# ─────────────────────────────────────────────────────────────────
#  4. SILHOUETTE — bg-diff PRIMAIRE
# ─────────────────────────────────────────────────────────────────
def _clean_mask(m: np.ndarray, frame_area: int) -> np.ndarray:
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  k, iterations=1)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=4)
    m = cv2.dilate(m, k, iterations=1)

    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(m)
    biggest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(biggest) < frame_area * MIN_AREA_RATIO:
        return np.zeros_like(m)
    out = np.zeros_like(m)
    cv2.drawContours(out, [biggest], -1, 255, thickness=cv2.FILLED)
    return out


def extract_skier_mask_bgdiff(frame_bgr: np.ndarray, bg_bgr: np.ndarray,
                              threshold: int = BG_DIFF_THRESH) -> np.ndarray:
    if frame_bgr.shape != bg_bgr.shape:
        bg_bgr = cv2.resize(bg_bgr, (frame_bgr.shape[1], frame_bgr.shape[0]))
    diff = cv2.absdiff(frame_bgr, bg_bgr)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, m = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return _clean_mask(m, frame_bgr.shape[0] * frame_bgr.shape[1])


# ─────────────────────────────────────────────────────────────────
#  5. SILHOUETTE — fallback frame-diff
# ─────────────────────────────────────────────────────────────────
def extract_skier_mask_framediff(curr_bgr: np.ndarray, prev_bgr: np.ndarray,
                                 threshold: int = FRAMEDIFF_THRESH) -> np.ndarray:
    if prev_bgr is None:
        return np.zeros(curr_bgr.shape[:2], dtype=np.uint8)
    g1 = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(g1, g2)
    _, m = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    return _clean_mask(m, curr_bgr.shape[0] * curr_bgr.shape[1])


# ─────────────────────────────────────────────────────────────────
#  6. RENDU SHADOW
# ─────────────────────────────────────────────────────────────────
def render_shadow(base_native, mask_native, opacity=SHADOW_OPACITY,
                  tint=SHADOW_TINT_BGR, outline=SHADOW_OUTLINE_BGR,
                  outline_w=SHADOW_OUTLINE_W):
    out = base_native.copy()
    if not mask_native.any():
        return out
    color_layer = np.full_like(out, tint, dtype=np.uint8)
    soft_mask = cv2.GaussianBlur(mask_native, (9, 9), 0)
    a = (soft_mask.astype(np.float32) / 255.0) * opacity
    a3 = cv2.merge([a, a, a])
    out = (out.astype(np.float32) * (1.0 - a3) +
           color_layer.astype(np.float32) * a3).astype(np.uint8)
    contours, _ = cv2.findContours(mask_native, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, outline, outline_w, cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────
#  7. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────
def process_videos(
    video1_path, video2_path, output_path,
    max_duration_s: float = MAX_REAL_DURATION_S,
    output_fps:     int   = OUTPUT_FPS_DEFAULT,
    speed_factor:   float = SPEED_FACTOR,
    shadow_opacity: float = SHADOW_OPACITY,
    keyframe_step:  int   = KEYFRAME_STEP,
    **kwargs,
) -> Path:
    video1_path = Path(video1_path)
    video2_path = Path(video2_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap1 = cv2.VideoCapture(str(video1_path))
    cap2 = cv2.VideoCapture(str(video2_path))
    if not cap1.isOpened() or not cap2.isOpened():
        cap1.release(); cap2.release()
        raise RuntimeError("Impossible d'ouvrir une des videos.")

    try:
        fps1   = cap1.get(cv2.CAP_PROP_FPS) or 30.0
        fps2   = cap2.get(cv2.CAP_PROP_FPS) or 30.0
        total1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        total2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        if total1 < 5 or total2 < 5:
            raise RuntimeError(f"Videos trop courtes ({total1}, {total2}).")

        real_duration  = min(total1 / fps1, total2 / fps2, max_duration_s)
        out_duration   = real_duration / speed_factor
        max_frames_out = int(out_duration * output_fps)

        # Native dims (video 1)
        cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, first1 = cap1.read()
        if not ok:
            raise RuntimeError("Video 1 illisible.")
        H_NAT, W_NAT = first1.shape[:2]
        W_PROC, H_PROC, scale = proc_size(W_NAT, H_NAT, PROC_MAX_W)

        print(f"[INFO] Native {W_NAT}x{H_NAT} | Proc {W_PROC}x{H_PROC} "
              f"scale={scale:.3f}")
        print(f"[INFO] dur={real_duration:.1f}s -> out={out_duration:.1f}s "
              f"({max_frames_out} frames @ {output_fps}fps)")

        # ── Backgrounds (basse res) ──
        print("[1/3] Calcul backgrounds medians...")
        bg1 = compute_background_median(cap1, real_duration, fps1, W_PROC, H_PROC)
        bg2 = compute_background_median(cap2, real_duration, fps2, W_PROC, H_PROC)
        if bg1 is None or bg2 is None:
            raise RuntimeError("Echec calcul background.")
        print(f"    -> bg1 {bg1.shape}, bg2 {bg2.shape} OK")

        cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # ── Writer NATIVE ──
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, output_fps,
                                 (W_NAT, H_NAT))
        if not writer.isOpened():
            raise RuntimeError("Writer video KO.")

        orb     = cv2.ORB_create(nfeatures=ORB_FEATURES)
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        H_proc        = None
        prev_gray1    = None
        prev_gray2    = None
        prev_f2_proc  = None       # pour le fallback frame-diff
        n_align_ok    = 0
        n_align_fail  = 0
        n_mask_bg     = 0          # mask trouve via bg-diff
        n_mask_fd     = 0          # mask trouve via frame-diff
        n_mask_empty  = 0

        print(f"[2/3] Rendu {max_frames_out} frames keyframe={keyframe_step}")

        for i in range(max_frames_out):
            t_real = (i / output_fps) * speed_factor
            if t_real > real_duration:
                break

            idx1 = int(min(t_real * fps1, total1 - 1))
            idx2 = int(min(t_real * fps2, total2 - 1))
            cap1.set(cv2.CAP_PROP_POS_FRAMES, idx1)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, idx2)
            r1, f1_native = cap1.read()
            r2, f2_native = cap2.read()
            if not (r1 and r2):
                break

            if f2_native.shape[:2] != (H_NAT, W_NAT):
                f2_native = cv2.resize(f2_native, (W_NAT, H_NAT))

            f1_proc = to_proc(f1_native, W_PROC, H_PROC)
            f2_proc = to_proc(f2_native, W_PROC, H_PROC)
            gray1 = cv2.cvtColor(f1_proc, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(f2_proc, cv2.COLOR_BGR2GRAY)

            # ── Recalcul homographie ──
            if i % keyframe_step == 0:
                m1 = motion_mask(prev_gray1, gray1) if prev_gray1 is not None else None
                m2 = motion_mask(prev_gray2, gray2) if prev_gray2 is not None else None
                new_H = compute_frame_homography(
                    f1_proc, f2_proc, m1, m2, orb, matcher, prev_H=H_proc,
                )
                if new_H is not None:
                    H_proc = new_H
                    n_align_ok += 1
                else:
                    n_align_fail += 1

            prev_gray1 = gray1
            prev_gray2 = gray2

            # ─────────────────────────────────────────────────────
            #  >>> CHANGEMENT CLE : masque calcule en repere f2 <<<
            # ─────────────────────────────────────────────────────
            mask_in_f2 = extract_skier_mask_bgdiff(f2_proc, bg2,
                                                   threshold=BG_DIFF_THRESH)
            if mask_in_f2.any():
                n_mask_bg += 1
            else:
                # Fallback frame-diff
                mask_in_f2 = extract_skier_mask_framediff(
                    f2_proc, prev_f2_proc, threshold=FRAMEDIFF_THRESH,
                )
                if mask_in_f2.any():
                    n_mask_fd += 1
                else:
                    n_mask_empty += 1

            prev_f2_proc = f2_proc

            # Warp du MASQUE vers le repere f1
            if H_proc is not None:
                mask_aligned = cv2.warpPerspective(
                    mask_in_f2, H_proc, (W_PROC, H_PROC),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0,
                )
            else:
                mask_aligned = mask_in_f2

            # Upscale vers resolution NATIVE
            if scale != 1.0:
                mask_native = cv2.resize(
                    mask_aligned, (W_NAT, H_NAT),
                    interpolation=cv2.INTER_LINEAR,
                )
                _, mask_native = cv2.threshold(mask_native, 64, 255,
                                               cv2.THRESH_BINARY)
            else:
                mask_native = mask_aligned

            # ── Composition shadow ──
            frame_out = render_shadow(
                f1_native, mask_native, opacity=shadow_opacity,
                tint=SHADOW_TINT_BGR, outline=SHADOW_OUTLINE_BGR,
                outline_w=SHADOW_OUTLINE_W,
            )

            cv2.putText(frame_out, f"t = {t_real:5.2f}s",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (255, 255, 255), 2, cv2.LINE_AA)

            writer.write(frame_out)

            if i % 60 == 0 and i > 0:
                pct = int(100 * i / max_frames_out)
                print(f"  -> {pct}% ({i}/{max_frames_out}) "
                      f"align ok={n_align_ok}/fail={n_align_fail} "
                      f"| mask bg={n_mask_bg} fd={n_mask_fd} "
                      f"empty={n_mask_empty}")

        writer.release()
        print(f"[3/3] OK -> {output_path}")
        print(f"       align: ok={n_align_ok}, fail={n_align_fail}")
        print(f"       mask:  bg={n_mask_bg}, framediff={n_mask_fd}, "
              f"empty={n_mask_empty}")
        return output_path

    except Exception as e:
        print(f"[ERROR] process_videos: {type(e).__name__}: {e}")
        traceback.print_exc()
        raise
    finally:
        cap1.release()
        cap2.release()
