from pathlib import Path
import cv2
import numpy as np

# Durée max de la partie utilisée des vidéos (en secondes, temps "réel")
MAX_REAL_DURATION_S = 40.0

# Résolution de sortie
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
OUTPUT_FPS = 30

# Facteur de vitesse (0.6 = 40% plus lent que l'original)
SPEED_FACTOR = 0.6  # la vidéo est ralentie de 40%


def _make_gate_mask(bgr_frame):
    """
    Masque pour isoler les portes de slalom :
    - drapeaux rouges / orangés
    - drapeaux bleus
    """
    hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)

    # Plages rouges / oranges
    lower_red1 = np.array([0, 80, 70], dtype=np.uint8)
    upper_red1 = np.array([15, 255, 255], dtype=np.uint8)

    lower_red2 = np.array([160, 80, 70], dtype=np.uint8)
    upper_red2 = np.array([179, 255, 255], dtype=np.uint8)

    lower_orange = np.array([10, 80, 70], dtype=np.uint8)
    upper_orange = np.array([30, 255, 255], dtype=np.uint8)

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_red = cv2.bitwise_or(mask_red, mask_orange)

    # Plage bleue
    lower_blue = np.array([90, 80, 70], dtype=np.uint8)
    upper_blue = np.array([130, 255, 255], dtype=np.uint8)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Combine rouge + bleu
    mask = cv2.bitwise_or(mask_red, mask_blue)

    # Nettoyage
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


def _find_active_gate_bbox(mask):
    """
    Trouve la "porte active" dans un masque binaire :
    - on détecte les contours
    - on filtre par taille
    - on prend la porte dont le centre est le plus proche d'une bande
      verticale en bas de l'image (60-80% de la hauteur).
    Retourne (x, y, w, h) ou None si rien de valable.
    """
    h, w = mask.shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area < (w * h) * 0.0005:  # trop petit -> bruit
            continue
        if area > (w * h) * 0.1:  # trop gros -> probablement pas une porte
            continue

        cx = x + cw / 2.0
        cy = y + ch / 2.0
        candidates.append((x, y, cw, ch, cx, cy))

    if not candidates:
        return None

    # On veut la porte dans une bande 60-80% de la hauteur (en bas/milieu)
    band_top = h * 0.55
    band_bottom = h * 0.85

    best = None
    best_score = None

    for x, y, cw, ch, cx, cy in candidates:
        # distance en y de la bande [band_top, band_bottom]
        if cy < band_top:
            dy = band_top - cy
        elif cy > band_bottom:
            dy = cy - band_bottom
        else:
            dy = 0.0

        # plus on est proche de la bande, plus c'est "bon"
        score = dy

        if best is None or score < best_score:
            best = (x, y, cw, ch)
            best_score = score

    return best  # (x, y, w, h)


def _build_similarity_transform_from_bboxes(bbox_ref, bbox_align):
    """
    À partir de deux bboxes de portes (video1, video2),
    calcule une transform 2x3 du type :
      [ [s, 0, tx],
        [0, s, ty] ]

    qui aligne la porte de la video2 sur celle de la video1.
    """
    (x1, y1, w1, h1) = bbox_ref
    (x2, y2, w2, h2) = bbox_align

    # centres
    cx1 = x1 + w1 / 2.0
    cy1 = y1 + h1 / 2.0
    cx2 = x2 + w2 / 2.0
    cy2 = y2 + h2 / 2.0

    # taille moyenne
    size1 = (w1 + h1) / 2.0
    size2 = (w2 + h2) / 2.0
    if size2 < 1.0:
        s = 1.0
    else:
        s = size1 / size2

    # limiter les valeurs extrêmes
    s = max(0.5, min(2.0, s))

    tx = cx1 - s * cx2
    ty = cy1 - s * cy2

    M = np.array([[s, 0.0, tx],
                  [0.0, s, ty]], dtype=np.float32)
    return M


def process_videos(
    video1_path: Path,
    video2_path: Path,
    output_path: Path,
    max_duration_s: float = MAX_REAL_DURATION_S,
    output_width: int = OUTPUT_WIDTH,
    output_height: int = OUTPUT_HEIGHT,
    output_fps: int = OUTPUT_FPS,
):
    """
    Crée une vidéo superposée :
      - video1 = référence (fond)
      - video2 = overlay, réaligné frame par frame sur la porte active
    La vidéo est ralentie de 40% (SPEED_FACTOR = 0.6).
    """
    video1_path = Path(video1_path)
    video2_path = Path(video2_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    dur1 = total1 / fps1
    dur2 = total2 / fps2
    real_duration = min(dur1, dur2, max_duration_s)

    # durée de la vidéo de sortie après ralenti
    out_duration = real_duration / SPEED_FACTOR
    max_frames_out = int(out_duration * output_fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        output_fps,
        (output_width, output_height),
    )
    if not writer.isOpened():
        cap1.release()
        cap2.release()
        raise RuntimeError("Impossible d'ouvrir le writer vidéo.")

    # transform de la frame précédente (pour continuité quand on ne trouve pas de porte)
    prev_M = np.array([[1.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0]], dtype=np.float32)

    for i in range(max_frames_out):
        # temps "réel" dans les vidéos originales (plus court que la sortie)
        t_real = (i / output_fps) * SPEED_FACTOR
        if t_real > real_duration:
            break

        idx1 = int(min(t_real * fps1, max(total1 - 1, 0)))
        idx2 = int(min(t_real * fps2, max(total2 - 1, 0)))

        cap1.set(cv2.CAP_PROP_POS_FRAMES, idx1)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, idx2)

        ret1, f1 = cap1.read()
        ret2, f2 = cap2.read()
        if not ret1 or not ret2:
            break

        f1r = cv2.resize(f1, (output_width, output_height))
        f2r = cv2.resize(f2, (output_width, output_height))

        # détection des portes dans chaque frame
        mask1 = _make_gate_mask(f1r)
        mask2 = _make_gate_mask(f2r)

        bbox1 = _find_active_gate_bbox(mask1)
        bbox2 = _find_active_gate_bbox(mask2)

        if bbox1 is not None and bbox2 is not None:
            M_cur = _build_similarity_transform_from_bboxes(bbox1, bbox2)
            # lissage avec la transform précédente (pour éviter les sauts)
            alpha = 0.5
            M = (alpha * M_cur + (1.0 - alpha) * prev_M).astype(np.float32)
            prev_M = M
        else:
            # si on ne trouve pas de porte, on réutilise la transform précédente
            M = prev_M

        warped2 = cv2.warpAffine(f2r, M, (output_width, output_height))

        blended = cv2.addWeighted(f1r, 0.5, warped2, 0.5, 0.0)
        writer.write(blended)

    cap1.release()
    cap2.release()
    writer.release()

    return output_path
