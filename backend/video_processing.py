from pathlib import Path
import cv2
import numpy as np

# Durée max de la partie utilisée des vidéos (en secondes, temps "réel")
MAX_REAL_DURATION_S = 40.0

# Résolution de sortie (après crop + resize)
OUTPUT_WIDTH = 1280
OUTPUT_HEIGHT = 720
OUTPUT_FPS = 30

# Facteur de vitesse (0.6 = 40% plus lent que l'original)
SPEED_FACTOR = 0.6  # la vidéo est ralentie de 40 %


# Résolution d'analyse pour le tracking (on travaille en plus petit pour aller plus vite)
ANALYSIS_WIDTH = 640
ANALYSIS_HEIGHT = 360

# Taille relative du crop (par rapport à la résolution source)
# Exemple : 0.6 -> on prend un rectangle couvrant ~60% de la largeur / hauteur
CROP_REL_W = 0.6
CROP_REL_H = 0.6


def _detect_moving_box(bg_subtractor, frame_bgr):
    """
    Détecte la personne (skieur) comme le plus gros blob en mouvement
    sur une frame (en BGR).
    On travaille sur une version réduite pour aller plus vite.
    Retourne (cx, cy) dans les coordonnées de la FRAME ORIGINALE,
    ou None si rien de fiable.
    """
    h, w = frame_bgr.shape[:2]

    # Resize pour analyse
    small = cv2.resize(frame_bgr, (ANALYSIS_WIDTH, ANALYSIS_HEIGHT))

    fgmask = bg_subtractor.apply(small)

    # On nettoie un peu le masque
    _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel, iterations=1)
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_DILATE, kernel, iterations=2)

    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # On prend le plus grand contour comme "skieur"
    best_cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best_cnt)

    # Si l'aire est ridicule, on considère qu'on n'a rien trouvé
    min_area = (ANALYSIS_WIDTH * ANALYSIS_HEIGHT) * 0.001
    if area < min_area:
        return None

    x, y, cw, ch = cv2.boundingRect(best_cnt)
    cx_small = x + cw / 2.0
    cy_small = y + ch / 2.0

    # Remap dans les coordonnées originales
    sx = w / float(ANALYSIS_WIDTH)
    sy = h / float(ANALYSIS_HEIGHT)

    cx = cx_small * sx
    cy = cy_small * sy

    return cx, cy


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
    Cas caméra FIXE sur pied (4K ou 1080p) :

    - Les deux vidéos ont le même cadrage (ou très proche),
    - On détecte le skieur sur chaque vidéo (blob en mouvement),
    - On calcule un centre moyen des deux skieurs,
    - On définit un CROP (fenêtre) autour de ce centre,
    - On extrait le même CROP dans les deux vidéos,
    - On les overlay (alpha 0.5),
    - On ralentit la vidéo de 40 % (SPEED_FACTOR = 0.6).

    Résultat : une vidéo recadrée qui suit les 2 skieurs au fil du run.
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

    # Nombre de frames de sortie après ralenti
    out_duration = real_duration / SPEED_FACTOR
    max_frames_out = int(out_duration * output_fps)

    # On supposera que les résolutions des deux vidéos sont identiques ou très proches.
    # On prendra la résolution de la première vidéo comme base.
    cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret1, first_frame1 = cap1.read()
    if not ret1:
        cap1.release()
        cap2.release()
        raise RuntimeError("Impossible de lire la première frame de la vidéo 1.")

    h1, w1 = first_frame1.shape[:2]

    # Dimension du crop dans l'image source
    crop_w = int(w1 * CROP_REL_W)
    crop_h = int(h1 * CROP_REL_H)
    crop_w = max(32, min(w1, crop_w))
    crop_h = max(32, min(h1, crop_h))

    # Writer pour la vidéo de sortie
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

    # Background subtractors pour les 2 vidéos (tracking des skieurs)
    bg1 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    bg2 = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    # Valeurs par défaut des centres (milieu de l'image)
    cx1_prev, cy1_prev = w1 / 2.0, h1 / 2.0
    cx2_prev, cy2_prev = w1 / 2.0, h1 / 2.0

    for i in range(max_frames_out):
        # Temps réel dans les vidéos originales
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

        # Assurer même résolution (au cas où les 2 vidéos diffèrent légèrement)
        f1r = f1
        f2r = cv2.resize(f2, (f1r.shape[1], f1r.shape[0]))

        h, w = f1r.shape[:2]

        # Détection skieurs
        c1 = _detect_moving_box(bg1, f1r)
        c2 = _detect_moving_box(bg2, f2r)

        if c1 is not None:
            cx1_prev, cy1_prev = c1
        if c2 is not None:
            cx2_prev, cy2_prev = c2

        # Centre moyen des 2 skieurs
        cx_avg = (cx1_prev + cx2_prev) / 2.0
        cy_avg = (cy1_prev + cy2_prev) / 2.0

        # On centre le crop sur cx_avg, cy_avg, en restant dans l'image
        half_w = crop_w / 2.0
        half_h = crop_h / 2.0

        x0 = int(cx_avg - half_w)
        y0 = int(cy_avg - half_h)

        # Clamp pour rester dans l'image
        x0 = max(0, min(w - crop_w, x0))
        y0 = max(0, min(h - crop_h, y0))

        x1 = x0 + crop_w
        y1 = y0 + crop_h

        # Crop des 2 vidéos au même endroit
        crop1 = f1r[y0:y1, x0:x1]
        crop2 = f2r[y0:y1, x0:x1]

        # Sécurité : si le crop sort de l'image pour une raison quelconque
        if crop1.shape[0] <= 0 or crop1.shape[1] <= 0 or crop2.shape[0] <= 0 or crop2.shape[1] <= 0:
            continue

        # Resize vers la résolution de sortie
        crop1_res = cv2.resize(crop1, (output_width, output_height))
        crop2_res = cv2.resize(crop2, (output_width, output_height))

        # Superposition
        blended = cv2.addWeighted(crop1_res, 0.5, crop2_res, 0.5, 0.0)
        writer.write(blended)

    cap1.release()
    cap2.release()
    writer.release()

    return output_path
