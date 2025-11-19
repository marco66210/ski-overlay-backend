from pathlib import Path
import cv2
import numpy as np

# Durée max traitée par vidéo (secondes)
MAX_DURATION_S = 40.0
# Résolution d'analyse pour les features (plus petite que la sortie pour gagner du temps)
ANALYSIS_WIDTH = 640
ANALYSIS_HEIGHT = 360
# Nombre maximum de frames utilisées pour calculer la transformation
MAX_ANALYSIS_FRAMES = 60


def _make_gate_mask(bgr_frame_resized):
    """
    Masque pour isoler les portes de slalom :
    - drapeaux rouges / orangés
    - drapeaux bleus

    On travaille en HSV, puis on combine les masques.
    """
    hsv = cv2.cvtColor(bgr_frame_resized, cv2.COLOR_BGR2HSV)

    # --- PLAGES ROUGES / ORANGE (portes rouges) ---

    # Rouge autour de 0°
    lower_red1 = np.array([0, 80, 70], dtype=np.uint8)
    upper_red1 = np.array([15, 255, 255], dtype=np.uint8)

    # Rouge autour de 180°
    lower_red2 = np.array([160, 80, 70], dtype=np.uint8)
    upper_red2 = np.array([179, 255, 255], dtype=np.uint8)

    # Orange / rouge clair (certains panneaux sont plus orangés)
    lower_orange = np.array([10, 80, 70], dtype=np.uint8)
    upper_orange = np.array([30, 255, 255], dtype=np.uint8)

    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask_red = cv2.bitwise_or(mask_red, mask_orange)

    # --- PLAGE BLEUE (portes bleues) ---
    # Bleu typique : teinte ~100–140° -> [90, 130] dans l’espace OpenCV (0–179)
    lower_blue = np.array([90, 80, 70], dtype=np.uint8)
    upper_blue = np.array([130, 255, 255], dtype=np.uint8)

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # --- COMBINAISON ROUGE + BLEU ---
    mask = cv2.bitwise_or(mask_red, mask_blue)

    # Nettoyage : petite fermeture morphologique pour supprimer le bruit
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return mask


def _sample_frames(cap, max_duration_s=MAX_DURATION_S, max_frames=MAX_ANALYSIS_FRAMES):
    """
    Choisit un sous-ensemble de frames (indice, temps) pour analyser la vidéo.
    """
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total_frames <= 0:
        return []

    total_duration = total_frames / fps
    effective_duration = min(max_duration_s, total_duration)
    if effective_duration <= 0:
        return []

    # On répartit max_frames sur la durée effective
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
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ret, frame = cap.read()
    return frame if ret else None


def _compute_affine_transform_with_gates(video_path_ref: Path, video_path_to_align: Path):
    """
    Calcule une transformation affine globale (scale + rotation + translation)
    qui aligne au mieux les portes (drapeaux) entre les deux vidéos.

    On détecte les features avec ORB, mais en les limitant à un masque de couleur
    correspondant aux drapeaux de slalom.
    """
    cap_ref = cv2.VideoCapture(str(video_path_ref))
    cap_al = cv2.VideoCapture(str(video_path_to_align))

    if not cap_ref.isOpened() or not cap_al.isOpened():
        cap_ref.release()
        cap_al.release()
        return np.array([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0]], dtype=np.float32)

    samples_ref = _sample_frames(cap_ref)
    samples_al = _sample_frames(cap_al)
    n = min(len(samples_ref), len(samples_al))
    if n == 0:
        cap_ref.release()
        cap_al.release()
        return np.array([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0]], dtype=np.float32)

    orb = cv2.ORB_create(400)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    transforms = []

    for i in range(n):
        idx_ref, _ = samples_ref[i]
        idx_al, _ = samples_al[i]

        fr1 = _read_frame(cap_ref, idx_ref)
        fr2 = _read_frame(cap_al, idx_al)
        if fr1 is None or fr2 is None:
            continue

        fr1_small = cv2.resize(fr1, (ANALYSIS_WIDTH, ANALYSIS_HEIGHT))
        fr2_small = cv2.resize(fr2, (ANALYSIS_WIDTH, ANALYSIS_HEIGHT))

        mask1 = _make_gate_mask(fr1_small)
        mask2 = _make_gate_mask(fr2_small)

        kp1, des1 = orb.detectAndCompute(fr1_small, mask1)
        kp2, des2 = orb.detectAndCompute(fr2_small, mask2)
        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            continue

        # KNN matching + ratio test de Lowe
        matches_knn = bf.knnMatch(des1, des2, k=2)
        good = []
        for m, n2 in matches_knn:
            if m.distance < 0.75 * n2.distance:
                good.append(m)

        if len(good) < 6:
            continue

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

        # Transformation affine partielle (rotation + scale + translation)
        M, inliers = cv2.estimateAffinePartial2D(
            pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=4.0
        )
        if M is not None:
            transforms.append(M.astype(np.float32))

    cap_ref.release()
    cap_al.release()

    if not transforms:
        # Identité si rien trouvé
        return np.array([[1.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0]], dtype=np.float32)

    # Moyenne des matrices 2x3
    T = np.mean(np.stack(transforms, axis=0), axis=0)
    return T.astype(np.float32)


def process_videos(
    video1_path: Path,
    video2_path: Path,
    output_path: Path,
    max_duration_s: float = MAX_DURATION_S,
    output_width: int = 1280,
    output_height: int = 720,
    output_fps: int = 30,
):
    """
    Crée une vidéo superposée à partir de deux vidéos :
      - video1 = référence (fond)
      - video2 = vidéo à aligner (overlay)

    On calcule une transformation affine globale basée sur les portes,
    puis on applique cette transform sur toutes les frames de la vidéo 2.
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

    # Transformation globale (affine) à partir des portes
    M = _compute_affine_transform_with_gates(video1_path, video2_path)

    max_frames_out = int(max_duration_s * output_fps)

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

    for i in range(max_frames_out):
        t = i / output_fps
        if t > max_duration_s:
            break

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

        # Application de la transform affine sur la vidéo 2
        warped2 = cv2.warpAffine(f2r, M, (output_width, output_height))

        # Superposition simple
        blended = cv2.addWeighted(f1r, 0.5, warped2, 0.5, 0.0)
        writer.write(blended)

    cap1.release()
    cap2.release()
    writer.release()

    return output_path
