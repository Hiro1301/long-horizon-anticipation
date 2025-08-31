import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score


def extract_features(video_path: str, stride: int = 5):
    """Extract simple frame difference features from a video.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    stride : int
        Sample every `stride`-th frame to reduce computation.

    Returns
    -------
    np.ndarray
        Array of shape (num_samples, 4) with mean, variance, min and max of pixel differences.
    """
    cap = cv2.VideoCapture(video_path)
    features = []
    ret, prev_frame = cap.read()
    if not ret:
        return np.empty((0, 4))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_counter += 1
        if frame_counter % stride != 0:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        f = [np.mean(diff), np.var(diff), np.min(diff), np.max(diff)]
        features.append(f)
        prev_gray = gray
    cap.release()
    return np.array(features)


def label_pre_anomaly(num_samples: int, anomaly_indices: list, fps: float, horizon_sec: float) -> np.ndarray:
    """Create labels for pre-anomaly frames based on anomaly frame indices.

    Parameters
    ----------
    num_samples : int
        Total number of feature samples (frames / stride).
    anomaly_indices : list
        Indices of anomaly frames (in terms of sampled frames, not original video frames).
    fps : float
        Frames per second of the original video.
    horizon_sec : float
        How many seconds before the anomaly to label as pre-anomaly.

    Returns
    -------
    np.ndarray
        Label array of length `num_samples` where 1 indicates pre-anomaly and 0 indicates normal. 
        Anomaly indices themselves are marked with 2 and should be excluded during training.
    """
    labels = np.zeros(num_samples, dtype=int)
    horizon_frames = int(horizon_sec * fps)
    for idx in anomaly_indices:
        start = max(0, idx - horizon_frames)
        end = idx
        labels[start:end] = 1
        labels[idx] = 2
    return labels


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """Train a logistic regression classifier."""
    clf = LogisticRegression(max_iter=100)
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(model: LogisticRegression, X_test: np.ndarray, y_true: np.ndarray):
    """Evaluate the model on test data and return AUC and other metrics."""
    # Consider pre-anomaly (label=1) versus normal (0). Exclude actual anomaly (label=2)
    mask = y_true != 2
    y_binary = (y_true[mask] == 1).astype(int)
    probs = model.predict_proba(X_test[mask])[:, 1]
    try:
        auc = roc_auc_score(y_binary, probs)
    except ValueError:
        auc = float('nan')
    pred_binary = (probs > 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_binary, pred_binary, average='binary', zero_division=0)
    return auc, precision, recall, f1


# Example usage
if __name__ == "__main__":
    # Placeholder paths (update these paths to actual video files)
    training_videos = [
        "path/to/your/training/video1.mp4",
        "path/to/your/training/video2.mp4",
    ]
    eval_video = "path/to/your/evaluation/video.mp4"
    eval_anomaly_indices = [300]  # indices where anomaly occurs in sampled frames
    fps = 30.0  # frames per second of original video
    horizon_sec = 15  # label pre-anomaly for 15 seconds before anomaly

    # Extract features from training videos and assign normal/pre-anomaly labels
    X_train = []
    y_train = []
    for vid in training_videos:
        feats = extract_features(vid)
        # For training data, assume last 20% of frames are pre-anomaly and remaining are normal
        n = len(feats)
        labels = np.zeros(n, dtype=int)
        pre_start = int(0.8 * n)
        labels[pre_start:] = 1
        X_train.append(feats)
        y_train.append(labels)
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)

    # Train model, ignoring anomaly label (2) if present
    model = train_model(X_train, y_train)

    # Extract features for evaluation video
    X_eval = extract_features(eval_video)
    # Create labels for evaluation (1 = pre-anomaly, 0 = normal, 2 = anomaly). Replace indices accordingly.
    y_eval = label_pre_anomaly(len(X_eval), eval_anomaly_indices, fps, horizon_sec)

    # Evaluate the model
    auc, precision, recall, f1 = evaluate_model(model, X_eval, y_eval)
    print(f"AUC: {auc:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
