#!/usr/bin/env python3
"""Eye-landmark based drowsiness monitor with optional DetectNet assistance."""

import argparse
import math
import shutil
import subprocess
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


DEFAULT_FACE_MODEL = Path("/home/nvidia/sleep_awake/landmark_models/face_landmarker.task")
DEFAULT_DETECT_MODEL = Path(
    "/home/nvidia/sleep_awake/retrained_models_167images_20260812/"
    "sleep_awake_ssd_mobilenet_167images.onnx"
)
DEFAULT_LABELS = DEFAULT_DETECT_MODEL.parent / "labels.txt"
DEFAULT_ALARM_SOUND = Path(
    "/home/nvidia/sleep_awake/drowsiness_warning.mp3"
)

# MediaPipe Face Mesh landmark indices.  The first/fourth points form the
# horizontal eye line, and the other pairs measure the two vertical openings.
RIGHT_EYE_EAR = (33, 160, 158, 133, 153, 144)
LEFT_EYE_EAR = (362, 385, 387, 263, 373, 380)
RIGHT_EYE_CONTOUR = (33, 160, 158, 133, 153, 144)
LEFT_EYE_CONTOUR = (362, 385, 387, 263, 373, 380)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect prolonged eye closure from an image, video, or USB camera."
    )
    parser.add_argument(
        "input",
        nargs="?",
        default="0",
        help="image/video path or camera number (default: 0)",
    )
    parser.add_argument("--output", help="optional output image/video path")
    parser.add_argument("--no-display", action="store_true", help="do not open a GUI window")
    parser.add_argument("--ear-threshold", type=float, default=0.19)
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=15.0,
        help="continuous eye-closure time before SLEEP (default: 15.0)",
    )
    parser.add_argument(
        "--alarm-sound",
        type=Path,
        default=DEFAULT_ALARM_SOUND,
        help="warning sound played after SLEEP is detected",
    )
    parser.add_argument(
        "--alarm-cooldown",
        type=float,
        default=10.0,
        help="seconds between repeated warnings while asleep (default: 10.0)",
    )
    parser.add_argument("--no-alarm", action="store_true", help="disable warning sound")
    parser.add_argument("--face-model", type=Path, default=DEFAULT_FACE_MODEL)
    parser.add_argument("--no-detectnet", action="store_true")
    parser.add_argument("--detect-model", type=Path, default=DEFAULT_DETECT_MODEL)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--detect-threshold", type=float, default=0.5)
    parser.add_argument(
        "--detect-every",
        type=int,
        default=3,
        help="run DetectNet every N video frames (default: 3)",
    )
    parser.add_argument(
        "--process-width",
        type=int,
        default=640,
        help="MediaPipe processing width; output keeps its original size (default: 640)",
    )
    parser.add_argument(
        "--detect-width",
        type=int,
        default=640,
        help="DetectNet processing width; boxes are mapped to the original (default: 640)",
    )
    parser.add_argument("--show-fps", action="store_true", help="show processing FPS")
    parser.add_argument("--verbose", action="store_true", help="print status for every frame")
    return parser.parse_args()


def distance(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def eye_aspect_ratio(points, indices):
    horizontal = distance(points[indices[0]], points[indices[3]])
    if horizontal <= 1e-8:
        return 0.0
    vertical_1 = distance(points[indices[1]], points[indices[5]])
    vertical_2 = distance(points[indices[2]], points[indices[4]])
    return (vertical_1 + vertical_2) / (2.0 * horizontal)


def pixel_point(landmark, width, height):
    return int(round(landmark.x * width)), int(round(landmark.y * height))


def draw_eye(frame, points, indices, color):
    height, width = frame.shape[:2]
    polygon = np.array([pixel_point(points[i], width, height) for i in indices], np.int32)
    cv2.polylines(frame, [polygon], True, color, 2, cv2.LINE_AA)
    for point in polygon:
        cv2.circle(frame, tuple(point), 3, color, -1, cv2.LINE_AA)


class DetectNetHelper:
    def __init__(self, model, labels, threshold, process_width):
        from jetson_inference import detectNet

        self._cuda_from_numpy = __import__("jetson_utils").cudaFromNumpy
        self.net = detectNet(
            model=str(model),
            labels=str(labels),
            input_blob="input_0",
            output_cvg="scores",
            output_bbox="boxes",
            threshold=threshold,
        )
        self.process_width = process_width

    def detect(self, frame):
        height, width = frame.shape[:2]
        processed = resize_for_inference(frame, self.process_width)
        processed_height, processed_width = processed.shape[:2]
        rgba = cv2.cvtColor(processed, cv2.COLOR_BGR2RGBA)
        detections = self.net.Detect(self._cuda_from_numpy(rgba), overlay="none")
        if not detections:
            return None
        best = max(detections, key=lambda item: item.Confidence)
        scale_x = width / processed_width
        scale_y = height / processed_height
        return {
            "label": self.net.GetClassDesc(best.ClassID),
            "confidence": float(best.Confidence),
            "box": (
                int(round(best.Left * scale_x)),
                int(round(best.Top * scale_y)),
                int(round(best.Right * scale_x)),
                int(round(best.Bottom * scale_y)),
            ),
        }


def resize_for_inference(frame, target_width):
    """Downscale large frames while preserving aspect ratio; never upscale."""
    if target_width <= 0 or frame.shape[1] <= target_width:
        return frame
    scale = target_width / frame.shape[1]
    target_height = max(1, int(round(frame.shape[0] * scale)))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


class AlarmPlayer:
    """Play warnings asynchronously so audio never blocks video processing."""

    def __init__(self, sound_path, cooldown):
        self.sound_path = sound_path
        self.cooldown = cooldown
        self.last_played = float("-inf")
        self.process = None
        if sound_path.suffix.lower() == ".mp3":
            self.command = [
                "gst-play-1.0", "--no-interactive", "--quiet", str(sound_path)
            ]
        else:
            self.command = ["paplay", str(sound_path)]

    def play_if_ready(self, now):
        if now - self.last_played < self.cooldown:
            return False
        if self.process is not None and self.process.poll() is None:
            return False
        self.process = subprocess.Popen(
            self.command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.last_played = now
        return True


def open_capture(source):
    path = Path(source)
    is_image = path.is_file() and path.suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".webp", ".bmp"
    }
    if is_image:
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"画像を読み込めません: {path}")
        return None, frame, True

    capture_source = int(source) if source.isdigit() else source
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise RuntimeError(f"カメラ/動画を開けません: {source}")
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture, None, False


def make_writer(path, frame, fps):
    if not path:
        return None
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        return None
    height, width = frame.shape[:2]
    return cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps if fps > 0 else 30.0, (width, height)
    )


def put_status(frame, state, ear, closed_duration, detect_result, fps=None):
    state_colors = {
        "AWAKE": (40, 220, 40),
        "EYES CLOSED": (0, 210, 255),
        "SLEEP": (30, 30, 240),
        "NO FACE": (180, 180, 180),
    }
    color = state_colors[state]
    scale = max(0.7, min(frame.shape[:2]) / 900.0)
    thickness = max(2, int(round(scale * 2)))
    lines = [state]
    if ear is not None:
        lines.append(f"EAR: {ear:.3f}  closed: {closed_duration:.1f}s")
    if detect_result:
        lines.append(
            f"DetectNet: {detect_result['label']} {detect_result['confidence'] * 100:.1f}%"
        )
        x1, y1, x2, y2 = detect_result["box"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    if fps is not None:
        lines.append(f"FPS: {fps:.1f}")

    y = 45
    for index, line in enumerate(lines):
        font_scale = scale * (1.15 if index == 0 else 0.72)
        (tw, th), baseline = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(frame, (12, y - th - 10), (24 + tw, y + baseline + 5), (0, 0, 0), -1)
        cv2.putText(
            frame, line, (18, y), cv2.FONT_HERSHEY_SIMPLEX,
            font_scale, color if index == 0 else (255, 255, 255), thickness, cv2.LINE_AA,
        )
        y += th + baseline + 20


def main():
    args = parse_args()
    if not args.face_model.is_file():
        raise FileNotFoundError(f"Face Landmarkerモデルがありません: {args.face_model}")
    if args.detect_every < 1:
        raise ValueError("--detect-every は1以上にしてください")
    if args.process_width < 1 or args.detect_width < 1:
        raise ValueError("--process-width と --detect-width は1以上にしてください")
    if args.sleep_seconds <= 0:
        raise ValueError("--sleep-seconds は0より大きい値にしてください")
    if args.alarm_cooldown <= 0:
        raise ValueError("--alarm-cooldown は0より大きい値にしてください")

    alarm = None
    if not args.no_alarm:
        if not args.alarm_sound.is_file():
            raise FileNotFoundError(f"警告音ファイルがありません: {args.alarm_sound}")
        player = "gst-play-1.0" if args.alarm_sound.suffix.lower() == ".mp3" else "paplay"
        if shutil.which(player) is None:
            raise FileNotFoundError(f"警告音の再生に必要な{player}がありません")
        alarm = AlarmPlayer(args.alarm_sound, args.alarm_cooldown)

    capture, still_frame, is_image = open_capture(args.input)
    detector = None
    if not args.no_detectnet:
        if not args.detect_model.is_file() or not args.labels.is_file():
            raise FileNotFoundError("DetectNetのモデルまたはlabels.txtがありません")
        detector = DetectNetHelper(
            args.detect_model, args.labels, args.detect_threshold, args.detect_width
        )

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.face_model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    closed_since = None
    frame_number = 0
    last_detection = None
    writer = None
    start_time = time.monotonic()
    previous_frame_time = None
    smoothed_fps = None

    with mp.tasks.vision.FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            if is_image:
                frame = still_frame.copy()
            else:
                ok, frame = capture.read()
                if not ok:
                    break

            now = time.monotonic()
            if previous_frame_time is not None:
                frame_interval = now - previous_frame_time
                if frame_interval > 1e-6:
                    current_fps = 1.0 / frame_interval
                    smoothed_fps = (
                        current_fps
                        if smoothed_fps is None
                        else 0.1 * current_fps + 0.9 * smoothed_fps
                    )
            previous_frame_time = now
            timestamp_ms = max(frame_number, int((now - start_time) * 1000))
            landmark_frame = resize_for_inference(frame, args.process_width)
            rgb = cv2.cvtColor(landmark_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            # Preserve small/distant-face detection: pay the full-resolution
            # cost only when the lightweight pass couldn't find a face.
            if not result.face_landmarks and landmark_frame is not frame:
                full_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                full_mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(full_rgb),
                )
                result = landmarker.detect_for_video(full_mp_image, timestamp_ms + 1)

            if detector and (is_image or frame_number % args.detect_every == 0):
                last_detection = detector.detect(frame)

            ear = None
            if result.face_landmarks:
                points = result.face_landmarks[0]
                left_ear = eye_aspect_ratio(points, LEFT_EYE_EAR)
                right_ear = eye_aspect_ratio(points, RIGHT_EYE_EAR)
                ear = (left_ear + right_ear) / 2.0
                eye_color = (0, 0, 255) if ear < args.ear_threshold else (0, 255, 0)
                draw_eye(frame, points, LEFT_EYE_CONTOUR, eye_color)
                draw_eye(frame, points, RIGHT_EYE_CONTOUR, eye_color)

            if ear is None:
                closed_since = None
                state = "NO FACE"
                closed_duration = 0.0
            elif ear < args.ear_threshold:
                if closed_since is None:
                    closed_since = now
                closed_duration = now - closed_since
                state = "SLEEP" if closed_duration >= args.sleep_seconds else "EYES CLOSED"
            else:
                closed_since = None
                closed_duration = 0.0
                state = "AWAKE"

            if state == "SLEEP" and alarm and not is_image:
                alarm.play_if_ready(now)

            put_status(
                frame,
                state,
                ear,
                closed_duration,
                last_detection,
                smoothed_fps if args.show_fps and not is_image else None,
            )
            if args.verbose or is_image:
                print(
                    f"state={state} ear={ear if ear is not None else 'none'} "
                    f"closed={closed_duration:.2f}s detect={last_detection}",
                    flush=True,
                )

            if writer is None and args.output and not is_image:
                fps = capture.get(cv2.CAP_PROP_FPS)
                writer = make_writer(args.output, frame, fps)
            if writer:
                writer.write(frame)

            if not args.no_display:
                preview = frame
                max_width = 1280
                if frame.shape[1] > max_width:
                    ratio = max_width / frame.shape[1]
                    preview = cv2.resize(frame, None, fx=ratio, fy=ratio)
                cv2.imshow("Sleep/Awake Monitor - q to quit", preview)
                if cv2.waitKey(1 if not is_image else 0) & 0xFF in (ord("q"), 27):
                    break

            if is_image:
                if args.output:
                    output = Path(args.output)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    if not cv2.imwrite(str(output), frame):
                        raise RuntimeError(f"出力画像を保存できません: {output}")
                    print(f"saved={output}")
                break
            frame_number += 1

    if capture:
        capture.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
