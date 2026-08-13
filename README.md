# sleep-awake-monitor2

## English

Jetson Drowsiness Monitor is a real-time drowsiness detection project for NVIDIA Jetson Orin Nano. It uses a USB camera and MediaPipe facial landmarks to measure whether the user's eyes are open or closed. If the eyes remain closed for 15 seconds, the system displays `SLEEP` and plays a warning sound. A custom DetectNet model trained to recognize `awake` and `sleep` can also be used as an optional secondary detector.

> The project supports live USB camera input, video files, and still images.

## Algorithm

The program reads an image from a camera or file and processes it as follows:

1. **Face detection:** MediaPipe Face Landmarker detects the face and the landmarks around both eyes.
2. **Eye measurement:** Six points from each eye are used to calculate the Eye Aspect Ratio (EAR).

   ```text
   EAR = (vertical distance 1 + vertical distance 2) / (2 × horizontal distance)
   ```

3. **State classification:** An EAR of `0.19` or higher is treated as `AWAKE`. A lower value is treated as `EYES CLOSED`.
4. **Time measurement:** A short closure is treated as a blink. If the eyes remain closed for at least 15 seconds, the state changes to `SLEEP` and the alarm is played. The alarm repeats every 10 seconds while the eyes remain closed.
5. **Optional DetectNet detection:** A custom MobileNet-SSD model can add an `awake` or `sleep` prediction, confidence score, and bounding box.
6. **Output:** The current state, EAR, closed-eye duration, eye landmarks, and optional FPS are drawn on the screen.

```text
USB camera / image / video
           ↓
MediaPipe facial landmarks
           ↓
      EAR calculation
           ↓
AWAKE → EYES CLOSED → SLEEP + alarm
```

Large frames are downscaled internally to reduce the load on the Jetson. Detection results are mapped back to the original resolution. If a small face is not found in the downscaled frame, the program retries at full resolution.

### Requirements

- NVIDIA Jetson Orin Nano with JetPack 6.0
- Python 3.10
- USB camera
- OpenCV
- NumPy
- MediaPipe Tasks
- GStreamer for MP3 alarm playback
- `jetson-inference` and `jetson-utils` only when using DetectNet

## Running the Project

1. Open a terminal and move to the project directory.

   ```bash
   cd /home/nvidia/sleep_awake
   ```

2. Install the required libraries and audio tools.

   ```bash
   python3 -m pip install --user numpy opencv-python mediapipe
   sudo apt update
   sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-good
   ```

3. Download the MediaPipe Face Landmarker model.

   ```bash
   mkdir -p landmark_models
   curl -fL -o landmark_models/face_landmarker.task \
     https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
   ```

4. Start monitoring with a USB camera. This command uses MediaPipe without DetectNet and displays the FPS.

   ```bash
   ./run_drowsiness_monitor.sh --no-detectnet --show-fps
   ```

   Press `q` or `Esc` to stop.

5. To use the included custom DetectNet model, run:

   ```bash
   ./run_drowsiness_monitor.sh --show-fps
   ```

   This option requires NVIDIA `jetson-inference` and `jetson-utils`.

To process a still image instead:

```bash
python3 drowsiness_monitor.py input.jpg --output result.jpg --no-display --no-detectnet
```

To process a video instead:

```bash
python3 drowsiness_monitor.py input.mp4 --output result.mp4 --no-detectnet
```

The EAR threshold can be adjusted for the user, camera angle, glasses, and lighting. For example:

```bash
./run_drowsiness_monitor.sh --no-detectnet --ear-threshold 0.20 --sleep-seconds 15
```

---

# Jetson 居眠り検知モニター

## 日本語

Jetson居眠り検知モニターは、NVIDIA Jetson Orin Nanoで動作するリアルタイム居眠り検知プロジェクトです。USBカメラとMediaPipeの顔ランドマークを使い、利用者の目が開いているか閉じているかを測定します。目を閉じた状態が15秒間続くと、画面に`SLEEP`と表示して警告音を再生します。必要に応じて、`awake`と`sleep`を学習した独自のDetectNetモデルも補助判定に使用できます。

> USBカメラのリアルタイム映像、動画ファイル、静止画像に対応しています。

## アルゴリズム

プログラムはカメラまたはファイルから画像を読み込み、次の順番で処理します。

1. **顔検出：** MediaPipe Face Landmarkerで顔と両目の周囲にあるランドマークを検出します。
2. **目の開き具合の測定：** 左右それぞれの目にある6点からEye Aspect Ratio（EAR）を計算します。

   ```text
   EAR =（縦方向の距離1 + 縦方向の距離2）/（2 × 横方向の距離）
   ```

3. **状態判定：** EARが`0.19`以上なら`AWAKE`、未満なら`EYES CLOSED`と判定します。
4. **閉眼時間の測定：** 短い閉眼はまばたきとして扱います。閉眼が15秒以上続くと`SLEEP`へ切り替え、警告音を鳴らします。閉眼中は10秒間隔で警告を繰り返します。
5. **DetectNetの補助判定（任意）：** 独自学習したMobileNet-SSDモデルから、`awake`または`sleep`のラベル、信頼度、検出枠を取得できます。
6. **結果表示：** 現在の状態、EAR、閉眼時間、目のランドマーク、必要に応じてFPSを画面に表示します。

```text
USBカメラ／画像／動画
          ↓
MediaPipe顔ランドマーク
          ↓
       EAR計算
          ↓
AWAKE → EYES CLOSED → SLEEP + 警告音
```

Jetsonの負荷を軽減するため、大きな画像は内部で縮小して処理し、検出結果を元の解像度へ戻します。縮小画像で小さな顔を検出できなかった場合は、元の解像度で自動的に再検出します。

### 必要な環境とライブラリ

- NVIDIA Jetson Orin Nano（JetPack 6.0）
- Python 3.10
- USBカメラ
- OpenCV
- NumPy
- MediaPipe Tasks
- MP3警告音の再生に使用するGStreamer
- DetectNetを使う場合のみ`jetson-inference`と`jetson-utils`

## プロジェクトの実行方法

1. 端末を開き、プロジェクトのフォルダへ移動します。

   ```bash
   cd /home/nvidia/sleep_awake
   ```

2. 必要なライブラリと音声再生ツールをインストールします。

   ```bash
   python3 -m pip install --user numpy opencv-python mediapipe
   sudo apt update
   sudo apt install gstreamer1.0-tools gstreamer1.0-plugins-good
   ```

3. MediaPipe Face Landmarkerモデルをダウンロードします。

   ```bash
   mkdir -p landmark_models
   curl -fL -o landmark_models/face_landmarker.task \
     https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
   ```

4. USBカメラで監視を開始します。次のコマンドはDetectNetを使わず、MediaPipeとFPS表示を使用します。

   ```bash
   ./run_drowsiness_monitor.sh --no-detectnet --show-fps
   ```

   終了するときは`q`またはEscキーを押します。

5. プロジェクトに含まれる独自DetectNetモデルを使用する場合は、次を実行します。

   ```bash
   ./run_drowsiness_monitor.sh --show-fps
   ```

   この方法では、NVIDIAの`jetson-inference`と`jetson-utils`が必要です。

静止画像を処理する場合：

```bash
python3 drowsiness_monitor.py input.jpg --output result.jpg --no-display --no-detectnet
```

動画を処理する場合：

```bash
python3 drowsiness_monitor.py input.mp4 --output result.mp4 --no-detectnet
```

EARの閾値は、利用者、カメラの角度、眼鏡、照明に合わせて変更できます。

```bash
./run_drowsiness_monitor.sh --no-detectnet --ear-threshold 0.20 --sleep-seconds 15
```

