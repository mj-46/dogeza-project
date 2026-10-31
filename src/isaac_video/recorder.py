"""動画保存用。isaac sim起動時に--recordをつけたときに使われる"""

import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np


class Mp4Writer:
    """RGB の画像を 1 枚ずつ受け取り、ffmpeg で MP4 にする。"""

    def __init__(self, path, width, height, fps):
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg が見つかりません。`sudo apt install ffmpeg` でインストールしてください。"
            )
        self.path = Path(path)
        self.width, self.height = width, height
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.process = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",  # 画像は標準入力から受け取る
                "-c:v",
                "libx264",
                "-preset",
                "slow",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",  # YouTube などで再生を始めやすくする
                str(self.path),
            ],
            stdin=subprocess.PIPE,
        )
        self.frames = 0

    def write(self, rgb):
        """(高さ, 幅, 3) の uint8 画像を 1 枚書き込む。"""
        self.process.stdin.write(np.ascontiguousarray(rgb, dtype=np.uint8).tobytes())
        self.frames += 1

    def close(self):
        self.process.stdin.close()
        self.process.wait()


class VideoRecorder:
    """Isaac Sim のカメラ（プリム）から映像を取り出して MP4 に保存する。"""

    def __init__(
        self, camera_path, video_dir, name="video", resolution=(1920, 1080), fps=60
    ):
        import omni.kit.app

        omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
            "omni.replicator.core", True
        )
        import omni.replicator.core as rep

        self.width, self.height = resolution
        self.fps = fps
        # カメラの映像を取り出す口（レンダープロダクト）と、色画像（rgb）の取り出し役（アノテーター）
        self.render_product = rep.create.render_product(camera_path, resolution)
        if hasattr(rep, "annotators"):
            self.annotator = rep.annotators.get("rgb")
        else:
            self.annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self.annotator.attach(self.render_product)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = Mp4Writer(
            Path(video_dir) / f"{name}_{stamp}.mp4", self.width, self.height, fps
        )
        print(f"[isaac_video] 録画を開始します: {self.writer.path}")

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()  # with ブロックを抜けたら（エラーでも）必ず保存して閉じる

    def run(self, simulation_app, step, duration, warmup_frames=30):
        """step(t) を呼びながら 1/fps 秒ずつ時刻を進め、duration 秒分を録画する。

        step: 時刻 t [秒] を受け取り、その時点の動作をさせる関数（ロボットの動かし方は呼び出す側が決める）
        """
        for i in range(-warmup_frames, int(duration * self.fps)):
            if not simulation_app.is_running():
                break
            step(
                max(i, 0) / self.fps
            )  # 最初の warmup_frames コマは t = 0 のまま（描画を安定させる）
            simulation_app.update()
            if i >= 0:
                self.capture()

    def capture(self):
        """直前に描画された 1 フレームを動画に追加する。画像がまだ準備できていなければ False。"""
        data = self.annotator.get_data()
        if data is None or np.size(data) == 0:
            return False
        image = np.asarray(data).reshape(self.height, self.width, -1)[
            :, :, :3
        ]  # RGBA → RGB
        self.writer.write(image)
        return True

    def close(self):
        self.writer.close()  # 先に動画を閉じて保存を確定させる
        self.annotator.detach()
        self.render_product.destroy()
        seconds = self.writer.frames / self.fps
        print(
            f"[isaac_video] 録画を保存しました（{self.writer.frames} フレーム, {seconds:.1f} 秒）: {self.writer.path}"
        )
