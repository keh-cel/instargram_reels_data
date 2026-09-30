#!/usr/bin/env python3
"""Fetch reels for an Instagram account or hashtag, transcribe each with
OpenAI Whisper, and save the first 0-3s of each reel as a separate "hook"
clip + hook text.

Pipeline per reel:
  1. Download the reel video (videoUrl from the Apify dataset item).
  2. Extract its audio track and send it to OpenAI Whisper
     (model=whisper-1, response_format=verbose_json, timestamp_granularities=["word"]).
  3. Cut a 0-3s video clip ("hook") from the downloaded reel.
  4. Build the hook transcript from the words whose start time falls within 0-3s.

Each run is stored under output/runs/<slug>/ so results from different
accounts/hashtags don't overwrite each other.

Usage:
    python transcribe_reels.py <username_or_#hashtag> [count] [--output-dir output]

Both APIFY_API_TOKEN and OPENAI_API_KEY are read from environment variables
(loaded from .env via python-dotenv) — never hardcode them in this file.
"""

import sys

if not sys.flags.utf8_mode:
    # cp949 로케일에서 UTF-8 모드가 꺼져 있으면 OpenAI 응답의 한글이 깨진다.
    # (app.py를 거치지 않고 이 파일을 단독 CLI로 실행하는 경우를 위한 안전장치.)
    import subprocess as _subprocess
    _result = _subprocess.run([sys.executable, "-X", "utf8", *sys.argv])
    sys.exit(_result.returncode)

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import imageio_ffmpeg
import requests
from dotenv import load_dotenv
from openai import OpenAI

from analyze import analyze_reel, analyze_run
from fetch_reels import fetch_account_followers, fetch_by_source, sort_and_trim
from sources import detect_source, slugify

load_dotenv()

HOOK_DURATION_SECONDS = 3.0
WHISPER_MODEL = "whisper-1"
DOWNLOAD_TIMEOUT_SECONDS = 300
MAX_SEGMENT_FRAMES = 8  # 시간대별 화면/자막 분석에 쓸 프레임 최대 개수 (vision 비용 통제)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Instagram username, @username, or #hashtag")
    parser.add_argument("count", type=int, nargs="?", default=10, help="Number of reels (default: 10)")
    parser.add_argument("--output-dir", default="output", help="Base directory to store runs")
    return parser.parse_args()


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} environment variable is not set (add it to .env).")
    return value


def download_video(url: str, dest_path: Path):
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_SECONDS) as r:
        r.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def run_ffmpeg(args: list, retries: int = 3):
    """이 환경에서는 libx264 인코더 초기화 중 malloc이 간헐적으로 실패하는 현상이
    관찰돼(시스템 메모리 여유와 무관), 짧은 대기 후 재시도한다."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    last_error = None
    for attempt in range(retries):
        try:
            subprocess.run([ffmpeg_exe, "-y", *args], check=True, capture_output=True)
            return
        except subprocess.CalledProcessError as e:
            last_error = e
            time.sleep(1.5 * (attempt + 1))
    raise last_error


def extract_audio(video_path: Path, audio_path: Path):
    run_ffmpeg(["-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", str(audio_path)])


def cut_hook_clip(video_path: Path, hook_path: Path, duration: float):
    run_ffmpeg(["-i", str(video_path), "-t", str(duration), "-c:v", "libx264", "-c:a", "aac", str(hook_path)])


def extract_thumbnail(video_path: Path, thumb_path: Path):
    run_ffmpeg(["-ss", "0.5", "-i", str(video_path), "-frames:v", "1", "-vf", "scale=320:-1", str(thumb_path)])


def extract_frame_at(video_path: Path, seconds: float, out_path: Path):
    ts = max(0.0, seconds)
    run_ffmpeg(["-ss", f"{ts:.3f}", "-i", str(video_path), "-frames:v", "1", "-vf", "scale=320:-1", str(out_path)])


def transcribe(client: OpenAI, audio_path: Path) -> dict:
    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model=WHISPER_MODEL,
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )
    return response.model_dump()


def extract_hook_text(transcript: dict, duration: float) -> str:
    words = transcript.get("words") or []
    hook_words = [w["word"] for w in words if w.get("start", 0) < duration]
    return " ".join(hook_words).strip()


def process_reel(client: OpenAI, index: int, reel: dict, dirs: dict, followers_count=None) -> dict:
    video_url = reel.get("videoUrl")
    shortcode = reel.get("shortCode") or reel.get("id") or str(index)
    base_name = f"{index:02d}_{shortcode}"

    common = {
        "reel_url": reel.get("url"),
        "owner_username": reel.get("ownerUsername") or reel.get("username"),
        "caption": reel.get("caption"),
        "timestamp": reel.get("timestamp"),
        "likes_count": reel.get("likesCount"),
        "comments_count": reel.get("commentsCount"),
        "views_count": reel.get("videoPlayCount") or reel.get("videoViewCount") or reel.get("igPlayCount"),
        "duration_seconds": reel.get("videoDuration"),
        "is_pinned": bool(reel.get("isPinned", False)),
        "followers_count": followers_count,
    }

    if not video_url:
        print(f"[{base_name}] skipped: no videoUrl in reel data", file=sys.stderr)
        return {**common, "error": "no videoUrl"}

    video_path = dirs["videos"] / f"{base_name}.mp4"
    audio_path = dirs["videos"] / f"{base_name}.mp3"
    hook_path = dirs["hooks"] / f"{base_name}_hook.mp4"
    thumb_path = dirs["thumbs"] / f"{base_name}.jpg"
    transcript_path = dirs["transcripts"] / f"{base_name}.json"

    print(f"[{base_name}] downloading video...")
    download_video(video_url, video_path)

    print(f"[{base_name}] extracting thumbnail...")
    extract_thumbnail(video_path, thumb_path)

    print(f"[{base_name}] extracting audio...")
    extract_audio(video_path, audio_path)

    print(f"[{base_name}] cutting 0-{HOOK_DURATION_SECONDS:g}s hook clip...")
    cut_hook_clip(video_path, hook_path, HOOK_DURATION_SECONDS)

    print(f"[{base_name}] transcribing with whisper-1...")
    transcript = transcribe(client, audio_path)
    transcript_path.write_text(json.dumps(transcript, ensure_ascii=False, indent=2), encoding="utf-8")

    hook_text = extract_hook_text(transcript, HOOK_DURATION_SECONDS)
    segments = [
        {"start": s.get("start"), "end": s.get("end"), "text": s.get("text", "").strip()}
        for s in (transcript.get("segments") or [])
    ]
    words = [
        {"word": w.get("word"), "start": w.get("start"), "end": w.get("end")}
        for w in (transcript.get("words") or [])
    ]

    print(f"[{base_name}] extracting up to {MAX_SEGMENT_FRAMES} segment frames...")
    segment_frame_paths = []
    for i, seg in enumerate(segments[:MAX_SEGMENT_FRAMES]):
        frame_path = dirs["frames"] / f"{base_name}_seg{i + 1:02d}.jpg"
        try:
            extract_frame_at(video_path, seg.get("start") or 0.0, frame_path)
            segment_frame_paths.append(str(frame_path))
        except subprocess.CalledProcessError:
            segment_frame_paths.append(None)
    segment_frame_paths += [None] * (len(segments) - len(segment_frame_paths))

    return {
        **common,
        "video_path": str(video_path),
        "thumbnail_path": str(thumb_path),
        "transcript_path": str(transcript_path),
        "transcript_text": transcript.get("text", ""),
        "segments": segments,
        "words": words,
        "segment_frame_paths": segment_frame_paths,
        "hook": {
            "duration_seconds": HOOK_DURATION_SECONDS,
            "video_path": str(hook_path),
            "text": hook_text,
        },
    }


def run_pipeline(query: str, count: int, output_dir: str = "output", mode: str = "auto") -> tuple:
    """Detects whether `query` is an account or a hashtag, fetches `count`
    reels, transcribes them, and cuts hook clips. Writes all artifacts under
    output_dir/runs/<slug>/ (including results.json + meta.json) and returns
    (slug, results). Raises RuntimeError/ValueError on bad input or missing
    env vars."""
    apify_token = require_env("APIFY_API_TOKEN")
    openai_key = require_env("OPENAI_API_KEY")

    if count <= 0:
        raise ValueError("count must be a positive integer.")

    source_type, value, label = detect_source(query, mode)
    slug = slugify(source_type, value)

    kind_label = "해시태그" if source_type == "hashtag" else "계정"
    print(f"[{kind_label}] {label} 최근 {count}개 수집 중...")
    raw_reels = fetch_by_source(apify_token, source_type, value, count)

    if not raw_reels or not any(r.get("videoUrl") for r in raw_reels):
        if source_type == "account":
            raise RuntimeError(
                f"계정 {label}을(를) 찾을 수 없거나 공개된 릴스가 없습니다. "
                f"키워드로 찾으려면 검색어 앞에 #을 붙이거나 '상세 설정'에서 해시태그 모드를 선택해 보세요."
            )
        raise RuntimeError(f"해시태그 {label}에 대한 릴스를 찾을 수 없습니다.")

    sort_by = "views" if source_type == "hashtag" else "timestamp"
    reels = sort_and_trim(raw_reels, count, by=sort_by)

    followers_count = None
    if source_type == "account":
        print(f"[{kind_label}] {label} 팔로워 수 조회 중...")
        try:
            followers_count = fetch_account_followers(value, apify_token)
        except requests.exceptions.RequestException as e:
            print(f"팔로워 수 조회 실패 (조회 효율 계산 없이 진행): {e}", file=sys.stderr)

    run_dir = Path(output_dir) / "runs" / slug
    dirs = {
        "videos": run_dir / "videos",
        "hooks": run_dir / "hooks",
        "thumbs": run_dir / "thumbs",
        "frames": run_dir / "frames",
        "transcripts": run_dir / "transcripts",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    meta = {
        "slug": slug,
        "type": source_type,
        "value": value,
        "label": label,
        "count": count,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if not reels:
        print("No reels found.")
        (run_dir / "results.json").write_text("[]", encoding="utf-8")
        return slug, []

    client = OpenAI(api_key=openai_key)

    results = []
    for i, reel in enumerate(reels, start=1):
        try:
            result = process_reel(client, i, reel, dirs, followers_count=followers_count)
        except (requests.exceptions.RequestException, subprocess.CalledProcessError) as e:
            print(f"[{i:02d}] failed: {e}", file=sys.stderr)
            results.append({"reel_url": reel.get("url"), "error": str(e)})
            continue

        if "error" not in result:
            print(f"[{i:02d}] 콘텐츠 구조 분석 중 (후킹/구간구조/키워드/설득포인트)...")
            try:
                result["analysis"] = analyze_reel(result, openai_key)
            except Exception as e:
                print(f"[{i:02d}] 분석 실패: {e}", file=sys.stderr)
                result["analysis_error"] = str(e)
        results.append(result)

    summary_path = run_dir / "results.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = sum(1 for r in results if "error" not in r)
    print(f"\nDone: {ok}/{len(results)} reels processed. Summary saved to {summary_path}")

    print("해부 분석(레이어 1~4) 생성 중...")
    try:
        report = analyze_run(meta, results, openai_key)
        (run_dir / "analysis.md").write_text(report, encoding="utf-8")
        print("해부 분석 완료.")
    except Exception as e:
        print(f"해부 분석 실패: {e}", file=sys.stderr)
        (run_dir / "analysis.md").write_text(f"분석 실패: {e}", encoding="utf-8")

    return slug, results


def main():
    args = parse_args()
    try:
        run_pipeline(args.query, args.count, args.output_dir)
    except (RuntimeError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"Apify API error: {e}\n{e.response.text}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
