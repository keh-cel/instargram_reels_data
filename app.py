#!/usr/bin/env python3
"""Local dev server / UI for the reels pipeline: collect (account or hashtag),
watch live progress, and browse transcripts + hooks.

Usage:
    python app.py [--output-dir output] [--port 5000]
"""

import sys

if not sys.flags.utf8_mode:
    # 이 환경(Windows + 한국어 로케일 cp949)에서는 UTF-8 모드가 꺼져 있으면
    # OpenAI 응답의 한글 텍스트가 내부적으로 cp949로 잘못 디코딩되어 깨진다.
    # -X utf8로 강제 재실행해 데이터 손상을 막는다. (Windows에는 진짜 exec()가 없어
    # os.execv를 쓰면 PID가 바뀌어 버리므로, 자식 프로세스로 실행하고 종료코드를 그대로 물려준다.)
    import subprocess
    result = subprocess.run([sys.executable, "-X", "utf8", *sys.argv])
    sys.exit(result.returncode)

import argparse
import contextlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import markdown
from flask import Flask, abort, jsonify, render_template, request, send_from_directory

from analyze import generate_benchmark_script
from transcribe_reels import run_pipeline

app = Flask(__name__)
OUTPUT_DIR = Path("output")

JOBS = {}
JOBS_LOCK = threading.Lock()
BENCHMARKS_LOCK = threading.Lock()


class JobLogWriter:
    """Minimal stdout-like object that appends print() output to a job's log."""

    def __init__(self, job_id):
        self.job_id = job_id

    def write(self, s):
        s = s.rstrip("\n")
        if s:
            with JOBS_LOCK:
                JOBS[self.job_id]["log"].append(s)

    def flush(self):
        pass


def media_url(path_str):
    if not path_str:
        return None
    try:
        rel = Path(path_str).resolve().relative_to(OUTPUT_DIR.resolve())
    except ValueError:
        return None
    return f"/media/{rel.as_posix()}"


def load_run(slug: str):
    run_dir = OUTPUT_DIR / "runs" / slug
    meta_path = run_dir / "meta.json"
    results_path = run_dir / "results.json"
    if not meta_path.exists():
        return None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else []

    for r in results:
        r["video_url"] = media_url(r.get("video_path"))
        r["thumbnail_url"] = media_url(r.get("thumbnail_path"))
        hook = r.get("hook") or {}
        r["hook_video_url"] = media_url(hook.get("video_path"))
        r["hook_text"] = hook.get("text")

    analysis_path = run_dir / "analysis.md"
    if analysis_path.exists():
        meta["analysis_html"] = markdown.markdown(analysis_path.read_text(encoding="utf-8"))
    else:
        meta["analysis_html"] = None

    meta["results"] = results
    return meta


def list_runs():
    runs_dir = OUTPUT_DIR / "runs"
    if not runs_dir.exists():
        return []
    runs = []
    for meta_path in runs_dir.glob("*/meta.json"):
        try:
            runs.append(load_run(meta_path.parent.name))
        except (json.JSONDecodeError, OSError):
            continue
    runs.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return runs


@app.route("/")
def index():
    return render_template("index.html", runs=list_runs())


@app.route("/api/runs")
def api_runs():
    return jsonify(list_runs())


@app.route("/api/run", methods=["POST"])
def api_run():
    data = request.get_json(silent=True) or {}
    query = (data.get("query") or "").strip()
    mode = data.get("mode", "auto")
    try:
        count = int(data.get("count", 10))
    except (TypeError, ValueError):
        return jsonify({"error": "개수는 숫자여야 합니다."}), 400

    if not query:
        return jsonify({"error": "계정 아이디, #해시태그 또는 URL을 입력하세요."}), 400

    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {"status": "running", "log": [], "slug": None, "error": None}

    def worker():
        try:
            with contextlib.redirect_stdout(JobLogWriter(job_id)):
                slug, _ = run_pipeline(query, count, str(OUTPUT_DIR), mode)
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "done"
                JOBS[job_id]["slug"] = slug
        except Exception as e:
            with JOBS_LOCK:
                JOBS[job_id]["status"] = "error"
                JOBS[job_id]["error"] = str(e)

    threading.Thread(target=worker, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/api/jobs/<job_id>")
def api_job(job_id):
    job = JOBS.get(job_id)
    if not job:
        abort(404)
    return jsonify(job)


def benchmarks_path():
    return OUTPUT_DIR / "benchmarks.json"


def load_benchmarks():
    path = benchmarks_path()
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def save_benchmarks(entries):
    path = benchmarks_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")


@app.route("/api/benchmarks")
def api_benchmarks():
    entries = load_benchmarks()
    entries.sort(key=lambda e: e.get("created_at", ""), reverse=True)
    return jsonify(entries)


@app.route("/api/benchmarks", methods=["POST"])
def api_create_benchmark():
    data = request.get_json(silent=True) or {}
    slug = data.get("slug")
    keyword = (data.get("keyword") or "").strip()
    try:
        reel_index = int(data.get("reel_index"))
    except (TypeError, ValueError):
        return jsonify({"error": "기준 릴스를 선택하세요."}), 400

    if not slug or not keyword:
        return jsonify({"error": "런과 키워드를 모두 입력하세요."}), 400

    run = load_run(slug)
    if not run:
        return jsonify({"error": "해당 수집 결과를 찾을 수 없습니다."}), 404
    results = run.get("results") or []
    if not (0 <= reel_index < len(results)):
        return jsonify({"error": "기준 릴스를 다시 선택하세요."}), 400

    reel = results[reel_index]
    if reel.get("error"):
        return jsonify({"error": "이 릴스는 처리에 실패해 기준으로 쓸 수 없습니다."}), 400

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        return jsonify({"error": "OPENAI_API_KEY 환경변수가 설정되어 있지 않습니다."}), 500

    try:
        script = generate_benchmark_script(reel, keyword, openai_key)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"대본 생성 실패: {e}"}), 500

    entry = {
        "id": uuid.uuid4().hex[:12],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "keyword": keyword,
        "run_slug": slug,
        "run_label": run.get("label"),
        "reel_url": reel.get("reel_url"),
        "reel_owner": reel.get("owner_username"),
        "reel_caption_first_line": (reel.get("caption") or "").split("\n")[0],
        "script": script,
    }

    with BENCHMARKS_LOCK:
        entries = load_benchmarks()
        entries.append(entry)
        save_benchmarks(entries)

    return jsonify(entry)


@app.route("/media/<path:filepath>")
def media(filepath):
    full_path = (OUTPUT_DIR / filepath).resolve()
    if OUTPUT_DIR.resolve() not in full_path.parents:
        abort(404)
    return send_from_directory(OUTPUT_DIR, filepath)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="바인딩할 주소. 같은 네트워크의 다른 기기에서 접속하려면 0.0.0.0으로 설정 (기본값: 127.0.0.1, 이 PC에서만 접속 가능)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    OUTPUT_DIR = Path(args.output_dir)
    if args.host != "127.0.0.1":
        print(
            f"경고: {args.host}로 바인딩합니다 — 이 PC 밖에서도 접속 가능해집니다. "
            "인증이 없고 Flask debug 모드(원격 코드 실행 위험)가 켜져 있으니, "
            "신뢰할 수 있는 네트워크에서만 사용하세요.",
            file=sys.stderr,
        )
    app.run(debug=True, host=args.host, port=args.port)
