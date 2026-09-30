"""릴스 구조 분석 + 벤치마킹 대본 생성.

세 entry point:
  - analyze_reel(result, api_key): 릴스 1개를 성과→후킹→구간구조→템포→
    정보속도→키워드→설득포인트→핵심카피→성공요인→벤치마킹포인트 순으로
    분석해 script_generation_input까지 포함한 구조화 JSON을 반환한다.
    실측 데이터(조회수 등)와 좋아요율/조회효율 같은 계산값은 코드에서 직접
    구하고, GPT에게는 캡션·대본·구간별 프레임 이미지만 근거로 판단하게 한다.
  - generate_benchmark_script(reference_result, keyword, api_key): 위 분석의
    script_generation_input을 템플릿으로 삼아 새 키워드용 대본을 생성한다.
  - analyze_run(run_meta, results, api_key): 런 전체(여러 릴스) 교차 비교
    마크다운 리포트.
"""

import base64
import json
import os

from openai import OpenAI

ANALYSIS_MODEL = os.environ.get("ANALYSIS_MODEL", "gpt-4o")

HOOK_TYPES = [
    "질문형", "공감형", "문제제기형", "결과선공개형", "숫자형",
    "비교형", "도발형", "실험형", "정보형", "가격형", "호기심형",
]
TIMELINE_ROLES = [
    "HOOK", "PROBLEM", "CURIOSITY", "PRODUCT", "FEATURE", "TEST",
    "DEMONSTRATION", "RESULT", "COMPARISON", "PROOF", "PRICE",
    "BENEFIT", "OFFER", "CTA",
]
PERSUASION_TYPES = [
    "문제 제기", "기능", "차별점", "비교", "직접 사용", "테스트", "결과 증명",
    "숫자", "전문가 의견", "사용자 후기", "가격", "할인", "희소성", "CTA",
]
PRODUCT_APPEARANCE_ROLES = {"PRODUCT", "FEATURE", "DEMONSTRATION"}

# ---------------------------------------------------------------------------
# 릴스 1개 분석 (성과 → 후킹 → 구간구조 → 템포 → 키워드 → 설득 → 성공요인)
# ---------------------------------------------------------------------------

REEL_SCHEMA = {
    "name": "reel_analysis",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "hook": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "types": {"type": "array", "items": {"type": "string", "enum": HOOK_TYPES}},
                    "first_visual": {"type": "string", "description": "0~3초 화면에 실제로 보이는 것"},
                    "first_line": {"type": "string", "description": "0~3초 첫 대사(대본 원문)"},
                    "first_caption": {"type": "string", "description": "0~3초 화면에 보이는 첫 자막(번인 텍스트), 없으면 '없음'"},
                    "product_shown": {"type": "boolean"},
                    "person_shown": {"type": "boolean"},
                    "number_used": {"type": "boolean"},
                    "question_used": {"type": "boolean"},
                    "result_preview": {"type": "boolean"},
                    "comparison_shown": {"type": "boolean"},
                    "hook_device": {"type": "string", "description": "실제로 사용한 후킹 장치가 무엇인지 설명"},
                    "benchmark_point": {"type": "string", "description": "다른 콘텐츠에 그대로 재사용할 수 있는 구조적 포인트"},
                },
                "required": [
                    "types", "first_visual", "first_line", "first_caption",
                    "product_shown", "person_shown", "number_used", "question_used",
                    "result_preview", "comparison_shown", "hook_device", "benchmark_point",
                ],
            },
            "timeline": {
                "type": "array",
                "description": "대본 구간마다 정확히 하나. index는 1부터 시작하며 제공된 구간 번호와 같아야 한다.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "index": {"type": "integer"},
                        "role": {"type": "string", "enum": TIMELINE_ROLES, "description": "이 구간이 전체 구조에서 하는 역할"},
                        "visual": {"type": "string", "description": "이 구간 프레임에 실제로 보이는 것. 프레임이 없으면 '프레임 없음'"},
                        "screen_caption": {"type": "string", "description": "이 구간 화면의 번인 자막. 없으면 '없음', 프레임 없으면 '프레임 없음'"},
                        "keywords": {"type": "array", "items": {"type": "string"}, "description": "이 구간에서 나온 핵심 단어"},
                    },
                    "required": ["index", "role", "visual", "screen_caption", "keywords"],
                },
            },
            "key_info_segment_indices": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "구간 중 실제로 새로운 핵심 정보(기능/사용법/비교/결과/가격/혜택 등)가 등장한 구간 번호만. 감탄사·전환용 대사는 제외.",
            },
            "keywords": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "product": {"type": "array", "items": {"type": "string"}},
                    "problem": {"type": "array", "items": {"type": "string"}},
                    "benefit": {"type": "array", "items": {"type": "string"}},
                    "result": {"type": "array", "items": {"type": "string"}},
                    "hook": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["product", "problem", "benefit", "result", "hook"],
            },
            "persuasion": {
                "type": "array",
                "description": "아래 14개 유형 전부에 대해 사용 여부를 판단해 정확히 14개 항목을 반환하라.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "type": {"type": "string", "enum": PERSUASION_TYPES},
                        "used": {"type": "boolean"},
                        "evidence": {"type": "string", "description": "used가 true면 실제 근거 원문/장면, false면 '없음'"},
                        "timestamp": {"type": "string", "description": "MM:SS 형식, 근거 없으면 빈 문자열"},
                    },
                    "required": ["type", "used", "evidence", "timestamp"],
                },
            },
            "key_copy": {
                "type": "array",
                "maxItems": 6,
                "description": "벤치마킹 가치가 있는 문장만 최대 6개. 대본 전체를 요약하지 마라.",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "text": {"type": "string"},
                        "timestamp": {"type": "string", "description": "MM:SS"},
                        "role": {"type": "string", "enum": TIMELINE_ROLES},
                        "copy_type": {"type": "array", "items": {"type": "string", "enum": HOOK_TYPES}},
                        "keywords": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["text", "timestamp", "role", "copy_type", "keywords"],
                },
            },
            "success_factors": {
                "type": "array",
                "maxItems": 3,
                "description": "이 영상이 반응이 좋았던 이유. 반드시 관찰 가능한 구체적 근거를 포함해라 ('후킹이 강했다' 같은 추상적 서술 금지).",
                "items": {"type": "string"},
            },
            "benchmark_points": {
                "type": "array",
                "description": "success_factors 중 다른 콘텐츠에 '구조'로 재사용 가능한 것만 뽑아 일반화한 문장. 내용 복사가 아니라 구조 복사 관점으로 써라.",
                "items": {"type": "string"},
            },
        },
        "required": ["hook", "timeline", "key_info_segment_indices", "keywords", "persuasion", "key_copy", "success_factors", "benchmark_points"],
    },
}

REEL_SYSTEM_PROMPT = """너는 숏폼 릴스를 분석해서 콘텐츠 기획자가 바로 벤치마킹 대본을 만들 수 있게
구조를 뜯어주는 분석가다. 아래 질문에 실제로 답할 수 있는 수준으로 분석해라:
1) 이 영상은 왜 조회가 잘 나왔는가 2) 첫 3초 후킹 방식 3) 내용 전개 순서
4) 제품 등장 시점 5) 시청 유지 변화가 몇 초 간격으로 발생했는가
6) 강조한 제품 특성/키워드 7) 구매 설득 핵심 근거
8) 다른 콘텐츠에 재사용 가능한 구조 9) 이 구조로 새 대본을 어떻게 만들 수 있는가.

지켜야 할 것:
- 너에게는 캡션 원문, whisper로 전사된 구간별 대본(번호+시작 시각 포함), 이미지 여러 장이 주어진다.
- 이미지 순서: 1번째 = 커버 프레임(영상 시작). 이후 = 대본 구간 1, 2, 3...의 시작 시점 프레임.
  프레임 수가 구간 수보다 적으면 뒤쪽 구간의 visual/screen_caption은 '프레임 없음'으로 채워라.
- timeline은 대본 구간 전부에 정확히 하나씩, 반드시 TIMELINE_ROLES 중 하나로 역할을 태깅해라. 임의로 구간을 합치거나 빼지 마라.
- key_info_segment_indices는 감탄사·필러(예: "Oh...", "Cut.")가 아니라 실제로 새 정보가 나온 구간만 골라라.
- persuasion은 반드시 14개 유형 전부에 대해 사용 여부를 판단해라. 사용되지 않았으면 evidence는 '없음'.
- key_copy·success_factors·benchmark_points는 전부 실제 대본/캡션/화면에 있는 근거에 기반해라. 감정 평가나 추상적 심리 분석("설렘을 유발한다" 등)은 쓰지 마라 — 관찰 가능한 사실만 써라.
- 정지 이미지로 알 수 없는 것(움직임, 배경음 등)은 추측하지 말고 visual에 '판단 불가'라고 써라.
- 전부 한국어로 답해라."""


def _b64_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _format_mmss(seconds) -> str:
    if seconds is None:
        return "00:00"
    total = int(round(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def _compute_performance(result: dict) -> dict:
    """실제 데이터([A])와 그걸로 계산한 값([B])만. AI 분석은 섞지 않는다."""
    views = result.get("views_count")
    likes = result.get("likes_count")
    comments = result.get("comments_count")
    followers = result.get("followers_count")
    duration = result.get("duration_seconds")
    segments = result.get("segments") or []
    if not duration and segments:
        duration = segments[-1].get("end", 0) - segments[0].get("start", 0)

    like_rate = (likes / views * 100) if (likes is not None and views) else None
    comment_rate = (comments / views * 100) if (comments is not None and views) else None
    follow_view_ratio = (views / followers) if (views is not None and followers) else None

    return {
        "views": views,
        "likes": likes,
        "comments": comments,
        "duration_seconds": duration,
        "posted_at": result.get("timestamp"),
        "followers": followers,
        "caption": result.get("caption"),
        "like_rate": like_rate,
        "comment_rate": comment_rate,
        "follow_view_ratio": follow_view_ratio,
    }


def _build_reel_user_text(result: dict, performance: dict, n_frames_attached: int) -> str:
    segments = result.get("segments") or []
    seg_lines = "\n".join(
        f"구간 {i + 1} [{_format_mmss(s.get('start'))} ~ {_format_mmss(s.get('end'))}] {s.get('text', '').strip()}"
        for i, s in enumerate(segments)
    ) or (result.get("transcript_text") or "(대본 없음)")

    return f"""[이 릴스의 실제 데이터]
캡션: {performance["caption"] or "없음"}
길이: {performance["duration_seconds"]}초
조회수/좋아요/댓글: {performance["views"]}/{performance["likes"]}/{performance["comments"]}

대본 (구간별, 번호와 시작~끝 시각 포함):
{seg_lines}

첨부된 이미지는 총 {n_frames_attached}장이다 (1번째 = 커버, 이후 = 구간 1, 2, 3...의 시작 프레임 순서).
위 데이터와 이미지만 근거로 삼아 JSON 스키마에 맞춰 답하라."""


def analyze_reel(result: dict, api_key: str) -> dict:
    """릴스 1개를 성과→후킹→구간구조→템포→키워드→설득→성공요인→벤치마킹포인트 순으로 분석한다."""
    client = OpenAI(api_key=api_key)
    performance = _compute_performance(result)

    thumbnail_path = result.get("thumbnail_path")
    frame_paths = [thumbnail_path] + list(result.get("segment_frame_paths") or [])
    frame_paths = [p for p in frame_paths if p and os.path.exists(p)]

    user_text = _build_reel_user_text(result, performance, len(frame_paths))

    content = [{"type": "text", "text": user_text}]
    for p in frame_paths:
        b64 = _b64_image(p)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    response = client.chat.completions.create(
        model=ANALYSIS_MODEL,
        messages=[
            {"role": "system", "content": REEL_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        response_format={"type": "json_schema", "json_schema": REEL_SCHEMA},
    )
    parsed = json.loads(response.choices[0].message.content)

    # timeline: GPT가 준 role/visual/keywords에 whisper 구간의 실측 start/end/원문을 합친다.
    transcript_segments = result.get("segments") or []
    timeline = []
    for item in parsed.get("timeline", []):
        idx = item.get("index")
        ts = transcript_segments[idx - 1] if idx and 1 <= idx <= len(transcript_segments) else {}
        timeline.append({
            **item,
            "start": ts.get("start"),
            "end": ts.get("end"),
            "script": ts.get("text", ""),
        })
    timeline.sort(key=lambda s: s.get("index") or 0)

    # 콘텐츠 구조 공식: 연속 중복 role을 접어서 한 줄 시퀀스로 만든다.
    structure = []
    for t in timeline:
        if not structure or structure[-1] != t["role"]:
            structure.append(t["role"])

    video_duration = performance["duration_seconds"] or 0
    scene_changes = len(timeline)
    average_change_interval = (video_duration / scene_changes) if scene_changes else None
    product_first_appearance = next(
        (t["start"] for t in timeline if t["role"] in PRODUCT_APPEARANCE_ROLES and t["start"] is not None),
        None,
    )
    pacing = {
        "video_duration": video_duration,
        "scene_changes": scene_changes,
        "average_change_interval": average_change_interval,
        "product_first_appearance": product_first_appearance,
    }

    key_info_indices = sorted(set(parsed.get("key_info_segment_indices") or []))
    key_info_count = len(key_info_indices)
    information_pace = {
        "key_info_count": key_info_count,
        "average_info_interval": (video_duration / key_info_count) if key_info_count else None,
        "key_info_segment_indices": key_info_indices,
    }

    keywords = parsed.get("keywords") or {}
    script_generation_input = {
        "structure": structure,
        "hook_type": (parsed.get("hook") or {}).get("types", []),
        "target_duration": video_duration,
        "product_keywords": keywords.get("product", []),
        "problem_keywords": keywords.get("problem", []),
        "benefit_keywords": keywords.get("benefit", []),
        "result_keywords": keywords.get("result", []),
    }

    return {
        "performance": performance,
        "hook": parsed["hook"],
        "timeline": timeline,
        "structure": structure,
        "pacing": pacing,
        "information_pace": information_pace,
        "keywords": keywords,
        "persuasion": parsed.get("persuasion", []),
        "key_copy": parsed.get("key_copy", []),
        "success_factors": parsed.get("success_factors", []),
        "benchmark_points": parsed.get("benchmark_points", []),
        "script_generation_input": script_generation_input,
    }


# ---------------------------------------------------------------------------
# 벤치마킹 대본 생성 (script_generation_input을 템플릿 삼아 새 키워드용 대본 작성)
# ---------------------------------------------------------------------------

BENCHMARK_SCHEMA = {
    "name": "benchmark_script",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "benchmarked_structure": {"type": "string", "description": "원본 구조를 한 줄 공식으로 (예: '질문형 후킹 → 제품 공개 → 특징 설명 → 실제 테스트 → 결과 → 장점 정리 → CTA')"},
            "segments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "start_sec": {"type": "number"},
                        "end_sec": {"type": "number"},
                        "role": {"type": "string", "enum": TIMELINE_ROLES},
                        "line": {"type": "string", "description": "실제로 말할 대본 문장"},
                        "visual": {"type": "string", "description": "이 구간에서 화면에 무엇을 보여줄지"},
                    },
                    "required": ["start_sec", "end_sec", "role", "line", "visual"],
                },
            },
            "caption_suggestion": {"type": "string"},
            "cta": {"type": "string"},
            "structure_notes": {"type": "string", "description": "원본의 어떤 구조 요소(후킹 유형/정보 전개 속도/제품 등장 시점/결과 공개 위치 등)를 어떻게 반영했는지 설명"},
        },
        "required": ["benchmarked_structure", "segments", "caption_suggestion", "cta", "structure_notes"],
    },
}

BENCHMARK_SYSTEM_PROMPT = """너는 숏폼 릴스 대본 작가다. 원본 릴스 하나의 구조 분석 데이터(script_generation_input +
원본 구간별 타임라인)가 주어진다. 원본 내용을 베끼지 말고, 원본이 쓴 "구조 설계"만 따라서
완전히 새로운 주제(키워드)의 새 대본을 써라.

반드시 지킬 것:
- 후킹 유형(hook_type)과 같은 방식으로 새 후킹을 만들어라.
- structure(역할 시퀀스)와 같은 순서로 구간을 배치해라.
- target_duration과 비슷한 총 길이로 만들고, 원본 구간별 시간 비율(각 role이 몇 초를 썼는지)을 참고해 새 구간 길이를 배분해라.
- 제품 등장 시점(product_first_appearance)과 비슷한 시점에 새 제품을 등장시켜라.
- 정보 전개 속도(average_info_interval)와 비슷한 간격으로 새 핵심 정보를 배치해라.
- product_keywords/problem_keywords/benefit_keywords/result_keywords에 있는 "역할"만 참고하고,
  실제 단어는 새 키워드에 맞게 새로 써라. 원본 키워드나 문장을 그대로 쓰지 마라.
  예: 입력 키워드를 먼저 제품특징/문제/혜택으로 분류한 뒤, 분류에 맞는 위치(HOOK/FEATURE/TEST/RESULT 등)에 배치해라.
- structure_notes에 원본의 어떤 구조 수치를 어떻게 반영했는지 설명해라.
- 원본 대본이 다른 언어여도 새 대본(line, caption_suggestion, cta)은 반드시 한국어로만 써라.
- 전부 한국어로 답해라."""


def _reference_structure_text(reference_result: dict) -> str:
    a = reference_result.get("analysis") or {}
    sgi = a.get("script_generation_input") or {}
    pacing = a.get("pacing") or {}
    info_pace = a.get("information_pace") or {}
    hook = a.get("hook") or {}

    timeline_summary = "\n".join(
        f"- [{t.get('role')}] {_format_mmss(t.get('start'))}~{_format_mmss(t.get('end'))} "
        f"({(t.get('end') or 0) - (t.get('start') or 0):.1f}초): \"{t.get('script')}\" / 화면: {t.get('visual')}"
        for t in (a.get("timeline") or [])
    ) or "(구간 정보 없음)"

    return f"""[원본 릴스 구조 데이터 — script_generation_input]
{json.dumps(sgi, ensure_ascii=False, indent=2)}

[원본 후킹]
유형: {hook.get("types")} / 장치: {hook.get("hook_device")}

[원본 템포]
총 길이: {pacing.get("video_duration")}초, 장면 변화 {pacing.get("scene_changes")}회 (평균 {pacing.get("average_change_interval")}초 간격)
제품 최초 등장: {pacing.get("product_first_appearance")}초
핵심 정보 {info_pace.get("key_info_count")}개 (평균 {info_pace.get("average_info_interval")}초 간격)

[원본 구간별 타임라인]
{timeline_summary}

[원본 설득 포인트]
{json.dumps([p for p in (a.get("persuasion") or []) if p.get("used")], ensure_ascii=False)}"""


def generate_benchmark_script(reference_result: dict, keyword: str, api_key: str) -> dict:
    """reference_result(분석이 끝난 릴스 하나)의 script_generation_input을 템플릿 삼아 새 대본을 생성한다."""
    if not reference_result.get("analysis"):
        raise ValueError("기준 릴스에 분석 데이터가 없습니다. 먼저 해당 릴스의 해부분석을 실행하세요.")

    client = OpenAI(api_key=api_key)
    structure_text = _reference_structure_text(reference_result)
    user_text = f"""{structure_text}

[새 키워드]
{keyword}

위 구조를 템플릿 삼아 "{keyword}"에 대한 완전히 새로운 릴스 대본을 JSON 스키마에 맞춰 작성하라."""

    response = client.chat.completions.create(
        model=ANALYSIS_MODEL,
        messages=[
            {"role": "system", "content": BENCHMARK_SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ],
        response_format={"type": "json_schema", "json_schema": BENCHMARK_SCHEMA},
    )
    return json.loads(response.choices[0].message.content)


# ---------------------------------------------------------------------------
# 런 전체(여러 릴스) 교차 비교 리포트 — 기존 기능, 변경 없음
# ---------------------------------------------------------------------------

RUBRIC = """[레이어 1 · 진입각 — 0~3초]
텍스트 채널: 정보공백 / 이득예고 / 손실경고 / 숫자고정 / 자격증명
  → 해당하는 것 전부, 그리고 겹친 개수를 세라
  → 3초에 던진 걸 본문에서 안 풀면 '미끼'로 따로 표시
화면 채널: 결과선공개 / 통념반박 / 예상이탈 / 전후대비 / 상황공감 / 없음
소리 채널: 첫 발화가 3초 안에 나왔는지
  → 실제로 일한 채널 수를 0~3으로

[레이어 2 · 체류 설계]
전개형: 문제→처방 / 의외→따라하기 / 경험→권유 / 나열
정보 항목 수, 그리고 항목당 몇 초인지
보류 장치: 유형 / 몇 초에 걸었는지 / 전체의 몇 % 지점인지

[레이어 3 · 행동 설계]
저장 / 공유 / 댓글 / 팔로우 / 없음
공유 유도면 대상이 특정됐는지
캡션 첫 줄이 단독으로 후킹하는지
캡션이 본문 반복인지 보완인지
총정리·순위형 구조인지

[레이어 4 · 수요 폭]
동기: 재미 / 해소 / 획득
욕구군: 자원 / 지위 / 역량 / 신체 / 정서
폭: 좁음 / 보통 / 넓음
이 주제를 한 단계 넓히는 경로 1개

[공통 규칙]
모든 항목에 근거 원문을 인용해라. 없으면 '불명'.
영상을 봐야 아는 항목은 [눈으로 확인]으로 따로 모아라.
추측으로 채우지 마라.
다 뜯은 다음에 공통 패턴 3개.
그리고 상위 그룹과 하위 그룹의 수치 차이를 비교해라."""

SYSTEM_PROMPT = """너는 숏폼 릴스 구조를 해부하는 분석가다. 다음을 반드시 지켜라.

- 너에게는 영상 화면과 소리를 직접 볼 권한이 없다. 오직 사용자가 제공하는 캡션 원문, \
whisper로 전사된 구간별 대본(시작 시각 포함), 조회수/좋아요/댓글/길이 수치만 근거로 쓸 수 있다.
- "화면 채널"처럼 화면을 봐야 판별되는 항목, 배경음/효과음 여부처럼 텍스트로 알 수 없는 항목은 \
절대 추측하지 말고 전부 "[눈으로 확인]" 섹션에 모아 어떤 항목이 확인 불가한지만 나열해라.
- 판정마다 근거가 된 원문(캡션 문장 또는 대본 문장)을 그대로 인용해라. 근거가 없으면 '불명'이라고 써라.
- 추측으로 빈칸을 채우지 마라. 데이터에 없으면 없다고 말해라.
- 릴스마다 레이어 1~4를 항목별로 정리하고, 마지막에 공통 패턴 3개, 그리고 조회수 기준 상위 그룹과 \
하위 그룹의 수치 차이 비교를 작성해라. 전부 한국어로, 마크다운 제목/목록을 사용해서 작성해라."""


def _reel_payload(index: int, r: dict) -> dict:
    caption = r.get("caption") or None
    first_line = caption.split("\n")[0] if caption else None
    return {
        "index": index,
        "url": r.get("reel_url"),
        "owner": r.get("owner_username") or "불명",
        "caption": caption or "없음",
        "caption_first_line": first_line or "없음",
        "views": r.get("views_count"),
        "likes": r.get("likes_count"),
        "comments": r.get("comments_count"),
        "duration_seconds": r.get("duration_seconds"),
        "hook_text_0_3s": (r.get("hook") or {}).get("text") or "없음",
        "segments": [
            {"start": s.get("start"), "end": s.get("end"), "text": s.get("text")}
            for s in (r.get("segments") or [])
        ],
    }


def build_user_prompt(run_meta: dict, results: list) -> str:
    usable = [r for r in results if not r.get("error")]
    skipped = len(results) - len(usable)
    payload = [_reel_payload(i, r) for i, r in enumerate(usable, start=1)]

    parts = [RUBRIC, ""]
    parts.append(f"대상: {run_meta.get('label')} ({run_meta.get('type')}), 분석 가능한 릴스 {len(usable)}개")
    if skipped:
        parts.append(f"(처리 실패로 제외된 릴스 {skipped}개는 분석 대상에서 뺐다)")
    parts.append("")
    parts.append("아래는 릴스별 원본 데이터(JSON)다. 이 데이터에 있는 내용만 근거로 사용해라:")
    parts.append(json.dumps(payload, ensure_ascii=False, indent=2))
    return "\n".join(parts)


def analyze_run(run_meta: dict, results: list, api_key: str) -> str:
    usable = [r for r in results if not r.get("error")]
    if not usable:
        return "분석할 수 있는 릴스가 없습니다 (전부 처리 실패)."

    client = OpenAI(api_key=api_key)
    user_prompt = build_user_prompt(run_meta, results)

    response = client.chat.completions.create(
        model=ANALYSIS_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content
