#!/usr/bin/env python3
"""
Apify를 이용해 인스타그램 릴스를 가져오는 공용 모듈.

계정(@username) 조회에는 apify/instagram-reel-scraper, 해시태그(#tag) 조회에는
apify/instagram-hashtag-scraper(resultsType=reels)를 사용한다. transcribe_reels.py와
app.py가 fetch_by_source / sort_and_trim을 그대로 가져다 쓴다.

환경변수:
  APIFY_API_TOKEN - Apify API 토큰 (코드에 직접 넣지 않음)

단독 실행 사용법:
  python fetch_reels.py <instagram_username> <count>
"""

import argparse
import os
import sys

import requests

REEL_SCRAPER_ACTOR = "apify~instagram-reel-scraper"
HASHTAG_SCRAPER_ACTOR = "apify~instagram-hashtag-scraper"
PROFILE_SCRAPER_ACTOR = "apify~instagram-profile-scraper"
API_BASE = "https://api.apify.com/v2"


def _run_actor_sync(actor_id: str, payload: dict, api_token: str) -> list[dict]:
    url = f"{API_BASE}/acts/{actor_id}/run-sync-get-dataset-items"
    response = requests.post(url, params={"token": api_token}, json=payload, timeout=300)
    response.raise_for_status()
    return response.json()


def fetch_account_reels(username: str, count: int, api_token: str) -> list[dict]:
    """apify/instagram-reel-scraper로 특정 계정의 릴스를 가져온다."""
    payload = {"username": [username], "resultsLimit": count}
    return _run_actor_sync(REEL_SCRAPER_ACTOR, payload, api_token)


def fetch_hashtag_reels(hashtag: str, count: int, api_token: str) -> list[dict]:
    """apify/instagram-hashtag-scraper(resultsType=reels)로 해시태그 릴스를 가져온다."""
    payload = {"hashtags": [hashtag], "resultsType": "reels", "resultsLimit": count}
    return _run_actor_sync(HASHTAG_SCRAPER_ACTOR, payload, api_token)


def fetch_account_followers(username: str, api_token: str) -> int | None:
    """apify/instagram-profile-scraper로 계정의 팔로워 수를 가져온다. 실패하면 None."""
    try:
        items = _run_actor_sync(PROFILE_SCRAPER_ACTOR, {"usernames": [username]}, api_token)
    except requests.HTTPError:
        return None
    if not items:
        return None
    return items[0].get("followersCount")


def fetch_by_source(api_token: str, source_type: str, value: str, count: int) -> list[dict]:
    """source_type("account" 또는 "hashtag")에 맞는 액터를 호출해 원본 릴스 목록을 반환한다."""
    if source_type == "hashtag":
        return fetch_hashtag_reels(value, count, api_token)
    return fetch_account_reels(value, count, api_token)


def sort_and_trim(items: list[dict], count: int, by: str = "timestamp") -> list[dict]:
    """by="views"면 조회수순, by="timestamp"면 최신순으로 정렬 후 count개로 자른다."""
    if by == "views":
        key_fn = lambda item: item.get("videoPlayCount") or item.get("videoViewCount") or 0
    else:
        key_fn = lambda item: item.get("timestamp") or ""
    items = sorted(items, key=key_fn, reverse=True)
    return items[:count]


def fetch_reels(username: str, count: int, api_token: str) -> list[dict]:
    """계정 아이디의 릴스를 최신순으로 count개 가져온다 (단독 스크립트용 편의 함수)."""
    items = fetch_account_reels(username, count, api_token)
    return sort_and_trim(items, count, by="timestamp")


def main() -> None:
    parser = argparse.ArgumentParser(description="인스타그램 계정의 최근 릴스를 가져옵니다.")
    parser.add_argument("username", help="인스타그램 계정 아이디 (예: nike)")
    parser.add_argument("count", type=int, help="가져올 릴스 개수")
    args = parser.parse_args()

    api_token = os.environ.get("APIFY_API_TOKEN")
    if not api_token:
        print("오류: 환경변수 APIFY_API_TOKEN이 설정되어 있지 않습니다.", file=sys.stderr)
        sys.exit(1)

    try:
        reels = fetch_reels(args.username, args.count, api_token)
    except requests.HTTPError as e:
        print(f"Apify API 호출 실패: {e}\n응답: {e.response.text}", file=sys.stderr)
        sys.exit(1)

    print(f"'{args.username}' 계정의 릴스 {len(reels)}개:\n")
    for i, reel in enumerate(reels, start=1):
        url = reel.get("url")
        timestamp = reel.get("timestamp")
        caption = (reel.get("caption") or "").replace("\n", " ")[:80]
        views = reel.get("videoPlayCount") or reel.get("videoViewCount")
        likes = reel.get("likesCount")
        comments = reel.get("commentsCount")

        print(f"{i}. {url}")
        print(f"   업로드: {timestamp}")
        print(f"   조회수: {views} | 좋아요: {likes} | 댓글: {comments}")
        if caption:
            print(f"   캡션: {caption}")
        print()


if __name__ == "__main__":
    main()
