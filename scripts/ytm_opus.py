#!/usr/bin/env python3
"""
Download YouTube Music tracks as Opus with metadata and cover art.

Two modes:

  Single track (original behaviour):
    ./ytm_opus.py "https://music.youtube.com/watch?v=..."
    ./ytm_opus.py --genre Pop --keep-webm "https://music.youtube.com/watch?v=..."

  Bulk collection for ML training data:
    ./ytm_opus.py --search "best phonk 2023" --search "indie folk 2018"
    ./ytm_opus.py --from-file scripts/yt_music_queries.txt --max-tracks 2000
    ./ytm_opus.py "https://music.youtube.com/playlist?list=..."

In bulk mode the script expands playlists and searches into individual video
URLs, deduplicates by video_id, skips files that already exist in --out-dir
(by video_id suffix), and continues past per-track failures.

Default --out-dir is ``data/raw/yt_music`` so collections drop straight into
the latentjam-research data tree.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import hashlib
import json
import re
import shutil
import struct
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


def run(cmd: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            cmd,
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
        )
    except subprocess.CalledProcessError as exc:
        if exc.stderr:
            print(exc.stderr.strip(), file=sys.stderr)
        raise


def require_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"Missing required command: {name}")


def clean_value(value: Any) -> str:
    if value is None:
        return ""
    value = str(value).strip()
    return "" if value == "NA" else value


def safe_filename(value: str, fallback: str = "track") -> str:
    value = value.strip() or fallback
    value = value.replace("/", "-").replace(":", " -")
    value = re.sub(r"[\0\r\n\t]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value[:180].rstrip(" .") or fallback


def first_year(text: str) -> str:
    match = re.search(r"(?:℗|©|Released on:|\b)(\d{4})(?:-\d{2}-\d{2})?", text)
    return match.group(1) if match else ""


def release_date(info: dict[str, Any], description: str) -> str:
    raw = clean_value(info.get("release_date"))
    if re.fullmatch(r"\d{8}", raw):
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return raw
    if re.fullmatch(r"\d{4}", raw):
        return raw
    match = re.search(r"Released on:\s*(\d{4}-\d{2}-\d{2})", description)
    if match:
        return match.group(1)
    return first_year(description)


def copyright_line(description: str) -> str:
    for line in description.splitlines():
        line = line.strip()
        if "℗" in line or line.startswith("©"):
            return line
    return ""


def infer_from_title(title: str) -> tuple[str, str]:
    if " - " not in title:
        return "", title
    artist, track = title.split(" - ", 1)
    track = re.sub(r"\s*\(?official (audio|video)\)?\s*", "", track, flags=re.I).strip()
    track = re.sub(r"\s+HD$", "", track, flags=re.I).strip()
    return artist.strip(), track.strip() or title


def infer_genre(info: dict[str, Any], title: str, album: str, artist: str) -> str:
    text = f"{title} {album} {artist}".lower()
    if any(word in text for word in ("ost", "soundtrack", "skyrim", "spider-verse", "game soundtrack")):
        return "Soundtrack"
    if any(word in text for word in ("phonk", "montagem", "funk")):
        return "Brazilian Funk" if "montagem" in text else "Phonk"
    if any(word in text for word in ("anime", "hajimemashita", "miku")):
        return "Anime" if "hajimemashita" in text else "Electronic"
    if any(word in text for word in ("rock", "guns n", "radiohead", "тигре", "княzz", "электрослабость")):
        return "Rock"
    if any(word in text for word in ("dance", "guetta", "inna", "bellini", "nightcrawlers")):
        return "Dance"
    if any(word in text for word in ("rap", "hip-hop", "metro boomin", "bruno mars", "yeat", "sean paul")):
        return "Hip-Hop/Rap"
    return clean_value(info.get("genre")) or "Pop"


def mb_request(endpoint: str, params: dict[str, str] | None = None) -> dict[str, Any]:
    query = urllib.parse.urlencode({**(params or {}), "fmt": "json"})
    url = f"https://musicbrainz.org/ws/2/{endpoint}?{query}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "ytm_opus.py/1.0 (local MusicBrainz tag lookup)",
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def mb_phrase(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def mb_artist_credit(credits: list[dict[str, Any]]) -> tuple[str, str, str]:
    names: list[str] = []
    sort_names: list[str] = []
    ids: list[str] = []
    for credit in credits:
        artist = credit.get("artist") or {}
        name = clean_value(credit.get("name")) or clean_value(artist.get("name"))
        sort_name = clean_value(artist.get("sort-name"))
        artist_id = clean_value(artist.get("id"))
        if name:
            names.append(name)
        if sort_name:
            sort_names.append(sort_name)
        if artist_id:
            ids.append(artist_id)
    return "; ".join(names), "; ".join(sort_names), "; ".join(ids)


def mb_norm(value: str) -> str:
    return re.sub(r"[^0-9a-z]+", " ", value.casefold()).strip()


def mb_tokens(value: str) -> set[str]:
    return {token for token in mb_norm(value).split() if len(token) >= 3}


def mb_release_matches_album(releases: list[dict[str, Any]], album: str) -> bool:
    album_norm = mb_norm(album)
    if not album_norm:
        return False
    for release in releases:
        release_norm = mb_norm(clean_value(release.get("title")))
        if release_norm and (release_norm == album_norm or album_norm in release_norm):
            return True
    return False


def mb_release_score(release: dict[str, Any], album: str) -> float:
    value = 0.0
    release_norm = mb_norm(clean_value(release.get("title")))
    album_norm = mb_norm(album)
    if album_norm and release_norm == album_norm:
        value += 30
    elif album_norm and release_norm and album_norm in release_norm:
        value += 15
    status = clean_value(release.get("status")).casefold()
    if status == "official":
        value += 20
    elif status == "pseudo-release":
        value -= 20
    if clean_value(release.get("date")):
        value += 2
    return value


def mb_recording_matches_artist(recording: dict[str, Any], artist: str) -> bool:
    artist_tokens = mb_tokens(artist)
    if not artist_tokens:
        return False
    candidate_names = []
    for credit in recording.get("artist-credit") or []:
        artist_obj = credit.get("artist") or {}
        candidate_names.append(clean_value(credit.get("name")))
        candidate_names.append(clean_value(artist_obj.get("name")))
        candidate_names.append(clean_value(artist_obj.get("sort-name")))
    candidate_tokens = mb_tokens(" ".join(candidate_names))
    return bool(artist_tokens & candidate_tokens)


def mb_best_recording(
    title: str,
    artist: str,
    album: str,
    duration: int | float | None,
) -> tuple[dict[str, Any], dict[str, Any] | None] | None:
    queries = []
    if title and artist and album:
        queries.append(
            f'recording:"{mb_phrase(title)}" AND artist:"{mb_phrase(artist)}" AND release:"{mb_phrase(album)}"'
        )
    if title and artist:
        queries.append(f'recording:"{mb_phrase(title)}" AND artist:"{mb_phrase(artist)}"')
    if title:
        queries.append(f'recording:"{mb_phrase(title)}"')

    duration_ms = int(float(duration) * 1000) if duration else 0
    for query in queries:
        data = mb_request("recording", {"query": query, "limit": "10"})
        recordings = data.get("recordings") or []
        if not recordings:
            continue

        def score(recording: dict[str, Any]) -> float:
            releases = recording.get("releases") or []
            artist_match = mb_recording_matches_artist(recording, artist)
            album_match = mb_release_matches_album(releases, album)
            if (artist or album) and not (artist_match or album_match):
                return -999
            value = float(recording.get("score") or 0)
            length = int(recording.get("length") or 0)
            if duration_ms and length:
                value -= min(abs(length - duration_ms) / 1000, 30)
            if artist_match:
                value += 20
            if album_match:
                value += 20
            return value

        recording = max(recordings, key=score)
        if score(recording) < 60:
            continue

        release = None
        releases = recording.get("releases") or []
        if releases:
            release = max(releases, key=lambda item: mb_release_score(item, album))
        return recording, release
    return None


def musicbrainz_tags(info: dict[str, Any], title: str, artist: str, album: str) -> dict[str, str]:
    match = mb_best_recording(title, artist, album, info.get("duration"))
    if not match:
        return {}

    recording, release_hint = match
    tags: dict[str, str] = {
        "musicbrainz_trackid": clean_value(recording.get("id")),
    }

    rec_artists, rec_artist_sort, rec_artist_ids = mb_artist_credit(recording.get("artist-credit") or [])
    if rec_artists:
        tags["artists"] = rec_artists
    if rec_artist_sort:
        tags["artistsort"] = rec_artist_sort
    if rec_artist_ids:
        tags["musicbrainz_artistid"] = rec_artist_ids

    release_id = clean_value((release_hint or {}).get("id"))
    if release_id:
        tags["musicbrainz_albumid"] = release_id
        time.sleep(1)
        release = mb_request(
            f"release/{release_id}",
            {"inc": "artist-credits+labels+media+recordings+release-groups"},
        )
        tags["album"] = clean_value(release.get("title"))
        tags["date"] = clean_value(release.get("date"))
        tags["releasecountry"] = clean_value(release.get("country"))
        tags["barcode"] = clean_value(release.get("barcode"))
        tags["releasestatus"] = clean_value(release.get("status"))

        album_artists, album_artist_sort, album_artist_ids = mb_artist_credit(
            release.get("artist-credit") or []
        )
        if album_artists:
            tags["album_artist"] = album_artists
        if album_artist_sort:
            tags["albumartistsort"] = album_artist_sort
        if album_artist_ids:
            tags["musicbrainz_albumartistid"] = album_artist_ids

        release_group = release.get("release-group") or {}
        release_group_id = clean_value(release_group.get("id"))
        if release_group_id:
            tags["musicbrainz_releasegroupid"] = release_group_id
        first_release_date = clean_value(release_group.get("first-release-date"))
        if first_release_date:
            tags["originaldate"] = first_release_date
            tags["originalyear"] = first_release_date[:4]
        release_types = [clean_value(release_group.get("primary-type"))]
        release_types += [clean_value(item) for item in release_group.get("secondary-types") or []]
        release_types = [item.lower() for item in release_types if item]
        if release_types:
            tags["releasetype"] = ";".join(release_types)

        labels = release.get("label-info") or []
        label_names = [
            clean_value((item.get("label") or {}).get("name"))
            for item in labels
            if clean_value((item.get("label") or {}).get("name"))
        ]
        catalog_numbers = [
            clean_value(item.get("catalog-number"))
            for item in labels
            if clean_value(item.get("catalog-number"))
        ]
        if label_names:
            tags["label"] = "; ".join(label_names)
        if catalog_numbers:
            tags["catalognumber"] = "; ".join(catalog_numbers)

        media = release.get("media") or []
        tags["disctotal"] = str(len(media))
        for medium in media:
            tracks = medium.get("tracks") or []
            for track_item in tracks:
                track_recording = track_item.get("recording") or {}
                if clean_value(track_recording.get("id")) != tags["musicbrainz_trackid"]:
                    continue
                tags["track"] = clean_value(track_item.get("number")) or str(
                    track_item.get("position") or ""
                )
                tags["tracktotal"] = str(medium.get("track-count") or len(tracks))
                tags["disc"] = str(medium.get("position") or "")
                return {key: value for key, value in tags.items() if value}

    return {key: value for key, value in tags.items() if value}


def build_picture_block(image_path: Path, width: int = 0, height: int = 0) -> str:
    data = image_path.read_bytes()
    mime = b"image/jpeg"
    desc = b"Cover (front)"
    block = b"".join(
        [
            struct.pack(">I", 3),
            struct.pack(">I", len(mime)),
            mime,
            struct.pack(">I", len(desc)),
            desc,
            struct.pack(">I", width),
            struct.pack(">I", height),
            struct.pack(">I", 24),
            struct.pack(">I", 0),
            struct.pack(">I", len(data)),
            data,
        ]
    )
    return base64.b64encode(block).decode("ascii")


def media_dimensions(path: Path) -> tuple[int, int]:
    result = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            str(path),
        ],
        capture=True,
    )
    width, height = result.stdout.strip().split(",", 1)
    return int(width), int(height)


def square_cover(image_path: Path, size: int = 1000) -> Path:
    output_path = image_path.with_name(f"{image_path.stem}.tmp-square.jpg")
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(image_path),
            "-vf",
            f"crop='min(iw,ih)':'min(iw,ih)',scale={size}:{size}",
            "-frames:v",
            "1",
            "-update",
            "1",
            "-q:v",
            "2",
            str(output_path),
        ]
    )
    output_path.replace(image_path)
    return image_path


_YT_DLP_COMMON_FLAGS: list[str] = []


def configure_yt_dlp_flags(
    *,
    cookies_from_browser: str | None,
    cookies_file: str | None,
    sleep_requests: float,
) -> None:
    """Set process-wide yt-dlp flags (cookies, throttling) that prefix every call."""
    _YT_DLP_COMMON_FLAGS.clear()
    if cookies_from_browser:
        _YT_DLP_COMMON_FLAGS.extend(["--cookies-from-browser", cookies_from_browser])
    if cookies_file:
        _YT_DLP_COMMON_FLAGS.extend(["--cookies", cookies_file])
    if sleep_requests > 0:
        _YT_DLP_COMMON_FLAGS.extend(["--sleep-requests", str(sleep_requests)])


def yt_dlp_cmd(*extra: str) -> list[str]:
    return ["yt-dlp", *_YT_DLP_COMMON_FLAGS, *extra]


def get_info(url: str) -> dict[str, Any]:
    result = run(yt_dlp_cmd("--skip-download", "--dump-single-json", url), capture=True)
    return json.loads(result.stdout)


_VIDEO_ID_RE = re.compile(r"(?:[?&]v=|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_-]{11})")


def video_id_from_url(url: str) -> str | None:
    match = _VIDEO_ID_RE.search(url)
    return match.group(1) if match else None


_TITLE_NOISE_PATTERNS = [
    # Presentation-only annotations: bracket contents containing any of these
    # keywords are stripped, regardless of what other words sit next to them.
    # Catches "(Official Video Remastered)", "(Music Video Remastered 2011)",
    # "(Lyric Video HD)", etc. without needing one regex per combination.
    r"\([^()]*\b(?:official|music\s*video|mv|m/v|lyric\s*video|lyrics?|audio|visualizer|remaster(?:ed)?|hd|hq)\b[^()]*\)",
    r"\[[^\[\]]*\b(?:official|music\s*video|mv|m/v|lyric\s*video|lyrics?|audio|visualizer|remaster(?:ed)?|hd|hq)\b[^\[\]]*\]",
    # Loop / pad uploads — same audio padded to 10 min / 1 hour. Real audio
    # variants — (Remix), (Slowed), (Sped Up), (Live) — stay outside this list.
    r"\[\s*extended\s*(?:version|mix|cut|edit)?\s*\]",
    r"\(\s*extended\s*(?:version|mix|cut|edit)?\s*\)",
    r"\[\s*\d+\s*(?:hours?|h|minutes?|mins?|m)\s*(?:loop|version|mix)?\s*\]",
    r"\(\s*\d+\s*(?:hours?|h|minutes?|mins?|m)\s*(?:loop|version|mix)?\s*\)",
    r"\[\s*loop(?:ed)?\s*\]",
    r"\(\s*loop(?:ed)?\s*\)",
    # Featured-artist sections: "(feat. Wizkid & Kyla)" / "[ft. Drake]" /
    # standalone " ft. Wizkid & Kyla". These are credit metadata, not
    # different audio. We strip them so two uploads of the same song that
    # disagree on whether to mention features dedup correctly.
    r"\(\s*(?:feat|ft|featuring|with)\b\.?[^)]*\)",
    r"\[\s*(?:feat|ft|featuring|with)\b\.?[^\]]*\]",
    r"\s+(?:feat|ft|featuring|with)\b\.?[^()\[\]]*$",
    r"\s+-\s*topic\s*$",
    r"\s*\|\s*[^|]*$",  # trailing " | Channel Name"
]
_TITLE_NOISE_RE = re.compile("|".join(_TITLE_NOISE_PATTERNS), flags=re.IGNORECASE)
_TITLE_PUNCT_RE = re.compile(r"[^\w\s]+")


def normalize_title(title: str) -> str:
    """Collapse common YouTube title decoration so re-uploads of the same
    song hash to the same key.

    Strips ``(Official Music Video Remastered)``-style brackets, ``[Lyrics]``,
    feat-credit sections, ``[Extended]``/``[N-hour-loop]`` pads, ``- Topic``,
    trailing channel names, and all punctuation; collapses whitespace and
    lowercases. Token order is preserved — ``title_dedup_key`` does the
    order-invariant pass.
    """
    if not title:
        return ""
    s = title.casefold().strip()
    s = _TITLE_NOISE_RE.sub("", s)
    s = _TITLE_PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def title_dedup_key(title: str) -> str:
    """Order-invariant dedup key derived from ``normalize_title``.

    Two uploads that disagree on artist-song token order ("Drake - One Dance"
    vs "One Dance - Drake") still hash to the same key. Real audio variants
    that contribute extra tokens — ``Live Aid 1985``, ``Slowed Reverb``,
    ``Reading Festival 2022`` — produce different keys and stay separate.
    """
    norm = normalize_title(title)
    if not norm:
        return ""
    tokens = sorted(set(t for t in norm.split() if len(t) >= 2))
    return " ".join(tokens)


def find_existing_by_id(out_dir: Path, video_id: str) -> Path | None:
    """Return the on-disk .opus file for ``video_id`` if it already exists."""
    if not out_dir.exists():
        return None
    suffix = f"[{video_id}].opus"
    for path in out_dir.iterdir():
        if path.name.endswith(suffix):
            return path
    return None


def expand_input(
    raw: str,
    search_limit: int,
    *,
    min_duration: float | None = None,
    max_duration: float | None = None,
) -> list[tuple[str, str]]:
    """Expand a search query / playlist URL into ``(url, title)`` tuples.

    - Plain video URL → returned as-is (its duration is checked later in
      ``process_url`` once full metadata is fetched). Title is empty.
    - ``ytsearch...:query`` or bare query (no scheme) → top N search results.
    - Playlist URL (``list=...``) → all entries.

    When ``min_duration`` / ``max_duration`` are set and the flat-playlist
    response includes a ``duration`` per entry, out-of-range entries are
    dropped here so we never even open the page. Entries with missing
    duration pass through and get re-checked in ``process_url``. The
    returned ``title`` (when present) lets the caller dedupe across
    re-uploads via ``normalize_title``.
    """
    raw = raw.strip()
    if not raw or raw.startswith("#"):
        return []

    is_url = raw.startswith(("http://", "https://"))
    is_search = raw.startswith("ytsearch")

    if is_url and "list=" in raw and "/watch?" not in raw:
        target = raw
    elif is_url:
        return [(raw, "")]
    elif is_search:
        target = raw
    else:
        target = f"ytsearch{search_limit}:{raw}"

    try:
        result = run(
            yt_dlp_cmd("--flat-playlist", "--dump-single-json", target),
            capture=True,
        )
        data = json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"warning: failed to expand {target!r}: {exc}", file=sys.stderr)
        return []

    out: list[tuple[str, str]] = []
    for entry in data.get("entries") or []:
        vid = clean_value(entry.get("id"))
        if not vid:
            continue
        duration = entry.get("duration")
        if isinstance(duration, (int, float)):
            if min_duration is not None and duration < min_duration:
                continue
            if max_duration is not None and duration > max_duration:
                continue
        title = clean_value(entry.get("title"))
        out.append((f"https://www.youtube.com/watch?v={vid}", title))
    return out


def process_url(url: str, args: argparse.Namespace) -> str:
    """Download one video. Returns ``downloaded``, ``skipped`` or ``failed``."""
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    early_id = video_id_from_url(url)
    if early_id and not args.overwrite:
        existing = find_existing_by_id(out_dir, early_id)
        if existing:
            print(f"skip: {existing.name}")
            return "skipped"

    try:
        info = get_info(url)
    except subprocess.CalledProcessError:
        print(f"failed: cannot fetch info for {url}", file=sys.stderr)
        return "failed"

    duration = info.get("duration")
    if isinstance(duration, (int, float)):
        if args.max_duration > 0 and duration > args.max_duration:
            mins = duration / 60
            print(f"skip: too long ({mins:.1f} min > {args.max_duration / 60:.1f} min)")
            return "skipped"
        if args.min_duration > 0 and duration < args.min_duration:
            print(f"skip: too short ({duration:.0f} s)")
            return "skipped"

    video_id = clean_value(info.get("id")) or early_id or "unknown"
    description = clean_value(info.get("description"))
    raw_title = clean_value(info.get("title"))

    artist = clean_value(info.get("artist"))
    title = clean_value(info.get("track")) or raw_title
    if not artist:
        parsed_artist, parsed_title = infer_from_title(raw_title)
        artist = parsed_artist
        title = parsed_title

    album = clean_value(info.get("album"))
    album_artist = artist.split(",")[0].strip() if artist else ""
    date = release_date(info, description)
    genre = args.genre or infer_genre(info, title, album, artist)
    copyright_tag = copyright_line(description)
    existing_tags = {
        "title": title,
        "artist": artist,
        "album_artist": album_artist,
        "album": album,
        "track": "1",
        "genre": genre,
        "date": date,
        "copyright": copyright_tag,
    }
    mb_tags: dict[str, str] = {}
    if args.musicbrainz:
        try:
            mb_tags = musicbrainz_tags(info, title, artist, album)
            if mb_tags:
                print(f"musicbrainz: matched {mb_tags.get('musicbrainz_trackid', 'recording')}")
            else:
                print("musicbrainz: no confident match")
        except (TimeoutError, urllib.error.URLError, json.JSONDecodeError, KeyError, ValueError) as exc:
            print(f"warning: MusicBrainz lookup failed: {exc}", file=sys.stderr)

    stem_artist = artist or album_artist or "Unknown Artist"
    stem = safe_filename(f"{stem_artist} - {title}")
    webm_path = out_dir / f"{stem} [{video_id}].webm"
    opus_path = out_dir / f"{stem} [{video_id}].opus"
    cover_prefix = str(out_dir / f"{stem} [{video_id}] cover")

    if opus_path.exists() and not args.overwrite:
        print(f"skip: {opus_path.name}")
        return "skipped"

    print(f"metadata: {title} - {artist or 'unknown artist'}")
    try:
        run(
            yt_dlp_cmd(
                "-f",
                "251",
                "-o",
                str(out_dir / f"{stem} [{video_id}].%(ext)s"),
                url,
            )
        )
    except subprocess.CalledProcessError:
        print(f"failed: yt-dlp download for {url}", file=sys.stderr)
        return "failed"

    cover_path: Path | None = None
    if not args.no_cover:
        try:
            run(
                yt_dlp_cmd(
                    "--skip-download",
                    "--write-thumbnail",
                    "--convert-thumbnails",
                    "jpg",
                    "-o",
                    f"{cover_prefix}.%(ext)s",
                    url,
                )
            )
            cover_path = Path(f"{cover_prefix}.jpg")
            if cover_path.exists():
                cover_path = square_cover(cover_path)
            else:
                cover_path = None
        except subprocess.CalledProcessError:
            print("warning: cover download failed; continuing without cover", file=sys.stderr)
            cover_path = None

    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(webm_path),
        "-map_metadata",
        "-1",
        "-vn",
        "-map",
        "0:a:0",
        "-c:a",
        "copy",
        "-f",
        "opus",
        "-metadata:s:a:0",
        f"title={title}",
    ]
    if artist:
        ffmpeg_cmd += ["-metadata:s:a:0", f"artist={artist}"]
    if album_artist:
        ffmpeg_cmd += ["-metadata:s:a:0", f"album_artist={album_artist}"]
    if album:
        ffmpeg_cmd += ["-metadata:s:a:0", f"album={album}"]
    ffmpeg_cmd += ["-metadata:s:a:0", "track=1"]
    if genre:
        ffmpeg_cmd += ["-metadata:s:a:0", f"genre={genre}"]
    if date:
        ffmpeg_cmd += ["-metadata:s:a:0", f"date={date}"]
    if copyright_tag:
        ffmpeg_cmd += ["-metadata:s:a:0", f"copyright={copyright_tag}"]
    for key, value in mb_tags.items():
        if key in existing_tags and existing_tags[key]:
            continue
        ffmpeg_cmd += ["-metadata:s:a:0", f"{key}={value}"]

    source_note = "Auto-generated by YouTube."
    if "Provided to YouTube by" in description:
        provider = description.splitlines()[0].strip()
        source_note = f"{provider}. Auto-generated by YouTube."
    elif description:
        source_note = "Source: YouTube Music. Metadata inferred where needed."
    ffmpeg_cmd += ["-metadata:s:a:0", f"description={source_note}"]

    if cover_path:
        try:
            cover_width, cover_height = media_dimensions(cover_path)
            ffmpeg_cmd += [
                "-metadata",
                f"METADATA_BLOCK_PICTURE={build_picture_block(cover_path, cover_width, cover_height)}",
            ]
        except (subprocess.CalledProcessError, ValueError) as exc:
            print(f"warning: cover embed skipped: {exc}", file=sys.stderr)

    ffmpeg_cmd.append(str(opus_path))
    try:
        run(ffmpeg_cmd)
    except subprocess.CalledProcessError:
        print(f"failed: ffmpeg encode for {url}", file=sys.stderr)
        webm_path.unlink(missing_ok=True)
        return "failed"

    try:
        run(["ffmpeg", "-v", "error", "-i", str(opus_path), "-f", "null", "-"])
    except subprocess.CalledProcessError:
        print(f"warning: opus integrity check failed for {opus_path.name}", file=sys.stderr)

    if not args.keep_webm:
        webm_path.unlink(missing_ok=True)
    print(f"done: {opus_path.name}")
    return "downloaded"


def cleanup_too_long(out_dir: Path, max_duration: float) -> None:
    """Remove any *.opus files in ``out_dir`` whose duration exceeds the limit.

    Useful after a run that pre-dated the duration filter (e.g. left a stray
    3-hour DJ-mix file). Walks the directory, probes each file with ffprobe,
    deletes those above ``max_duration`` seconds. No-op when ``max_duration``
    is 0 or the directory does not exist.
    """
    if max_duration <= 0 or not out_dir.exists():
        return
    deleted = 0
    inspected = 0
    for path in sorted(out_dir.glob("*.opus")):
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
        except subprocess.CalledProcessError:
            continue
        inspected += 1
        try:
            duration = float(result.stdout.strip())
        except ValueError:
            continue
        if duration > max_duration:
            print(f"removing {path.name} ({duration / 60:.1f} min)")
            path.unlink(missing_ok=True)
            deleted += 1
    print(f"cleanup: removed {deleted} of {inspected} files (limit {max_duration / 60:.1f} min)")


def _resolution_cache_key(
    raw: str, search_limit: int, min_d: float | None, max_d: float | None
) -> str:
    """Hash that invalidates only when the resolution semantics change.

    Same query under same search-limit + duration filters → same cache hit.
    Bumping search-limit or changing the duration cap busts that key cleanly.
    """
    payload = f"{raw}\x00{search_limit}\x00{min_d}\x00{max_d}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _load_resolution_cache(path: Path) -> dict[str, list[dict[str, str]]]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: cache at {path} unreadable ({exc}); starting fresh", file=sys.stderr)
        return {}


def _save_resolution_cache(path: Path, cache: dict[str, list[dict[str, str]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def resolve_inputs_parallel(
    raw_inputs: list[str],
    *,
    search_limit: int,
    min_duration: float | None,
    max_duration: float | None,
    cache_path: Path,
    workers: int,
) -> list[tuple[str, str]]:
    """Expand all inputs (with on-disk cache + thread parallelism) and return
    the flat list of ``(url, title)`` tuples in the original input order.

    A keyed sidecar JSON at ``cache_path`` skips any (raw, search_limit,
    duration filters) tuple that already resolved on a previous run, so
    interrupted sessions don't lose work and adding queries only resolves
    the new ones.
    """
    cache = _load_resolution_cache(cache_path)
    pending: list[tuple[int, str, str]] = []
    cached_hits = 0
    for idx, raw in enumerate(raw_inputs):
        key = _resolution_cache_key(raw, search_limit, min_duration, max_duration)
        if key in cache:
            cached_hits += 1
            continue
        pending.append((idx, raw, key))

    if cached_hits:
        print(f"  cache hits: {cached_hits}/{len(raw_inputs)}")
    if not pending:
        print("  all queries already resolved")
    else:
        print(f"  resolving {len(pending)} new with {workers} parallel workers")

    save_lock = threading.Lock()
    save_every = 50

    def _do(raw: str) -> list[tuple[str, str]]:
        return expand_input(
            raw,
            search_limit,
            min_duration=min_duration,
            max_duration=max_duration,
        )

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_meta = {
            pool.submit(_do, raw): (idx, raw, key) for idx, raw, key in pending
        }
        for fut in concurrent.futures.as_completed(future_to_meta):
            idx, raw, key = future_to_meta[fut]
            try:
                results = fut.result()
            except Exception as exc:  # noqa: BLE001 — keep batch alive
                print(f"  resolve failed: {raw!r}: {exc}", file=sys.stderr)
                results = []
            with save_lock:
                cache[key] = [{"url": u, "title": t} for u, t in results]
                completed += 1
                if completed % save_every == 0:
                    _save_resolution_cache(cache_path, cache)
                    print(f"  resolved {completed}/{len(pending)}", flush=True)

    if pending:
        _save_resolution_cache(cache_path, cache)
        print(f"  resolved {completed}/{len(pending)} (cache saved)")

    flat: list[tuple[str, str]] = []
    for raw in raw_inputs:
        key = _resolution_cache_key(raw, search_limit, min_duration, max_duration)
        for entry in cache.get(key, []):
            flat.append((entry.get("url", ""), entry.get("title", "")))
    return flat


def collect_inputs(args: argparse.Namespace) -> list[str]:
    raw_inputs: list[str] = list(args.urls)
    for query in args.search:
        raw_inputs.append(f"ytsearch{args.search_limit}:{query}")
    if args.from_file:
        path = Path(args.from_file)
        if not path.exists():
            raise SystemExit(f"--from-file not found: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                raw_inputs.append(stripped)
    return raw_inputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download YouTube Music links to tagged .opus files."
    )
    parser.add_argument(
        "urls",
        nargs="*",
        help="Video URLs, playlist URLs, or bare search phrases.",
    )
    parser.add_argument(
        "--search",
        action="append",
        default=[],
        metavar="QUERY",
        help="Add YouTube search results for QUERY (repeatable).",
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        default=3,
        help="Number of search results per --search query (default: 3). "
        "Each search uses one network call; bigger values slow expansion. "
        "Top-1 is often a compilation/album for some queries, so 3 gives "
        "duration filter + title dedup room to land on a real song.",
    )
    parser.add_argument(
        "--from-file",
        help="File with URLs / playlist URLs / queries (one per line, # for comments).",
    )
    # Default to ``<repo_root>/data/raw/yt_music`` regardless of where the
    # script is launched from. ``Path(__file__).resolve().parent`` is
    # ``scripts/``; its parent is the repo root.
    default_out = Path(__file__).resolve().parent.parent / "data" / "raw" / "yt_music"
    parser.add_argument(
        "--out-dir",
        default=str(default_out),
        help=f"Output directory (default: {default_out}).",
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="Stop after this many unique videos resolve from the inputs.",
    )
    parser.add_argument(
        "--resolve-workers",
        type=int,
        default=4,
        help="Parallel workers for the search-resolution phase (default: 4). "
        "Each worker spawns its own yt-dlp subprocess; bigger values resolve "
        "faster but risk hitting YouTube anti-bot blocks (which require cookies "
        "to recover).",
    )
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        metavar="BROWSER",
        help="Pass cookies from a logged-in browser to yt-dlp. Supported: "
        "chrome, chromium, firefox, safari, brave, edge, opera, vivaldi, whale. "
        "Arc isn't directly supported; export cookies via 'Get cookies.txt "
        "LOCALLY' extension and use --cookies instead. Recommended for bulk "
        "runs: without cookies YouTube triggers 'Sign in to confirm you're "
        "not a bot' after ~30-50 anonymous requests.",
    )
    parser.add_argument(
        "--cookies",
        default=None,
        metavar="FILE",
        help="Path to a Netscape-format cookies.txt file (alternative to "
        "--cookies-from-browser). Use this when the browser isn't directly "
        "supported by yt-dlp (e.g. Arc): install a 'Get cookies.txt' "
        "extension, log in to YouTube, export the cookies file, point here.",
    )
    parser.add_argument(
        "--sleep-requests",
        type=float,
        default=0.0,
        help="Pass --sleep-requests SEC to yt-dlp (delay between requests). "
        "Use 1-2 for bulk runs to reduce anti-bot trigger rate.",
    )
    parser.add_argument(
        "--resolve-cache",
        type=str,
        default=None,
        help="JSON cache for resolved URLs (default: <out_dir>/.resolved_urls.json). "
        "Re-runs hit the cache for queries that haven't changed; only new / "
        "modified queries hit the network. Set to '' to disable.",
    )
    parser.add_argument(
        "--resolve-only",
        action="store_true",
        help="Only resolve queries to URLs (filling the cache); skip downloads. "
        "Useful for splitting the slow expand phase from the slow download phase.",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=60.0,
        help="Drop videos shorter than this many seconds (default: 60). 0 disables.",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=600.0,
        help="Drop videos longer than this many seconds (default: 600 = 10 min). "
        "Filters out hour-long DJ mixes, full albums, 'best of' compilations, "
        "and most ``[Extended]`` / ``[10 MINUTE]`` looped re-uploads. Long prog "
        "tracks (e.g. Pink Floyd 'Echoes' at 23 min) won't pass — bump if you "
        "need them. 0 disables.",
    )
    parser.add_argument("--genre", help="Override genre for all URLs")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .opus files")
    parser.add_argument("--keep-webm", action="store_true", help="Keep downloaded source .webm")
    parser.add_argument("--no-cover", action="store_true", help="Do not download/embed cover art")
    parser.add_argument("--musicbrainz", action="store_true", help="Add Picard-style tags from MusicBrainz")
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Before downloading, scan --out-dir and delete any existing .opus "
        "files longer than --max-duration. Use to drop accidental DJ-mix / "
        "compilation downloads from earlier runs.",
    )
    parser.add_argument(
        "--allow-title-duplicates",
        action="store_true",
        help="Disable normalize-title-based dedup. Default: drop entries whose "
        "stripped title (without (Official Video) / (Lyrics) / [Remastered] / "
        "trailing channel name / punctuation) matches an already-resolved one. "
        "Useful when you actually want multiple uploads of the same song.",
    )
    args = parser.parse_args()

    require_tool("yt-dlp")
    require_tool("ffmpeg")
    require_tool("ffprobe")

    configure_yt_dlp_flags(
        cookies_from_browser=args.cookies_from_browser,
        cookies_file=args.cookies,
        sleep_requests=args.sleep_requests,
    )
    if args.cookies_from_browser:
        print(f"yt-dlp cookies: browser={args.cookies_from_browser}")
    elif args.cookies:
        print(f"yt-dlp cookies: file={args.cookies}")

    raw_inputs = collect_inputs(args)
    if not raw_inputs and not args.cleanup:
        parser.error("provide at least one URL / --search / --from-file, or --cleanup")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output: {out_dir}")

    if args.cleanup:
        cleanup_too_long(out_dir, args.max_duration)
        if not raw_inputs:
            return

    min_d = args.min_duration if args.min_duration > 0 else None
    max_d = args.max_duration if args.max_duration > 0 else None
    if args.resolve_cache is None:
        cache_path = out_dir / ".resolved_urls.json"
    elif args.resolve_cache == "":
        cache_path = None
    else:
        cache_path = Path(args.resolve_cache).resolve()

    print(f"resolving {len(raw_inputs)} input(s)...")
    if cache_path is not None:
        print(f"  cache: {cache_path}")
        flat_results = resolve_inputs_parallel(
            raw_inputs,
            search_limit=args.search_limit,
            min_duration=min_d,
            max_duration=max_d,
            cache_path=cache_path,
            workers=max(1, args.resolve_workers),
        )
    else:
        # Sequential fallback when cache is explicitly disabled.
        flat_results = []
        for raw in raw_inputs:
            flat_results.extend(
                expand_input(raw, args.search_limit, min_duration=min_d, max_duration=max_d)
            )

    seen_ids: set[str] = set()
    seen_keys: set[str] = set()
    title_dupes = 0
    urls: list[str] = []
    for resolved_url, resolved_title in flat_results:
        vid = video_id_from_url(resolved_url)
        if vid is None or vid in seen_ids:
            continue
        # Token-sorted key catches both "Artist - Song" / "Song - Artist"
        # ordering and feat./ft. variants.
        dedup_key = title_dedup_key(resolved_title) if not args.allow_title_duplicates else ""
        if dedup_key and dedup_key in seen_keys:
            title_dupes += 1
            continue
        seen_ids.add(vid)
        if dedup_key:
            seen_keys.add(dedup_key)
        urls.append(resolved_url)
        if args.max_tracks is not None and len(urls) >= args.max_tracks:
            break
    print(
        f"resolved to {len(urls)} unique video(s); "
        f"deduped {title_dupes} title-duplicate re-upload(s)"
    )

    if args.resolve_only:
        print("--resolve-only set; cache populated, skipping downloads")
        return

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    for index, url in enumerate(urls, start=1):
        print(f"\n[{index}/{len(urls)}] {url}")
        try:
            outcome = process_url(url, args)
        except KeyboardInterrupt:
            print("\ninterrupted; partial summary follows", file=sys.stderr)
            break
        except Exception as exc:  # noqa: BLE001 — bulk path must keep going
            print(f"failed: {exc}", file=sys.stderr)
            outcome = "failed"
        counts[outcome] = counts.get(outcome, 0) + 1

    total = sum(counts.values())
    print(
        "\nsummary: "
        f"downloaded={counts.get('downloaded', 0)} "
        f"skipped={counts.get('skipped', 0)} "
        f"failed={counts.get('failed', 0)} "
        f"of {total}"
    )


if __name__ == "__main__":
    main()
