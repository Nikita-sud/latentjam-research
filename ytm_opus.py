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
import json
import re
import shutil
import struct
import subprocess
import sys
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


def get_info(url: str) -> dict[str, Any]:
    result = run(["yt-dlp", "--skip-download", "--dump-single-json", url], capture=True)
    return json.loads(result.stdout)


_VIDEO_ID_RE = re.compile(r"(?:[?&]v=|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_-]{11})")


def video_id_from_url(url: str) -> str | None:
    match = _VIDEO_ID_RE.search(url)
    return match.group(1) if match else None


def find_existing_by_id(out_dir: Path, video_id: str) -> Path | None:
    """Return the on-disk .opus file for ``video_id`` if it already exists."""
    if not out_dir.exists():
        return None
    suffix = f"[{video_id}].opus"
    for path in out_dir.iterdir():
        if path.name.endswith(suffix):
            return path
    return None


def expand_input(raw: str, search_limit: int) -> list[str]:
    """Expand a search query / playlist URL into individual video URLs.

    - Plain video URL → returned as-is.
    - ``ytsearch...:query`` or bare query (no scheme) → top N search results.
    - Playlist URL (``list=...``) → all entries.
    """
    raw = raw.strip()
    if not raw or raw.startswith("#"):
        return []

    is_url = raw.startswith(("http://", "https://"))
    is_search = raw.startswith("ytsearch")

    if is_url and "list=" in raw and "/watch?" not in raw:
        # Pure playlist URL; expand entries.
        target = raw
    elif is_url:
        return [raw]
    elif is_search:
        target = raw
    else:
        target = f"ytsearch{search_limit}:{raw}"

    try:
        result = run(
            ["yt-dlp", "--flat-playlist", "--dump-single-json", target],
            capture=True,
        )
        data = json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        print(f"warning: failed to expand {target!r}: {exc}", file=sys.stderr)
        return []

    out: list[str] = []
    for entry in data.get("entries") or []:
        vid = clean_value(entry.get("id"))
        if vid:
            out.append(f"https://www.youtube.com/watch?v={vid}")
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
            [
                "yt-dlp",
                "-f",
                "251",
                "-o",
                str(out_dir / f"{stem} [{video_id}].%(ext)s"),
                url,
            ]
        )
    except subprocess.CalledProcessError:
        print(f"failed: yt-dlp download for {url}", file=sys.stderr)
        return "failed"

    cover_path: Path | None = None
    if not args.no_cover:
        try:
            run(
                [
                    "yt-dlp",
                    "--skip-download",
                    "--write-thumbnail",
                    "--convert-thumbnails",
                    "jpg",
                    "-o",
                    f"{cover_prefix}.%(ext)s",
                    url,
                ]
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
        default=20,
        help="Number of search results per --search query.",
    )
    parser.add_argument(
        "--from-file",
        help="File with URLs / playlist URLs / queries (one per line, # for comments).",
    )
    parser.add_argument(
        "--out-dir",
        default="data/raw/yt_music",
        help="Output directory (default: data/raw/yt_music).",
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=None,
        help="Stop after this many unique videos resolve from the inputs.",
    )
    parser.add_argument("--genre", help="Override genre for all URLs")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .opus files")
    parser.add_argument("--keep-webm", action="store_true", help="Keep downloaded source .webm")
    parser.add_argument("--no-cover", action="store_true", help="Do not download/embed cover art")
    parser.add_argument("--musicbrainz", action="store_true", help="Add Picard-style tags from MusicBrainz")
    args = parser.parse_args()

    require_tool("yt-dlp")
    require_tool("ffmpeg")
    require_tool("ffprobe")

    raw_inputs = collect_inputs(args)
    if not raw_inputs:
        parser.error("provide at least one URL, --search, or --from-file")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output: {out_dir}")

    print(f"resolving {len(raw_inputs)} input(s)...")
    seen_ids: set[str] = set()
    urls: list[str] = []
    for raw in raw_inputs:
        for resolved in expand_input(raw, args.search_limit):
            vid = video_id_from_url(resolved)
            if vid is None:
                continue
            if vid in seen_ids:
                continue
            seen_ids.add(vid)
            urls.append(resolved)
            if args.max_tracks is not None and len(urls) >= args.max_tracks:
                break
        if args.max_tracks is not None and len(urls) >= args.max_tracks:
            break
    print(f"resolved to {len(urls)} unique video(s)")

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
