import argparse
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from openai import OpenAI
from mutagen import File as MutagenFile


SUPPORTED_EXTENSIONS = {".mp3", ".m4a", ".aac", ".flac", ".wav", ".ogg"}
DEFAULT_TARGET_MINUTES = 120


@dataclass
class TrackInfo:
    path: Path
    title: str
    artist: str
    album: str
    duration_seconds: Optional[float]

    @property
    def display_name(self) -> str:
        if self.artist and self.title:
            return f"{self.artist} - {self.title}"
        if self.title:
            return self.title
        return self.path.stem


class PlaylistMaker:
    def __init__(
        self,
        library_root: Path,
        output_root: Path,
        client: OpenAI,
        model: str,
        target_minutes: int = DEFAULT_TARGET_MINUTES,
        max_prompt_tokens: int = 200_000,
    ) -> None:
        self.library_root = library_root
        self.output_root = output_root
        self.client = client
        self.model = model
        self.target_minutes = target_minutes
        self.max_prompt_tokens = max_prompt_tokens

    def scan_library(self) -> List[TrackInfo]:
        tracks: List[TrackInfo] = []
        for path in self.library_root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            try:
                info = self._extract_metadata(path)
            except ValueError as exc:
                print(f"Skipping invalid file {path}: {exc}", file=sys.stderr)
                continue
            tracks.append(info)
        return tracks

    def _extract_metadata(self, path: Path) -> TrackInfo:
        try:
            metadata = MutagenFile(path)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Unable to read metadata: {exc}") from exc
        title = ""
        artist = ""
        album = ""
        duration = None

        if metadata:
            duration = getattr(metadata.info, "length", None)
            tags = metadata.tags or {}
            title = self._first_tag(tags, ["TIT2", "title"]) or path.stem
            artist = self._first_tag(tags, ["TPE1", "artist"]) or ""
            album = self._first_tag(tags, ["TALB", "album"]) or ""
        else:
            title = path.stem

        return TrackInfo(
            path=path,
            title=str(title).strip(),
            artist=str(artist).strip(),
            album=str(album).strip(),
            duration_seconds=duration,
        )

    def _first_tag(self, tags: dict, keys: Sequence[str]) -> Optional[str]:
        for key in keys:
            if key in tags:
                value = tags[key]
                if isinstance(value, (list, tuple)):
                    return str(value[0])
                return str(value)
        return None

    def build_prompt(self, tracks: Sequence[TrackInfo]) -> str:
        limited_tracks = self._limit_tracks_for_prompt(tracks)
        lines = self._prompt_header_lines()
        for track in limited_tracks:
            lines.append(self._format_track_line(track))

        if len(limited_tracks) < len(tracks):
            lines.append(
                "Note: Library truncated due to rate limits; choose from the provided sample only."
            )

        return "\n".join(lines)

    def request_playlist(self, tracks: Sequence[TrackInfo]) -> List[str]:
        prompt = self.build_prompt(tracks)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        message = response.choices[0].message.content or ""
        return self._parse_playlist_lines(message)

    def _limit_tracks_for_prompt(self, tracks: Sequence[TrackInfo]) -> Sequence[TrackInfo]:
        shuffled_tracks = list(tracks)
        random.shuffle(shuffled_tracks)

        header_lines = self._prompt_header_lines()
        header = "\n".join(header_lines)
        selected: List[TrackInfo] = []

        for track in shuffled_tracks:
            line = self._format_track_line(track)
            new_total = self._estimate_tokens("\n".join([header, *[self._format_track_line(t) for t in selected], line]))
            if new_total > self.max_prompt_tokens:
                break
            selected.append(track)

        return selected

    def _prompt_header_lines(self) -> List[str]:
        return [
            "You are a playlist curator tasked with building a 2-hour playlist for swimming.",
            "Pick songs from the provided library only. Return the playlist as a numbered list with artist and title.",
            "Keep the total duration as close to 120 minutes as possible and avoid duplicates.",
            "Library:",
        ]

    def _format_track_line(self, track: TrackInfo) -> str:
        duration_text = f"{int(track.duration_seconds)}s" if track.duration_seconds else "unknown"
        return f"- {track.display_name} (album: {track.album or 'unknown'}, duration: {duration_text})"

    def _estimate_tokens(self, text: str) -> int:
        return max(1, (len(text) + 3) // 4)

    def _parse_playlist_lines(self, content: str) -> List[str]:
        entries: List[str] = []
        for line in content.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            cleaned = re.sub(r"^[0-9]+[.)]\s*", "", cleaned)
            entries.append(cleaned)
        return entries

    def select_tracks(self, desired: Sequence[str], tracks: Sequence[TrackInfo]) -> List[TrackInfo]:
        remaining_seconds = self.target_minutes * 60
        selected: List[TrackInfo] = []
        available = list(tracks)

        for desired_entry in desired:
            match = self._match_track(desired_entry, available)
            if not match:
                continue
            if match.duration_seconds and match.duration_seconds > remaining_seconds + 60:
                continue
            selected.append(match)
            remaining_seconds -= match.duration_seconds or 0
            available.remove(match)
            if remaining_seconds <= 0:
                break
        return selected

    def _match_track(self, desired: str, candidates: Sequence[TrackInfo]) -> Optional[TrackInfo]:
        normalized_desired = desired.lower()
        for track in candidates:
            name = track.display_name.lower()
            if normalized_desired in name or name in normalized_desired:
                return track
            if track.title and track.title.lower() in normalized_desired:
                return track
        return None

    def copy_playlist(self, tracks: Iterable[TrackInfo]) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        for track in tracks:
            destination = self.output_root / track.path.name
            shutil.copy2(track.path, destination)

    def preview_playlist(self, tracks: Sequence[TrackInfo], preview_file: Path) -> None:
        preview_lines = ["Playlist Preview:"]
        total_seconds = 0.0
        for idx, track in enumerate(tracks, start=1):
            duration = track.duration_seconds or 0
            total_seconds += duration
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            preview_lines.append(
                f"{idx}. {track.display_name} ({minutes:02d}:{seconds:02d}) - from {track.album or 'Unknown'}"
            )
        preview_lines.append(f"Total duration: {int(total_seconds // 60)} min {int(total_seconds % 60)} sec")
        preview_file.write_text("\n".join(preview_lines), encoding="utf-8")
        print(f"Saved playlist preview to {preview_file}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a 2-hour swimming playlist using ChatGPT")
    parser.add_argument("library", type=Path, help="Root folder containing music files")
    parser.add_argument("output", type=Path, help="Destination folder (e.g., the Shokz drive)")
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model to use for playlist curation",
    )
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY"), help="OpenAI API key")
    parser.add_argument(
        "--preview-file",
        type=Path,
        default=Path("playlist_preview.txt"),
        help="Path to save the playlist preview",
    )
    parser.add_argument(
        "--no-copy",
        action="store_true",
        help="Do not copy files, only generate the preview",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.api_key:
        print("Missing OpenAI API key. Set --api-key or OPENAI_API_KEY.", file=sys.stderr)
        return 1

    client = OpenAI(api_key=args.api_key)
    maker = PlaylistMaker(args.library, args.output, client, model=args.model)

    print("Scanning library...")
    tracks = maker.scan_library()
    if not tracks:
        print("No supported audio files found.", file=sys.stderr)
        return 1
    print(f"Found {len(tracks)} tracks. Requesting playlist...")

    playlist_entries = maker.request_playlist(tracks)
    selected_tracks = maker.select_tracks(playlist_entries, tracks)

    if not selected_tracks:
        print("ChatGPT did not return any matching tracks.", file=sys.stderr)
        return 1

    maker.preview_playlist(selected_tracks, args.preview_file)
    if not args.no_copy:
        maker.copy_playlist(selected_tracks)
        print(f"Copied {len(selected_tracks)} files to {args.output}")
    else:
        print("Preview generated; skipping copy as requested.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
