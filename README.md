# swimplaylistmaker

A command-line helper to curate a 2-hour swimming playlist from your music library. The tool scans every subfolder for compatible audio files, asks ChatGPT to propose a playlist, saves a preview, and then copies the selected tracks to a destination such as a Shokz OpenSwim Pro drive.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Provide your OpenAI API key via the `OPENAI_API_KEY` environment variable or `--api-key` flag.

## Usage

```bash
python -m playlistmaker.cli /path/to/library /path/to/shokz \
  --model gpt-4o-mini \
  --preview-file playlist_preview.txt
```

### Arguments
- `library`: Root folder containing your music files. All subdirectories are scanned.
- `output`: Destination folder for the playlist files (e.g., the mounted Shokz drive).
- `--model`: OpenAI model to use for playlist curation (default: `gpt-4o-mini`).
- `--api-key`: OpenAI API key. Defaults to `OPENAI_API_KEY`.
- `--preview-file`: Path for the saved playlist preview (defaults to `playlist_preview.txt`).
- `--no-copy`: Only generate the preview without copying files.

## How it works

1. Scans the library for supported formats (`.mp3`, `.m4a`, `.aac`, `.flac`, `.wav`, `.ogg`).
2. Extracts basic metadata (title, artist, album, duration) using `mutagen`.
3. Sends the library summary to ChatGPT to request a 2-hour playlist.
4. Matches ChatGPT suggestions to your files, preferring title and artist matches.
5. Saves a playlist preview (so every run has a preview) and copies the selected files unless `--no-copy` is set.

## Notes

- The tool keeps the playlist as close to 2 hours as possible based on available duration metadata.
- A preview file is always created so you can review the selections before or after copying.
