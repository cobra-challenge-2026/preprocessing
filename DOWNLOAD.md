# Download COBRA2026 training data

Requires **Python 3.8+**. No extra packages or API token are needed.
Open a terminal in the folder containing `download_cobra_case.py`.

## Download a center

```sh
python3 download_cobra_case.py --center A
```

Downloads every training case in center A, one at a time. Cases are unpacked into
`COBRA2026_Train/A/CASE/` (for example, `COBRA2026_Train/A/A002/`).

## Download one case

```sh
python3 download_cobra_case.py --case A002
```

The center is inferred from the case name. Running without arguments selects A002.

## Useful options

```sh
# Preview cases without downloading
python3 download_cobra_case.py --center A --dry-run

# Choose a destination directory
python3 download_cobra_case.py --center A --output-dir data/center_A

# Choose the ZIP location for a single case (unpacks beside it)
python3 download_cobra_case.py --case A002 --output data/A002.zip

# Verify existing cases without downloading or changing data files
python3 download_cobra_case.py --center A --verify-only
```

Each case is checked for file names, counts, byte sizes, and CRC integrity.
**ZIPs are deleted only after successful extraction and verification.** Keep the
small `.manifest.json` files: they allow later verification without the ZIPs.
Use `--keep-zip` to retain archives, or `--no-extract` for ZIP-only downloads.

After an interruption, rerun the same command. Verified cases are skipped.
If an existing case folder fails verification, move or remove that folder and
rerun to download/unpack a fresh copy.

A JSON report records case results; `complete: true` means every selected case
passed. Center reports are saved as `download_report.json` in the destination.
For a single case, the report is saved beside its ZIP path as `CASE.zip.report.json`.
When verifying a custom destination, repeat the same `--output-dir` or `--output`.