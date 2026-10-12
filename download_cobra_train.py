#!/usr/bin/env python3
"""Download, verify and unpack COBRA2026 training data, case by case."""

import argparse
from collections import Counter
from http.client import HTTPException
import json
from pathlib import Path
import re
import shutil
import tempfile
import zlib
import ssl
import sys
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen
import zipfile

API = "https://tomoradio-warehouse.creatis.insa-lyon.fr/api/v1"
COLLECTION_ID = "696f75f74c0b0d3d4bc7019d"
TLS = ssl.create_default_context()
# Some macOS Python installations have no configured default CA bundle.
if (sys.platform == "darwin" and ssl.get_default_verify_paths().cafile is None
        and Path("/etc/ssl/cert.pem").is_file()):
    TLS.load_verify_locations("/etc/ssl/cert.pem")
ERRORS = (URLError, OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile, HTTPException, EOFError)


def get_json(endpoint, **params):
    url = f"{API}/{endpoint}?{urlencode(params)}"
    with urlopen(url, timeout=60, context=TLS) as response:
        return json.load(response)


def list_all(endpoint, **params):
    """Paginate: center A has more cases than Girder's default page size."""
    result = []
    while True:
        page = get_json(endpoint, limit=100, offset=len(result), sort="_id",
                        sortdir=1, **params)
        result.extend(page)
        if len(page) < 100:
            return result


def child_folder(parent_id, parent_type, name):
    folders = get_json("folder", parentId=parent_id,
                       parentType=parent_type, name=name, limit=2)
    if len(folders) != 1:
        raise ValueError(f"Folder {name!r} not found uniquely under {parent_id}.")
    return folders[0]


def safe_name(name):
    if not name or name in (".", "..") or "/" in name or "\\" in name:
        raise ValueError(f"Unexpected server filename: {name!r}")
    return name


def inventory(folder, prefix=""):
    """Build expected ZIP paths and byte sizes from live file metadata."""
    prefix += safe_name(folder["name"]) + "/"
    expected = []
    for item in list_all("item", folderId=folder["_id"]):
        files = list_all(f"item/{item['_id']}/files")
        # Girder omits the item directory for a single file named like its item.
        item_prefix = prefix
        if len(files) != 1 or files[0]["name"] != item["name"]:
            item_prefix += safe_name(item["name"]) + "/"
        for file in files:
            expected.append((item_prefix + safe_name(file["name"]), int(file["size"])))
    for child in list_all("folder", parentType="folder", parentId=folder["_id"]):
        expected.extend(inventory(child, prefix))
    return expected


def verify_zip(path, expected):
    """Require the exact remote file inventory, sizes, and valid ZIP CRCs."""
    if not expected:
        raise ValueError("Remote case contains no files; refusing to mark it complete.")
    with zipfile.ZipFile(path) as archive:
        actual = [(entry.filename, entry.file_size) for entry in archive.infolist()
                  if not entry.is_dir()]
        missing = Counter(expected) - Counter(actual)
        extra = Counter(actual) - Counter(expected)
        if missing or extra:
            raise ValueError(f"ZIP inventory mismatch: missing/wrong-size={list(missing)}, "
                             f"unexpected/wrong-size={list(extra)}")
        bad_file = archive.testzip()  # Reads every entry, checks decompression and CRC.
        if bad_file:
            raise ValueError(f"ZIP CRC check failed: {bad_file}")
    return {"files": len(expected), "uncompressed_bytes": sum(size for _, size in expected)}


def download_case(case, output, expected):
    partial = output.with_name(output.name + ".part")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Folder ZIP streams cannot be resumed. Discard only an old temporary download.
    partial.unlink(missing_ok=True)
    downloaded = 0
    url = f"{API}/folder/{case['_id']}/download"
    try:
        with urlopen(url, timeout=120, context=TLS) as response, partial.open("xb") as target:
            length = response.headers.get("Content-Length")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)
                downloaded += len(chunk)
                print(f"\rDownloaded {downloaded / 1_000_000:.1f} MB", end="", flush=True)
        print("\nChecking file inventory, byte sizes and ZIP CRCs...")
        if length is not None and downloaded != int(length):
            raise ValueError(f"Incomplete HTTP response: {downloaded} of {length} bytes")
        details = verify_zip(partial, expected)
        partial.replace(output)  # Keep an existing invalid ZIP until its replacement passes.
        return details
    finally:
        partial.unlink(missing_ok=True)



def manifest_path(archive_path):
    return archive_path.with_name(archive_path.name + ".manifest.json")


def finalize_extraction(archive_path, directory, expected, keep_zip=False):
    """Persist ZIP CRCs for future audits before removing the verified archive."""
    verify_extracted(directory, archive_path, expected)
    with zipfile.ZipFile(archive_path) as archive:
        records = [{"path": name, "size": size, "crc32": archive.getinfo(name).CRC}
                   for name, size in expected]
    manifest = manifest_path(archive_path)
    temporary = manifest.with_name(manifest.name + ".tmp")
    try:
        temporary.write_text(json.dumps({"version": 1, "files": records}, indent=2) + "\n",
                             encoding="utf-8")
        temporary.replace(manifest)
    finally:
        temporary.unlink(missing_ok=True)
    if not keep_zip:
        archive_path.unlink()
        print(f"Deleted verified ZIP: {archive_path}")


def verify_extracted(directory, archive_path, expected):
    """Check extracted files against a verified ZIP or its saved checksum manifest."""
    if not expected:
        raise ValueError("Remote case contains no files.")
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"Missing case directory (or symlink): {directory}")
    actual = []
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Unexpected symlink: {path}")
        if path.is_file():
            name = directory.name + "/" + path.relative_to(directory).as_posix()
            actual.append((name, path.stat().st_size))
    if Counter(actual) != Counter(expected):
        raise ValueError(f"Extracted file inventory/size mismatch: {directory}")
    if archive_path.exists():
        with zipfile.ZipFile(archive_path) as archive:
            checksums = {name: archive.getinfo(name).CRC for name, _ in expected}
    else:
        manifest = json.loads(manifest_path(archive_path).read_text(encoding="utf-8"))
        records = manifest["files"]
        if Counter((record["path"], record["size"]) for record in records) != Counter(expected):
            raise ValueError("Saved verification manifest differs from the live API inventory.")
        checksums = {record["path"]: record["crc32"] for record in records}
    for name, _ in expected:
        relative = name.split("/", 1)[1]
        checksum = 0
        with (directory / relative).open("rb") as source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                checksum = zlib.crc32(chunk, checksum)
        if checksum != checksums[name]:
            raise ValueError(f"Extracted file CRC mismatch: {name}")


def unpack_case(archive_path, case_name, expected, verify_only=False):
    """Extract into a temporary directory, verify, then publish the case folder."""
    directory = archive_path.parent / safe_name(case_name)
    if directory.exists() or directory.is_symlink():
        try:
            verify_extracted(directory, archive_path, expected)
        except ERRORS as exc:
            raise ValueError(f"{exc}. Move/remove the existing case folder and rerun "
                             "to unpack a fresh copy.") from exc
        print(f"Verified extracted case: {directory}")
        return directory
    if verify_only:
        raise ValueError(f"Missing extracted case: {directory}")
    print(f"Unpacking: {directory}")
    # Extract only the inventory entries, rejecting absolute/parent-traversal paths.
    # Do not use extractall or write into an existing user directory.
    with tempfile.TemporaryDirectory(prefix=f".{case_name}.extract-",
                                     dir=archive_path.parent) as staging:
        staged_case = Path(staging) / case_name
        with zipfile.ZipFile(archive_path) as archive:
            for name, _ in expected:
                parts = name.split("/")
                if len(parts) < 2 or parts[0] != case_name:
                    raise ValueError(f"Unexpected archive path: {name!r}")
                for part in parts:
                    safe_name(part)
                target = Path(staging).joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as source, target.open("xb") as dest:
                    shutil.copyfileobj(source, dest, length=1024 * 1024)
        verify_extracted(staged_case, archive_path, expected)
        staged_case.rename(directory)
    print(f"Unpacked and verified: {directory}")
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--case", help="Single case name (default: A002)")
    selection.add_argument("--center", help="All training cases in a center, e.g. A")
    parser.add_argument("--output", type=Path, help="Single-case ZIP path (default: COBRA2026_Train/CENTER/CASE.zip)")
    parser.add_argument("--output-dir", type=Path,
                        help="Center download directory (default: COBRA2026_Train/CENTER)")
    parser.add_argument("--no-extract", action="store_true",
                        help="Keep only ZIPs; skip automatic extraction")
    parser.add_argument("--keep-zip", action="store_true",
                        help="Keep ZIPs after verified extraction (default: delete them)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="List cases without downloading")
    mode.add_argument("--verify-only", action="store_true",
                      help="Check existing ZIPs and extracted cases; download/extract nothing")
    args = parser.parse_args()
    if args.center and args.output:
        parser.error("use --output-dir with --center")
    if not args.center and args.output_dir:
        parser.error("use --output for a single case")
    case_name = (args.case or "A002").upper()
    center_name = args.center.upper() if args.center else case_name[0:1]
    if not re.fullmatch(r"[A-Z]", center_name):
        parser.error("--center must be a single letter, e.g. A")
    if not args.center and not re.fullmatch(r"[A-Z][0-9]+", case_name):
        parser.error("--case must look like A002")

    train = child_folder(COLLECTION_ID, "collection", "COBRA2026_Train")
    center = child_folder(train["_id"], "folder", center_name)
    if args.center:
        cases = list_all("folder", parentType="folder", parentId=center["_id"])
        cases.sort(key=lambda case: case["name"])
        for case in cases:
            if not re.fullmatch(center_name + r"[0-9]+", case["name"]):
                raise ValueError(f"Unexpected case folder: {case['name']!r}")
        directory = args.output_dir or Path("COBRA2026_Train") / center_name
        paths = [directory / f"{case['name']}.zip" for case in cases]
        report_path = directory / "download_report.json"
    else:
        cases = [child_folder(center["_id"], "folder", case_name)]
        paths = [args.output or Path("COBRA2026_Train") / center_name / f"{case_name}.zip"]
        report_path = paths[0].with_name(paths[0].name + ".report.json")
    if not cases:
        raise ValueError("No training cases found.")
    print(f"Training center {center_name}: {len(cases)} selected case(s)")
    if args.dry_run:
        for case, path in zip(cases, paths):
            print(f"{case['name']}: {case.get('size', 0) / 1_000_000:.1f} MB -> {path}")
        return 0

    report = {"api": API, "split": "COBRA2026_Train", "center": center_name,
              "complete": False, "cases": [
                  {"case": case["name"], "folder_id": case["_id"],
                   "path": str(path.resolve()), "status": "pending"}
                  for case, path in zip(cases, paths)]}
    try:
        for index, (case, path, entry) in enumerate(zip(cases, paths, report["cases"]), 1):
            print(f"\n[{index}/{len(cases)}] {case['name']}")
            try:
                expected = inventory(case)
                directory = path.parent / case["name"]
                if (not args.no_extract and not path.exists()
                        and manifest_path(path).exists()
                        and (directory.exists() or directory.is_symlink())):
                    verify_extracted(directory, path, expected)
                    entry.update(status="verified", extracted_path=str(directory.resolve()),
                                 files=len(expected),
                                 uncompressed_bytes=sum(size for _, size in expected))
                    print("Verified extracted case using saved checksums; skipping download.")
                    continue
                if path.exists():
                    try:
                        details = verify_zip(path, expected)
                    except ERRORS as exc:
                        if args.verify_only:
                            raise
                        print(f"Existing ZIP failed verification ({exc}); downloading again.")
                    else:
                        if not args.no_extract:
                            directory = unpack_case(path, case["name"], expected, args.verify_only)
                            if not args.verify_only:
                                finalize_extraction(path, directory, expected, args.keep_zip)
                            entry["extracted_path"] = str(directory.resolve())
                        entry.update(status="verified", **details)
                        print("Verified existing ZIP; skipping download.")
                        continue
                if args.verify_only:
                    raise ValueError(f"Missing ZIP: {path}")
                details = download_case(case, path, expected)
                if not args.no_extract:
                    directory = unpack_case(path, case["name"], expected)
                    finalize_extraction(path, directory, expected, args.keep_zip)
                    entry["extracted_path"] = str(directory.resolve())
                entry.update(status="downloaded_and_verified", **details)
                print(f"Complete: {path if args.no_extract else directory}")
            except ERRORS as exc:
                entry.update(status="failed", error=str(exc))
                print(f"FAILED: {exc}", file=sys.stderr)
            except KeyboardInterrupt:
                entry.update(status="interrupted")
                raise
    finally:
        statuses = Counter(entry["status"] for entry in report["cases"])
        report["complete"] = all(entry["status"] in ("verified", "downloaded_and_verified")
                                 for entry in report["cases"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nSummary: {dict(statuses)}\nReport: {report_path.resolve()}")
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ERRORS as exc:
        sys.exit(f"Error: {exc}")
    except KeyboardInterrupt:
        sys.exit("\nDownload cancelled. Rerun to skip verified cases and retry the rest.")