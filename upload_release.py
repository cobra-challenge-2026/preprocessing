import io
import os
import time
from contextlib import redirect_stdout

import girder_client

GIRDER_URL = "https://tomoradio-warehouse.creatis.insa-lyon.fr/api/v1/"
API_KEY = os.environ.get("GIRDER_API_KEY")

COLLECTION_NAME = "COBRA2026_challenge"
MAKE_COLLECTION_PUBLIC = False

SET = "train"
SET_FOLDER = {
    "test": "COBRA2026_Test",
    "train": "COBRA2026_Train",
    "val": "COBRA2026_Val",
}[SET]

CENTERS = ['A','B','C','D','F','G']  # list of centers to upload
RELEASE_ROOT = f"/data/RELEASE/{SET_FOLDER}"

# When True, files that already exist in the target folder are skipped.
# When False, existing items of the same name are deleted first and re-uploaded
# (overwrite), so you don't end up with duplicate "name (1)" items.
REUSE_EXISTING = False

# Only upload these files from each case folder (by name). Leave empty to
# upload all files in the folder.
ONLY_FILES = ['reconstruction.yaml', 'cbct_rtk.mha']


def get_client() -> girder_client.GirderClient:
    gc = girder_client.GirderClient(apiUrl=GIRDER_URL)
    if API_KEY:
        gc.authenticate(apiKey=API_KEY)
    else:
        gc.authenticate(interactive=True)
    return gc


def get_or_create_collection(gc, name, public=False):
    for coll in gc.listCollection():
        if coll["name"] == name:
            return coll["_id"]
    return gc.createCollection(name, public=public)["_id"]


def log(msg=""):
    print(msg, flush=True)


def rule(char="─", width=60):
    return char * width


def delete_existing_items(gc, folder_id, names):
    """Delete any items in the folder whose name matches one of ``names``,
    so a subsequent upload replaces rather than duplicates them. Returns the
    number of items removed."""
    removed = 0
    for name in names:
        for item in gc.listItem(folder_id, name=name):
            gc.delete(f"item/{item['_id']}")
            removed += 1
    return removed


def upload_case(gc, case_path, folder_id, reuse_existing=True, only_files=None):
    if only_files:
        patterns = [os.path.join(case_path, name) for name in only_files]
        names = list(only_files)
    else:
        patterns = [f"{case_path}/*"]
        names = [
            f for f in os.listdir(case_path)
            if os.path.isfile(os.path.join(case_path, f))
        ]

    # Overwrite mode: clear existing items first so re-uploading replaces them
    # instead of creating "name (1)" duplicates.
    if not reuse_existing:
        delete_existing_items(gc, folder_id, names)

    buf = io.StringIO()
    with redirect_stdout(buf):
        for pattern in patterns:
            gc.upload(
                pattern,
                folder_id,
                parentType="folder",
                reuseExisting=reuse_existing,
            )
    lines = buf.getvalue().splitlines()
    attempted = sum(1 for line in lines if line.startswith("Uploading Item from"))
    skipped = sum(1 for line in lines if "already exists in parent" in line)
    return attempted - skipped, skipped


def main():
    start = time.time()

    gc = get_client()

    collection_id = get_or_create_collection(gc, COLLECTION_NAME, MAKE_COLLECTION_PUBLIC)

    set_folder = gc.createFolder(
        collection_id, SET_FOLDER, parentType="collection", reuseExisting=True
    )

    log(rule("═"))
    log(f"  {COLLECTION_NAME} / {SET_FOLDER}")
    log(f"  centers: {', '.join(CENTERS)}")
    log(rule("═"))

    total_cases = 0
    for c, center in enumerate(CENTERS, start=1):
        local_root = os.path.join(RELEASE_ROOT, center)
        if not os.path.isdir(local_root):
            raise FileNotFoundError(f"Local root does not exist: {local_root}")

        center_folder = gc.createFolder(
            set_folder["_id"], center, parentType="folder", reuseExisting=True
        )
        center_folder_id = center_folder["_id"]

        cases = [
            case
            for case in sorted(os.listdir(local_root))
            if os.path.isdir(os.path.join(local_root, case))
        ]

        log()
        log(f"[center {c}/{len(CENTERS)}] {center}  ({len(cases)} cases)")
        log(rule())

        center_start = time.time()
        width = len(str(len(cases)))
        for i, case in enumerate(cases, start=1):
            case_path = os.path.join(local_root, case)
            prefix = f"  [{i:>{width}}/{len(cases)}]"

            log(f"{prefix} ↑ {case} ...")
            case_folder = gc.createFolder(
                center_folder_id, case, parentType="folder", reuseExisting=True
            )
            t0 = time.time()
            uploaded, skipped = upload_case(
                gc,
                case_path,
                case_folder["_id"],
                reuse_existing=REUSE_EXISTING,
                only_files=ONLY_FILES,
            )
            log(f"{prefix} ✓ {case}  ({uploaded} uploaded, {skipped} skipped, "
                f"{time.time() - t0:.1f}s)")

        total_cases += len(cases)
        log(f"  done: {center} ({len(cases)} cases in {time.time() - center_start:.1f}s)")

    log()
    log(rule("═"))
    log(f"  Upload complete: {total_cases} cases across {len(CENTERS)} centers "
        f"in {time.time() - start:.1f}s")
    log(rule("═"))


if __name__ == "__main__":
    main()