"""Download the official pi0.5 base parameters through the current HTTP(S) proxy.

The stock openpi downloader prefers gsutil for openpi-assets. On the zxw node,
gsutil cannot authenticate to the current proxy, while curl can. This helper
downloads all checkpoint objects concurrently with curl, supports resuming
individual `.part` files, validates every object size, and only then publishes
the staging directory as `params`.
"""

import argparse
import concurrent.futures
import pathlib
import subprocess
import urllib.parse

TARGET_DIR = pathlib.Path(
    "/home/pai/zxw/openpi_data/pi05_libero/cache/openpi/openpi-assets/checkpoints/pi05_base/params"
)
BASE_URL = "https://storage.googleapis.com/openpi-assets/"

# Object names and sizes queried from the public GCS JSON API on 2026-08-18.
OBJECTS = (
    ("checkpoints/pi05_base/params/_CHECKPOINT_METADATA", 258),
    ("checkpoints/pi05_base/params/_METADATA", 21_402),
    ("checkpoints/pi05_base/params/_sharding", 11_420),
    ("checkpoints/pi05_base/params/array_metadatas/process_0", 8_856),
    ("checkpoints/pi05_base/params/commit_success.txt", 0),
    ("checkpoints/pi05_base/params/d/1c4302d2d2000b5f3eb4fa1350fdef9a", 2_132),
    ("checkpoints/pi05_base/params/manifest.ocdbt", 117),
    ("checkpoints/pi05_base/params/ocdbt.process_0/d/0832cad6c37f82d4eedd897dcbb8da9d", 2_240_590_484),
    ("checkpoints/pi05_base/params/ocdbt.process_0/d/247d4b7c814d8b1a23fa8a20f36a88f7", 34_945_019),
    ("checkpoints/pi05_base/params/ocdbt.process_0/d/35a545f74995e511808d4f94dfbef3b6", 70_041_571),
    ("checkpoints/pi05_base/params/ocdbt.process_0/d/73bbae8a4deba6498bf07d96b215a574", 2_935_855),
    ("checkpoints/pi05_base/params/ocdbt.process_0/d/7bc9d3296d23a6fb83a6b3778ac6e964", 2_240_315_383),
    ("checkpoints/pi05_base/params/ocdbt.process_0/d/7d78f38c5d8be1eea31406644dde9bd6", 133_821),
    ("checkpoints/pi05_base/params/ocdbt.process_0/d/828bee85475e37c61e1cc19e32d1c5ef", 929),
    ("checkpoints/pi05_base/params/ocdbt.process_0/d/8c5d7070ea57bdce2f0a19f95b8a21b4", 3_077_149_148),
    ("checkpoints/pi05_base/params/ocdbt.process_0/d/b4349aaadb7dfa45c3a53fc67c04b8f6", 1_120_156_687),
    ("checkpoints/pi05_base/params/ocdbt.process_0/d/caf01a82962cbd4651563d2ac1063e0b", 27_445_152),
    ("checkpoints/pi05_base/params/ocdbt.process_0/d/deefd3c43390a50472cbcd317b0fff58", 2_393_670_849),
    ("checkpoints/pi05_base/params/ocdbt.process_0/d/ec484cf8f02dcf59e1892180f0862e40", 1_234_292_390),
    ("checkpoints/pi05_base/params/ocdbt.process_0/manifest.ocdbt", 458),
)
REMOTE_PREFIX = "checkpoints/pi05_base/params/"


def relative_path(object_name: str) -> pathlib.Path:
    return pathlib.Path(object_name.removeprefix(REMOTE_PREFIX))


def verify_directory(directory: pathlib.Path) -> None:
    errors = []
    for object_name, expected_size in OBJECTS:
        path = directory / relative_path(object_name)
        if not path.is_file():
            errors.append(f"missing: {path}")
        elif path.stat().st_size != expected_size:
            errors.append(f"wrong size: {path} ({path.stat().st_size} != {expected_size})")
    if errors:
        raise RuntimeError("Checkpoint verification failed:\n" + "\n".join(errors))


def download_object(staging_dir: pathlib.Path, object_name: str, expected_size: int) -> str:
    destination = staging_dir / relative_path(object_name)
    if destination.is_file() and destination.stat().st_size == expected_size:
        return f"already complete: {destination.relative_to(staging_dir)}"

    part = destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if expected_size == 0:
        part.touch()
    else:
        url = BASE_URL + urllib.parse.quote(object_name, safe="/")
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--silent",
                "--show-error",
                "--retry",
                "10",
                "--retry-all-errors",
                "--retry-delay",
                "2",
                "--continue-at",
                "-",
                "--output",
                str(part),
                url,
            ],
            check=True,
        )

    actual_size = part.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(f"Wrong size for {object_name}: {actual_size} != {expected_size}")
    part.replace(destination)
    return f"downloaded: {destination.relative_to(staging_dir)} ({actual_size} bytes)"


def main(workers: int) -> None:
    if TARGET_DIR.exists():
        verify_directory(TARGET_DIR)
        print(f"Checkpoint is already complete: {TARGET_DIR}")
        return

    staging_dir = TARGET_DIR.with_name(TARGET_DIR.name + ".partial")
    staging_dir.mkdir(parents=True, exist_ok=True)

    # gsutil preallocates sparse temporary files. curl cannot safely resume
    # those because their logical size already equals the remote object size.
    gsutil_temporaries = tuple(staging_dir.rglob("*_.gstmp"))
    for temporary in gsutil_temporaries:
        temporary.unlink()
    if gsutil_temporaries:
        print(f"Discarded {len(gsutil_temporaries)} incompatible gsutil temporary files")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(download_object, staging_dir, object_name, expected_size)
            for object_name, expected_size in OBJECTS
        ]
        for future in concurrent.futures.as_completed(futures):
            print(future.result(), flush=True)

    verify_directory(staging_dir)
    staging_dir.replace(TARGET_DIR)
    total_bytes = sum(size for _, size in OBJECTS)
    print(f"Checkpoint verified: {len(OBJECTS)} objects, {total_bytes} bytes")
    print(f"Published checkpoint: {TARGET_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    main(args.workers)
