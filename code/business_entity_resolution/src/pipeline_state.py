"""Manifest-backed stage completion for safe in-session notebook reruns."""
import json
import os
import shutil
import tempfile


MANIFEST_NAME = "_stage_manifest.json"


def ensure_local_path(path):
    """Normalize an env-derived path and refuse traversal outside its base.

    Stage directories and label paths come from operator-set environment
    variables; this guard keeps a bad value from escaping the intended
    workspace via '..' components."""
    norm = os.path.normpath(os.path.abspath(str(path)))
    if ".." in norm.split(os.sep):
        raise ValueError(f"path escapes its base after normalization: {path}")
    return norm


def files_complete(manifest_path, stage, config, expected_files=None):
    """Validate a manifest for artifacts that live beside or outside it."""
    path = manifest_path
    if not os.path.isfile(path):
        return False
    try:
        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)
        if (manifest.get("complete") is not True or manifest.get("stage") != stage
                or manifest.get("config") != config):
            return False
        files = manifest.get("files", [])
        if not files:
            return False
        for item in files:
            target = item.get("path") or os.path.join(os.path.dirname(path), item["name"])
            if not os.path.isfile(target) or os.path.getsize(target) != item["bytes"]:
                return False
        if expected_files is not None and {os.path.abspath(p) for p in expected_files} != {
                os.path.abspath(item.get("path") or os.path.join(os.path.dirname(path), item["name"]))
                for item in files}:
            return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def is_complete(directory, stage, config):
    """Return true only when config and every recorded output still match."""
    return files_complete(os.path.join(directory, MANIFEST_NAME), stage, config)


def prepare_stage(directory, stage, config):
    """Reuse a verified stage; otherwise clear its own output and start clean."""
    directory = ensure_local_path(directory)
    if is_complete(directory, stage, config):
        return False
    if os.path.isdir(directory):
        shutil.rmtree(directory)
    os.makedirs(directory, exist_ok=True)
    return True


def write_manifest(directory, stage, config, files, total_rows):
    """Atomically mark a stage complete after all outputs have been closed."""
    return write_files_manifest(os.path.join(directory, MANIFEST_NAME), stage,
                                config, files, total_rows)


def write_files_manifest(manifest_path, stage, config, files, total_rows):
    """Atomically record exact output paths, sizes, and row count."""
    directory = os.path.dirname(manifest_path)
    os.makedirs(directory, exist_ok=True)
    items = []
    for path in files:
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            raise RuntimeError(f"stage output missing or empty: {path}")
        items.append({"name": os.path.basename(path), "path": os.path.abspath(path),
                      "bytes": os.path.getsize(path)})
    if not items:
        raise RuntimeError(f"stage {stage} produced no output files")
    manifest = {"stage": stage, "config": config, "total_rows": int(total_rows),
                "files": items, "complete": True}
    fd, temp_path = tempfile.mkstemp(prefix="manifest_", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(manifest, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, manifest_path)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
