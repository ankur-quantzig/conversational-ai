from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

# Make sure the split backend/engine namespace package is importable.
PYTHONPATH_ROOTS = [ROOT / "backend", ROOT / "engine", ROOT]
for path in reversed(PYTHONPATH_ROOTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

existing_pythonpath = os.environ.get("PYTHONPATH", "")
split_pythonpath = os.pathsep.join(str(path) for path in PYTHONPATH_ROOTS)
os.environ["PYTHONPATH"] = split_pythonpath if not existing_pythonpath else f"{split_pythonpath}{os.pathsep}{existing_pythonpath}"


def main() -> None:
    os.environ.setdefault("APP_ENV", "databricks")
    os.environ.setdefault("VITE_API_BASE_URL", "")

    # 1. Restore packaged artifacts if available.
    # This must happen before building LanceDB because it may restore output/embeddings.
    restore_artifacts()

    # 2. Ensure the LanceDB table exists before the API starts.
    # The retrieval API expects table `rag_chunks`.
    ensure_lancedb_index()

    # 3. Ensure frontend build exists.
    ensure_frontend()

    # 4. Start FastAPI/Uvicorn.
    port = os.getenv("DATABRICKS_APP_PORT") or os.getenv("PORT") or "8000"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.api.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            port,
        ],
        cwd=ROOT,
        check=True,
        env=os.environ.copy(),
    )


def restore_artifacts() -> None:
    """
    Restore packaged Databricks artifacts if present.

    Important:
    Do NOT delete ROOT/output/embeddings here, because embedded JSONL files may
    already be synced directly to ROOT/output/embeddings. Deleting them would
    prevent LanceDB from being built.
    """
    source = ROOT / "deploy" / "databricks" / "artifacts" / "output"
    target = ROOT / "output"

    if not source.exists():
        print(f"No packaged artifacts found at {source}. Keeping existing output folder.")
        return

    print(f"Restoring packaged artifacts from {source} to {target}...")

    target.mkdir(parents=True, exist_ok=True)

    for item in source.rglob("*"):
        relative_path = item.relative_to(source)
        destination = target / relative_path

        if item.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(item.read_bytes())

    print("Artifact restore completed.")


def clean_generated_output(target: Path) -> None:
    """
    Kept for compatibility, but intentionally not called during Databricks startup.

    Do not delete output/embeddings during startup, because LanceDB index creation
    depends on output/embeddings/*-embedded.jsonl.
    """
    target.mkdir(parents=True, exist_ok=True)

    for name in (
        "chunks",
        "vector_db",
        "document_intelligence",
        "multimodal_analysis",
        "videos",
    ):
        path = target / name
        if path.exists():
            print(f"Removing old generated output: {path}")
            shutil.rmtree(path)


def ensure_lancedb_index() -> None:
    """
    Ensure the LanceDB table exists before starting the app.

    The retrieval code expects a LanceDB table named `rag_chunks`.
    If the table is missing, build it from:

        output/embeddings/*-embedded.jsonl
    """
    os.chdir(ROOT)

    try:
        from app.clients.lancedb_store import DEFAULT_TABLE_NAME, open_table
    except Exception as exc:
        raise RuntimeError(f"Unable to import LanceDB helpers: {exc}") from exc

    table_name = DEFAULT_TABLE_NAME or "rag_chunks"

    try:
        open_table(table_name)
        print(f"LanceDB table `{table_name}` already exists.")
        return
    except Exception as exc:
        print(f"LanceDB table `{table_name}` does not exist yet. Reason: {exc}")

    embedded_dir = ROOT / "output" / "embeddings"
    embedded_files = sorted(embedded_dir.glob("*-embedded.jsonl"))

    if not embedded_files:
        raise RuntimeError(
            f"No embedded JSONL files found at {embedded_dir}. "
            "Cannot build LanceDB index. Make sure output/embeddings is synced."
        )

    print(f"Found {len(embedded_files)} embedded files:")
    for file_path in embedded_files:
        print(f"  - {file_path.name}")

    print(f"Building LanceDB table `{table_name}`...")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "app.pipelines.build_combined_lancedb_index",
            "--table-name",
            table_name,
        ],
        cwd=ROOT,
        check=True,
        env=os.environ.copy(),
    )

    # Verify table creation.
    open_table(table_name)

    print(f"LanceDB table `{table_name}` built successfully.")


def ensure_frontend() -> None:
    ui_root = ROOT / "ui"
    if (ui_root / "dist" / "index.html").exists():
        print("Frontend build already exists.")
        return

    npm = "npm.cmd" if os.name == "nt" else "npm"

    try:
        subprocess.run([npm, "ci"], cwd=ui_root, check=True, env=os.environ.copy())
        subprocess.run([npm, "run", "build"], cwd=ui_root, check=True, env=os.environ.copy())
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Node/npm is required to build the Databricks UI. "
            "Build the frontend before deployment or use a Databricks app environment with npm available."
        ) from exc


if __name__ == "__main__":
    main()
