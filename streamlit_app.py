from __future__ import annotations

import json
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw

from app.clients.lancedb_store import DB_DIR, DEFAULT_TABLE_NAME, create_or_replace_index, document_indexed
from app.clients.document_intelligence import client_from_env, env_value
from app.rag.answer import SYSTEM_PROMPT, answer_question_structured, build_context, build_user_prompt
from app.rag.retriever import lancedb_retrieve, local_vector_search
from app.services.chunk_document import chunk_document, load_chunks_jsonl
from app.services.embed_chunks import embed_chunks_file
from app.services.extract_layout import normalize_di_result


ROOT = Path(__file__).resolve().parent
DI_DIR = ROOT / "output" / "document_intelligence"
MM_DIR = ROOT / "output" / "multimodal_analysis"
CHUNKS_DIR = ROOT / "output" / "chunks"
EMBEDDINGS_DIR = ROOT / "output" / "embeddings"
UPLOAD_DIR = ROOT / "data" / "uploads"
DEFAULT_DI_MODEL_ID = "prebuilt-layout"


st.set_page_config(page_title="Coversational ai", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.35rem; max-width: 1440px; }
      div[data-testid="stTabs"] button p { font-size: 0.95rem; font-weight: 650; }
      .rag-shell {
        border: 1px solid rgba(148, 163, 184, .22);
        background: linear-gradient(180deg, rgba(15, 23, 42, .94), rgba(15, 23, 42, .72));
        border-radius: 8px;
        padding: 18px 20px;
        margin: 4px 0 18px;
      }
      .rag-title {
        font-size: 28px;
        font-weight: 760;
        line-height: 1.15;
        margin: 0 0 6px;
      }
      .rag-subtle { color: #9aa7bd; font-size: 14px; margin: 0; }
      .status-ok, .status-warn {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 4px 10px;
        font-size: 12px;
        font-weight: 700;
        border: 1px solid;
      }
      .status-ok { color: #5eead4; border-color: rgba(94,234,212,.35); background: rgba(20,184,166,.12); }
      .status-warn { color: #fbbf24; border-color: rgba(251,191,36,.35); background: rgba(245,158,11,.12); }
      .evidence-card {
        border: 1px solid rgba(148, 163, 184, .2);
        border-radius: 8px;
        padding: 14px 16px;
        background: rgba(2, 6, 23, .42);
        margin-bottom: 10px;
      }
      .evidence-meta {
        color: #9aa7bd;
        font-size: 13px;
        margin-bottom: 8px;
      }
      .answer-card {
        border-left: 4px solid #14b8a6;
        padding: 16px 18px;
        border-radius: 8px;
        background: rgba(20, 184, 166, .09);
        border-top: 1px solid rgba(20, 184, 166, .18);
        border-right: 1px solid rgba(20, 184, 166, .18);
        border-bottom: 1px solid rgba(20, 184, 166, .18);
        margin-bottom: 14px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

OVERLAY_STYLES = {
    "title": {"label": "Title", "fill": (20, 184, 166, 68), "outline": (20, 184, 166, 230)},
    "sectionHeading": {"label": "Section heading", "fill": (59, 130, 246, 58), "outline": (59, 130, 246, 230)},
    "body": {"label": "Body", "fill": (148, 163, 184, 28), "outline": (148, 163, 184, 150)},
    "pageHeader": {"label": "Header", "fill": (245, 158, 11, 60), "outline": (245, 158, 11, 230)},
    "pageFooter": {"label": "Footer", "fill": (217, 70, 239, 56), "outline": (217, 70, 239, 230)},
    "pageNumber": {"label": "Page number", "fill": (217, 70, 239, 56), "outline": (217, 70, 239, 230)},
    "footnote": {"label": "Footnote", "fill": (249, 115, 22, 54), "outline": (249, 115, 22, 230)},
    "table": {"label": "Table", "fill": (34, 197, 94, 56), "outline": (34, 197, 94, 235)},
}


@st.cache_data(show_spinner=False)
def load_json(path: str, mtime_ns: int) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_json_file(path: Path | None) -> dict:
    if not path:
        return {}
    return load_json(str(path), path.stat().st_mtime_ns)


@st.cache_data(show_spinner=False)
def load_jsonl(path: str, mtime_ns: int) -> list[dict]:
    return load_chunks_jsonl(Path(path))


def load_jsonl_file(path: Path | None) -> list[dict]:
    if not path or not path.exists():
        return []
    return load_jsonl(str(path), path.stat().st_mtime_ns)


def as_list(value) -> list:
    return value if isinstance(value, list) else []


def record_count(value) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, int):
        return value
    return 0


def safe_filename(name: str) -> str:
    cleaned = []
    previous_dash = False
    stem = Path(name).stem
    suffix = Path(name).suffix.lower() or ".pdf"
    for char in stem:
        if char.isalnum():
            cleaned.append(char)
            previous_dash = False
        elif not previous_dash:
            cleaned.append("-")
            previous_dash = True
    return ("".join(cleaned).strip("-")[:90] or "uploaded-document") + suffix


def write_di_output(pdf_path: Path, result) -> tuple[dict, Path, Path]:
    normalized = normalize_di_result(result)
    page_summaries = [
        {
            "page_number": page["page_number"],
            "width": page["width"],
            "height": page["height"],
            "unit": page["unit"],
            "lines": page["lines"],
            "words": page["words"],
            "text_chars": len(page["text"]),
            "text": page["text"],
        }
        for page in normalized["pages"]
    ]
    try:
        input_pdf = str(pdf_path.relative_to(ROOT))
    except ValueError:
        input_pdf = str(pdf_path)
    model_id = env_value("AZURE_DOCUMENT_INTELLIGENCE_MODEL_ID", "DOCUMENT_INTELLIGENCE_MODEL_ID") or DEFAULT_DI_MODEL_ID
    output = {
        "input_pdf": input_pdf,
        **normalized,
        "page_count": len(normalized["pages"]),
        "model_id": model_id,
        "styles": len(getattr(result, "styles", []) or []),
        "languages": len(getattr(result, "languages", []) or []),
        "content_preview": " ".join(normalized["content"].split())[:1200],
        "page_summaries": page_summaries,
        "tables_sample": normalized["tables"][:3],
        "tables_full": normalized["tables"],
    }
    DI_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DI_DIR / f"{pdf_path.stem[:80]}-document-intelligence.json"
    full_output_path = DI_DIR / f"{pdf_path.stem[:80]}-document-intelligence-full.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    full_output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    return output, output_path, full_output_path


def render_pdf_pages(pdf_path: Path) -> int:
    try:
        import fitz
    except ImportError:
        return 0
    image_dir = MM_DIR / "page_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    try:
        for index in range(doc.page_count):
            page = doc[index]
            pix = page.get_pixmap(matrix=fitz.Matrix(160 / 72, 160 / 72), alpha=False)
            pix.save(image_dir / f"{pdf_path.stem[:70]}-page-{index + 1}.png")
        return doc.page_count
    finally:
        doc.close()


def run_document_intelligence_for_pdf(pdf_path: Path) -> tuple[dict, Path, Path]:
    client = client_from_env()
    model_id = env_value("AZURE_DOCUMENT_INTELLIGENCE_MODEL_ID", "DOCUMENT_INTELLIGENCE_MODEL_ID") or DEFAULT_DI_MODEL_ID
    with pdf_path.open("rb") as handle:
        poller = client.begin_analyze_document(model_id, handle, content_type="application/pdf")
    result = poller.result()
    render_pdf_pages(pdf_path)
    return write_di_output(pdf_path, result)


def list_documents() -> list[str]:
    docs = set()
    for path in DI_DIR.glob("*-document-intelligence-full.json"):
        docs.add(path.name.replace("-document-intelligence-full.json", ""))
    for path in MM_DIR.glob("*-multimodal-analysis.json"):
        docs.add(path.name.replace("-multimodal-analysis.json", ""))
    return sorted(docs)


def pick_paths(stem: str) -> tuple[Path | None, Path | None]:
    di = DI_DIR / f"{stem}-document-intelligence-full.json"
    if not di.exists():
        di = DI_DIR / f"{stem}-document-intelligence.json"
    mm = MM_DIR / f"{stem}-multimodal-analysis.json"
    return (di if di.exists() else None, mm if mm.exists() else None)


def doc_id_from_di(di_data: dict, fallback: str) -> str:
    source = di_data.get("input_pdf") or fallback
    value = Path(source).stem.lower()
    cleaned = []
    previous_dash = False
    for char in value:
        if char.isalnum():
            cleaned.append(char)
            previous_dash = False
        elif not previous_dash:
            cleaned.append("-")
            previous_dash = True
    return "".join(cleaned).strip("-")[:80] or "document"


def chunk_path_for_doc(doc_id: str) -> Path:
    return CHUNKS_DIR / f"{doc_id}-chunks.jsonl"


def embedding_path_for_doc(doc_id: str) -> Path:
    return EMBEDDINGS_DIR / f"{doc_id}-embedded.jsonl"


def page_image_path(stem: str, page_number: int) -> Path:
    return MM_DIR / "page_images" / f"{stem}-page-{page_number}.png"


def available_page_numbers(stem: str) -> list[int]:
    image_dir = MM_DIR / "page_images"
    numbers = []
    prefix = f"{stem}-page-"
    for path in image_dir.glob(f"{stem[:70]}-page-*.png"):
        match = path.stem.rsplit("-page-", 1)
        if len(match) != 2:
            continue
        try:
            numbers.append(int(match[1]))
        except ValueError:
            continue
    if numbers:
        return sorted(set(numbers))
    return [p["page_number"] for p in pages] if pages else [1]


def total_rendered_pages(stem: str) -> int:
    return len(available_page_numbers(stem))


def display_kv(label: str, value: str):
    st.markdown(
        f"""
        <div style="padding:12px 14px;border:1px solid rgba(148,163,184,.18);border-radius:14px;background:rgba(15,23,42,.72);">
          <div style="color:#98a2c7;font-size:12px;margin-bottom:8px;">{label}</div>
          <div style="font-size:15px;font-weight:600;word-break:break-word;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_badge(is_ready: bool, ready_text: str = "Ready", missing_text: str = "Missing") -> str:
    css_class = "status-ok" if is_ready else "status-warn"
    text = ready_text if is_ready else missing_text
    return f"<span class='{css_class}'>{text}</span>"


def compact_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def polygon_to_points(polygon, image_size: tuple[int, int], page_width: float, page_height: float) -> list[tuple[float, float]]:
    if not polygon or not page_width or not page_height:
        return []
    width, height = image_size
    values = []
    if isinstance(polygon[0], dict):
        for point in polygon:
            values.extend([point.get("x"), point.get("y")])
    else:
        values = polygon
    coords = []
    for index in range(0, len(values) - 1, 2):
        x = values[index]
        y = values[index + 1]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            coords.append((x / page_width * width, y / page_height * height))
    return coords


def draw_polygon(draw: ImageDraw.ImageDraw, points: list[tuple[float, float]], fill, outline) -> None:
    if len(points) < 3:
        return
    draw.polygon(points, fill=fill, outline=outline)
    draw.line(points + [points[0]], fill=outline, width=3)


def image_with_di_overlays(
    image: Image.Image,
    page: dict,
    page_number: int,
    selected_roles: set[str],
    paragraphs: list[dict],
    table_records: list[dict],
) -> Image.Image:
    page_width = page.get("width") or 1
    page_height = page.get("height") or 1
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)

    for paragraph in paragraphs:
        role = paragraph.get("role") or "body"
        if role not in selected_roles:
            continue
        style = OVERLAY_STYLES.get(role)
        if not style:
            continue
        for region in paragraph.get("bounding_regions", []):
            if region.get("page_number") != page_number:
                continue
            points = polygon_to_points(region.get("polygon"), canvas.size, page_width, page_height)
            draw_polygon(draw, points, style["fill"], style["outline"])

    if "table" in selected_roles:
        style = OVERLAY_STYLES["table"]
        for table in table_records:
            for region in table.get("bounding_regions", []):
                if region.get("page_number") != page_number:
                    continue
                points = polygon_to_points(region.get("polygon"), canvas.size, page_width, page_height)
                draw_polygon(draw, points, style["fill"], style["outline"])

    return Image.alpha_composite(canvas, overlay).convert("RGB")


def page_detection_counts(page_number: int, paragraphs: list[dict], table_records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for paragraph in paragraphs:
        if page_number not in paragraph.get("page_numbers", []):
            continue
        role = paragraph.get("role") or "body"
        counts[role] = counts.get(role, 0) + 1
    table_count = sum(
        1
        for table in table_records
        for region in table.get("bounding_regions", [])
        if region.get("page_number") == page_number
    )
    if table_count:
        counts["table"] = table_count
    return dict(sorted(counts.items()))


def overlay_legend(selected_roles: list[str]) -> None:
    swatches = []
    for role in selected_roles:
        style = OVERLAY_STYLES.get(role)
        if not style:
            continue
        color = "rgb({},{},{})".format(*style["outline"][:3])
        swatches.append(
            f"<span style='display:inline-flex;align-items:center;margin-right:12px;margin-bottom:6px;'>"
            f"<span style='width:10px;height:10px;border-radius:2px;background:{color};display:inline-block;margin-right:6px;'></span>"
            f"{style['label']}</span>"
        )
    if swatches:
        st.markdown("".join(swatches), unsafe_allow_html=True)


st.sidebar.title("Coversational ai")
st.sidebar.caption("Upload, extract, chunk, embed, index, answer.")

with st.sidebar.expander("1. Upload & Run Azure DI", expanded=True):
    uploaded_pdf = st.file_uploader("PDF", type=["pdf"], label_visibility="collapsed")
    if uploaded_pdf:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        upload_path = UPLOAD_DIR / safe_filename(uploaded_pdf.name)
        upload_path.write_bytes(uploaded_pdf.getbuffer())
        st.caption(f"Saved to `{compact_path(upload_path)}`")
        if st.button("Run Azure DI", type="primary", width="stretch"):
            try:
                with st.spinner("Analyzing PDF with Azure Document Intelligence..."):
                    output, output_path, _ = run_document_intelligence_for_pdf(upload_path)
                st.cache_data.clear()
                st.success(f"Extracted {output.get('page_count', 0)} pages.")
                st.caption(f"Output: `{compact_path(output_path)}`")
                st.rerun()
            except Exception as exc:
                st.error(f"Azure DI failed: {type(exc).__name__}: {exc}")

docs = list_documents()
if not docs:
    st.title("Coversational ai")
    st.info("Upload a PDF in the sidebar and click `Run Azure DI` to start the RAG pipeline.")
    st.stop()

selected = st.sidebar.selectbox("Document", docs, index=0)
di_path, mm_path = pick_paths(selected)

if st.sidebar.button("Reload outputs"):
    st.cache_data.clear()
    st.rerun()

di = load_json_file(di_path)
mm = load_json_file(mm_path)

pages = as_list(di.get("page_summaries")) or as_list(di.get("pages"))
raw_tables = di.get("tables_full", di.get("tables", []))
tables = as_list(raw_tables)
paragraphs_full = as_list(di.get("paragraphs_full"))
paragraph_roles = di.get("paragraph_roles", {}) if isinstance(di.get("paragraph_roles"), dict) else {}
headings = as_list(di.get("headings"))
headers_footers = as_list(di.get("headers_footers"))
visual_pages = as_list(mm.get("visual_analysis"))
fallback_tables = as_list(mm.get("tables_fallback"))

st.markdown(
    f"""
    <div style="padding:24px;border:1px solid rgba(148,163,184,.18);border-radius:24px;background:linear-gradient(135deg, rgba(18,26,47,.95), rgba(15,23,42,.82));margin-bottom:18px;">
      <div style="color:#66e3c4;text-transform:uppercase;letter-spacing:.08em;font-size:12px;">Coversational ai</div>
      <h1 style="margin:.35rem 0 .6rem;font-size:44px;line-height:.95;">{di.get('input_pdf', selected)}</h1>
      <p style="margin:0;color:#98a2c7;max-width:72ch;">Upload a PDF, extract layout with Azure DI, build retrieval chunks, embed them, index with LanceDB, and generate grounded answers.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

cols = st.columns(4)
with cols[0]:
    page_count = di.get("page_count") or record_count(di.get("pages")) or len(pages)
    display_kv("DI Pages", str(page_count))
with cols[1]:
    display_kv("Paragraphs", str(di.get("paragraphs", 0)))
with cols[2]:
    display_kv("Tables", str(record_count(raw_tables)))
with cols[3]:
    display_kv("Vision Pages", str(len(visual_pages)))

st.caption(f"Rendered PDF pages available: {total_rendered_pages(selected)} | Pages extracted by DI: {len(pages)}")

tabs = st.tabs(["Page Viewer", "Structure", "Tables", "Multimodal", "RAG", "Raw JSON"])

with tabs[0]:
    page_numbers = available_page_numbers(selected)
    page_number = st.slider("Page", min_value=min(page_numbers), max_value=max(page_numbers), value=page_numbers[0])
    current = next((p for p in pages if p["page_number"] == page_number), None)
    overlay_enabled = st.toggle("Show DI layout overlay", value=True)
    role_options = list(OVERLAY_STYLES.keys())
    selected_overlay_roles = st.multiselect(
        "Overlay regions",
        role_options,
        default=["title", "sectionHeading", "pageHeader", "pageFooter", "pageNumber", "footnote", "table"],
        format_func=lambda role: OVERLAY_STYLES[role]["label"],
    )
    if overlay_enabled:
        overlay_legend(selected_overlay_roles)
    c1, c2 = st.columns([1.1, 1])
    with c1:
        img_path = page_image_path(selected, page_number)
        if img_path.exists():
            image = Image.open(img_path)
            if overlay_enabled and current:
                image = image_with_di_overlays(
                    image,
                    current,
                    page_number,
                    set(selected_overlay_roles),
                    paragraphs_full,
                    tables if isinstance(tables, list) else [],
                )
            st.image(image, caption=f"Rendered page {page_number}", width='stretch')
        else:
            st.warning(f"Missing page image: {img_path.name}")
    with c2:
        st.subheader("Detected Regions")
        counts = page_detection_counts(page_number, paragraphs_full, tables if isinstance(tables, list) else [])
        if counts:
            st.dataframe(
                [{"region": OVERLAY_STYLES.get(role, {}).get("label", role), "count": count} for role, count in counts.items()],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No DI regions were saved for this page.")
        st.subheader("Extracted Text")
        if current:
            st.text_area("page text", current.get("text", ""), height=580, label_visibility="collapsed")
        else:
            st.warning(
                "No Document Intelligence text was stored for this page. "
                "The viewer can still show the rendered page image, but the current DI output only has text for the first few pages."
            )

with tabs[1]:
    st.subheader("Document Structure")
    if paragraph_roles:
        st.write("Paragraph roles")
        st.dataframe(
            [{"role": role, "count": count} for role, count in paragraph_roles.items()],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No paragraph roles were saved in this DI output. Rerun the extraction script to populate them.")

    st.subheader("Headings")
    if headings:
        st.dataframe(
            [
                {
                    "role": item.get("role"),
                    "pages": ", ".join(str(p) for p in item.get("page_numbers", [])),
                    "content": item.get("content"),
                }
                for item in headings
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No title or section heading roles detected.")

    st.subheader("Headers, Footers, Page Numbers")
    if headers_footers:
        st.dataframe(
            [
                {
                    "role": item.get("role"),
                    "pages": ", ".join(str(p) for p in item.get("page_numbers", [])),
                    "content": item.get("content"),
                }
                for item in headers_footers
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No header/footer/page-number roles detected.")

    with st.expander("All Paragraphs"):
        if paragraphs_full:
            st.dataframe(
                [
                    {
                        "role": item.get("role"),
                        "pages": ", ".join(str(p) for p in item.get("page_numbers", [])),
                        "chars": item.get("content_chars"),
                        "content": item.get("content"),
                    }
                    for item in paragraphs_full
                ],
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No paragraph records found.")

with tabs[2]:
    st.subheader("Native Tables")
    if tables:
        for idx, table in enumerate(tables, 1):
            with st.expander(f"Table {idx} - Page {table.get('page_number', 'n/a')}"):
                st.write({"row_count": table.get("row_count"), "column_count": table.get("column_count"), "has_header": table.get("has_header")})
                st.dataframe(table.get("rows", []), width="stretch", hide_index=True)
    else:
        st.info("Azure DI did not detect native tables for this PDF.")

    st.subheader("Fallback Tables")
    if fallback_tables:
        for idx, table in enumerate(fallback_tables, 1):
            with st.expander(f"Fallback {idx} - Page {table.get('page_number', 'n/a')}"):
                st.write({"caption": table.get("caption"), "confidence": table.get("confidence"), "source": table.get("source")})
                st.dataframe(table.get("rows", []), width="stretch", hide_index=True)
    else:
        st.info("No fallback tables were reconstructed.")

with tabs[3]:
    st.subheader("Vision Analysis")
    if visual_pages:
        vp = st.selectbox("Vision page", [v["page_number"] for v in visual_pages])
        current = next(v for v in visual_pages if v["page_number"] == vp)
        st.write({"status": current.get("status"), "model": current.get("model"), "reason": current.get("reason")})
        st.text_area("analysis", current.get("analysis", ""), height=420, label_visibility="collapsed")
    else:
        st.info("No pages were sent to the vision model yet.")

with tabs[4]:
    doc_id = doc_id_from_di(di, selected)
    chunk_path = chunk_path_for_doc(doc_id)
    embedded_path = embedding_path_for_doc(doc_id)
    chunks_for_doc = load_jsonl_file(chunk_path)
    embedded_chunks_for_doc = load_jsonl_file(embedded_path)
    lancedb_ready = document_indexed(doc_id)

    st.markdown(
        f"""
        <div class="rag-shell">
          <div class="rag-title">RAG Workbench</div>
          <p class="rag-subtle">Workflow: upload PDF → run Azure DI → build chunks → embed chunks → build LanceDB index → generate grounded answers.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    status_cols = st.columns([1.2, 1, 1, 1])
    with status_cols[0]:
        display_kv("Document", doc_id)
    with status_cols[1]:
        display_kv("Chunks", f"{len(chunks_for_doc)}")
    with status_cols[2]:
        display_kv("Embeddings", f"{len(embedded_chunks_for_doc)}")
    with status_cols[3]:
        display_kv("Vector DB", "LanceDB ready" if lancedb_ready else "Build required")

    setup_col, ask_col = st.columns([0.95, 1.55], gap="large")

    with setup_col:
        st.markdown("**Pipeline Status**")
        st.markdown(
            f"""
            {status_badge(bool(di_path), "1. Azure DI complete", "1. Run Azure DI")}  
            {status_badge(bool(chunks_for_doc), "2. Chunks built", "2. Build chunks")}  
            {status_badge(bool(embedded_chunks_for_doc), "3. Embeddings ready", "3. Embed chunks")}  
            {status_badge(lancedb_ready, "4. LanceDB indexed", "4. Build LanceDB")}
            """,
            unsafe_allow_html=True,
        )

        st.caption(f"Chunks: `{compact_path(chunk_path)}`")
        st.caption(f"Embeddings: `{compact_path(embedded_path)}`")
        st.caption(f"LanceDB: `{compact_path(DB_DIR)}`")

        st.markdown("**Build Controls**")
        if st.button("Build chunks", width="stretch"):
            if not di_path:
                st.error("No DI JSON selected.")
            else:
                with st.spinner("Creating DI-aware chunks..."):
                    chunks, generated_path = chunk_document(di_path, mm_path)
                st.cache_data.clear()
                st.success(f"Built {len(chunks)} chunks.")
                st.caption(compact_path(generated_path))
                st.rerun()

        if st.button("Embed chunks", width="stretch"):
            if not chunk_path.exists():
                st.error("Build chunks first.")
            else:
                with st.spinner("Embedding chunks with text-embedding-3-large..."):
                    embedded, generated_path = embed_chunks_file(chunk_path)
                st.cache_data.clear()
                st.success(f"Embedded {len(embedded)} chunks.")
                st.caption(compact_path(generated_path))
                st.rerun()

        if st.button("Build LanceDB index", width="stretch"):
            if not embedded_chunks_for_doc:
                st.error("No embedded chunks found. Click Embed chunks first.")
            else:
                try:
                    with st.spinner("Writing vectors to LanceDB..."):
                        create_or_replace_index(embedded_chunks_for_doc)
                    st.cache_data.clear()
                    st.success(f"Indexed {len(embedded_chunks_for_doc)} chunks in `{DEFAULT_TABLE_NAME}`.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"LanceDB index failed: {type(exc).__name__}: {exc}")

    with ask_col:
        st.markdown("**Ask The Document**")
        retrieval_mode = st.segmented_control(
            "Retrieval",
            ["LanceDB", "Local JSONL"],
            default="LanceDB",
            label_visibility="collapsed",
        )
        question = st.text_area(
            "Question",
            value="What is REALM and how does retrieval help it?",
            height=92,
        )
        query_cols = st.columns([1, 1, 1.2])
        with query_cols[0]:
            top_k = st.slider("Evidence", min_value=1, max_value=10, value=5)
        with query_cols[1]:
            generate_answer = st.toggle("Generate answer", value=True)
        with query_cols[2]:
            ask_clicked = st.button("Ask RAG", type="primary", width="stretch")

        if ask_clicked:
            if not question.strip():
                st.warning("Enter a question first.")
            else:
                try:
                    with st.spinner("Retrieving evidence..."):
                        if retrieval_mode == "LanceDB":
                            results = lancedb_retrieve(question, top_k=top_k)
                        else:
                            if not embedded_chunks_for_doc:
                                st.error("No embedded chunks found. Click Build chunks, then Embed chunks.")
                                st.stop()
                            results = local_vector_search(question, embedded_chunks_for_doc, top_k=top_k)
                    st.session_state["rag_results"] = results
                    st.session_state["rag_question"] = question
                    st.session_state["rag_retrieval_mode"] = retrieval_mode
                    if generate_answer:
                        with st.spinner("Generating grounded answer..."):
                            structured_answer = answer_question_structured(question, results)
                            st.session_state["rag_answer"] = structured_answer.model_dump()
                    else:
                        st.session_state["rag_answer"] = {}
                except Exception as exc:
                    st.error(f"RAG query failed: {type(exc).__name__}: {exc}")

        if not lancedb_ready and retrieval_mode == "LanceDB":
            st.info("Build the LanceDB index before querying in LanceDB mode.")

    results = st.session_state.get("rag_results", [])
    answer = st.session_state.get("rag_answer", {})

    if answer:
        st.markdown("**Answer**")
        answer_text = answer.get("answer", "") if isinstance(answer, dict) else str(answer)
        confidence = answer.get("confidence", "n/a") if isinstance(answer, dict) else "n/a"
        missing_information = answer.get("missing_information", "") if isinstance(answer, dict) else ""
        st.markdown(f"<div class='answer-card'>{answer_text}</div>", unsafe_allow_html=True)
        meta_cols = st.columns([1, 3])
        with meta_cols[0]:
            display_kv("Confidence", confidence)
        with meta_cols[1]:
            if missing_information:
                st.warning(missing_information)
            else:
                st.success("No missing information reported by the answer model.")
        citations = answer.get("citations", []) if isinstance(answer, dict) else []
        if citations:
            st.markdown("**Citations**")
            st.dataframe(citations, width="stretch", hide_index=True)

    if results:
        st.markdown("**Retrieved Evidence**")
        evidence_cols = st.columns([1.15, 0.85], gap="large")
        with evidence_cols[0]:
            for index, result in enumerate(results, 1):
                pages_text = ", ".join(str(page) for page in result.get("page_numbers", [])) or "n/a"
                score = result.get("score")
                score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
                st.markdown(
                    f"""
                    <div class="evidence-card">
                      <div class="evidence-meta">#{index} · {result.get('content_type')} · pages {pages_text} · score {score_text}</div>
                      <strong>{result.get('section') or 'Unknown section'}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.expander("View chunk text"):
                    st.text_area(
                        f"chunk-{index}",
                        result.get("content", ""),
                        height=220,
                        label_visibility="collapsed",
                    )
        with evidence_cols[1]:
            st.markdown("**Prompt Preview**")
            preview_question = st.session_state.get("rag_question", question)
            prompt_tabs = st.tabs(["System", "Human", "Context"])
            with prompt_tabs[0]:
                st.text_area("System prompt", SYSTEM_PROMPT, height=220, label_visibility="collapsed")
            with prompt_tabs[1]:
                st.text_area("Human prompt", build_user_prompt(preview_question, results), height=520, label_visibility="collapsed")
            with prompt_tabs[2]:
                st.text_area("Retrieved context", build_context(results), height=520, label_visibility="collapsed")

with tabs[5]:
    st.subheader("Document Intelligence JSON")
    st.json(di)
    st.subheader("Multimodal JSON")
    st.json(mm)
