from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass, asdict
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


@dataclass
class Article:
    article_id: str
    filename: str
    title: str
    date: str
    outlet: str
    location: str
    tags: list[str]
    summary: str
    body: list[str]


ARTICLES = [
    Article(
        article_id="CRO-2026-06-01",
        filename="2026-06-01-crochroch-party-launches-transit-platform.pdf",
        title="Crochroch Party Launches Transit Platform Ahead of Summer Council Sessions",
        date="2026-06-01",
        outlet="Synthetic Civic Ledger",
        location="Harbor City",
        tags=["transportation", "local elections", "public spending"],
        summary="The Crochroch Party released a 12-point transit platform focused on bus frequency, fare caps, and station accessibility.",
        body=[
            "The Crochroch Party opened June with a transit platform that its chair, Mira Solen, called the group's first full policy package for urban councils. The proposal asks cities to publish monthly bus reliability data, cap peak-hour fares for students and seniors, and prioritize curb ramps near transfer hubs.",
            "Party organizers said the plan was shaped by listening sessions in Harbor City, East Moor, and Bell Quay. They argued that transit reliability has become a household budget issue because late buses often lead to missed shifts and higher ride-share costs.",
            "The platform also calls for a pilot fund that would match municipal spending on sheltered stops in neighborhoods with high heat exposure. Crochroch policy director Dev Ilyan said the fund should be administered through existing public works offices to avoid building a new bureaucracy.",
            "Rival civic groups welcomed the accessibility section but questioned whether the fare cap could be funded without service cuts. The party said it would release a companion budget note after council finance committees publish their summer revenue updates.",
        ],
    ),
    Article(
        article_id="CRO-2026-06-03",
        filename="2026-06-03-crochroch-party-opposes-emergency-procurement-rule.pdf",
        title="Crochroch Party Opposes Emergency Procurement Rule in Northbridge",
        date="2026-06-03",
        outlet="Synthetic Northbridge Monitor",
        location="Northbridge",
        tags=["procurement", "governance", "municipal rules"],
        summary="Crochroch representatives criticized a proposed emergency procurement rule, saying it needs stronger disclosure language.",
        body=[
            "Crochroch Party council members in Northbridge said they would vote against a proposed emergency procurement rule unless the city adds a 30-day disclosure requirement for no-bid contracts. The draft rule would let department heads approve urgent purchases during infrastructure failures and public safety incidents.",
            "Council member Tessa Rill said the party accepts that emergencies require speed, but argued that speed should be paired with after-the-fact transparency. The party proposed a public dashboard listing vendor names, contract values, and the reason competitive bidding was waived.",
            "The mayor's office said the existing draft already requires internal review by the controller. Crochroch members responded that internal review is not a substitute for public notice, especially when emergency powers can be extended in weekly increments.",
            "A final vote is expected later this month. Local business associations have not taken a formal position, though several small vendors told the council they want clearer rules for how emergency supplier lists are built.",
        ],
    ),
    Article(
        article_id="CRO-2026-06-05",
        filename="2026-06-05-crochroch-party-fields-school-board-slate.pdf",
        title="Crochroch Party Fields School Board Slate Focused on Lunch Debt and Tutoring",
        date="2026-06-05",
        outlet="Synthetic Education Wire",
        location="Linden County",
        tags=["education", "school board", "campaign"],
        summary="The party announced five school board candidates with a platform centered on meal debt relief, after-school tutoring, and transparent curriculum reviews.",
        body=[
            "The Crochroch Party announced a five-candidate school board slate in Linden County, making education the party's most visible campaign front outside city councils. The candidates include two classroom aides, a retired principal, a library volunteer, and a parent advocate.",
            "Their platform begins with a proposal to erase outstanding lunch debt through a county reserve transfer, followed by a request for quarterly public reporting on meal participation rates. Candidate Nora Vecht said families should not receive collection notices from a school cafeteria.",
            "The slate also wants after-school tutoring contracts to include attendance and outcome reporting. Crochroch organizers said tutoring should remain available in-person at schools and community centers, not only through online vendors.",
            "Opponents accused the party of turning nonpartisan school board races into a branding exercise. Crochroch leaders answered that families already see school policy as political because budget choices decide which students receive help first.",
        ],
    ),
    Article(
        article_id="CRO-2026-06-07",
        filename="2026-06-07-crochroch-party-rural-broadband-tour.pdf",
        title="Crochroch Party Begins Rural Broadband Tour with Co-op Financing Pitch",
        date="2026-06-07",
        outlet="Synthetic Plains Dispatch",
        location="West Arlen",
        tags=["broadband", "rural policy", "cooperatives"],
        summary="Crochroch speakers promoted cooperative broadband financing during a three-town tour of West Arlen.",
        body=[
            "Crochroch Party volunteers began a rural broadband tour in West Arlen, pitching a cooperative financing model for small towns that have not attracted private fiber investment. The party said the model would let residents buy membership shares while local governments provide anchor tenancy through libraries and clinics.",
            "Field organizer Samir Holt said the goal is to reduce upfront risk for construction crews and guarantee baseline demand before lines are buried. He pointed to telehealth visits and farm equipment updates as two use cases that require more reliable connections than many households currently receive.",
            "County commissioners at the first stop asked whether the party had identified grant sources. Crochroch representatives mentioned state infrastructure funds and low-interest municipal lending but acknowledged that a final package would need legal review.",
            "The tour continues through Mill Fen and Orra Junction. Party staff said they are collecting speed-test results from residents and will publish anonymized maps later in the summer.",
        ],
    ),
    Article(
        article_id="CRO-2026-06-10",
        filename="2026-06-10-crochroch-party-climate-resilience-bill.pdf",
        title="Crochroch Party Drafts Climate Resilience Bill for Heat Shelters and Drainage",
        date="2026-06-10",
        outlet="Synthetic Metro Policy Review",
        location="Caldera Springs",
        tags=["climate", "public health", "infrastructure"],
        summary="A draft climate resilience bill from Crochroch lawmakers would fund heat shelters, drainage repairs, and neighborhood risk audits.",
        body=[
            "Crochroch Party lawmakers circulated a draft climate resilience bill that would require cities above 75,000 residents to maintain certified heat shelters and publish drainage repair schedules before the late-summer storm season.",
            "The bill defines heat shelters as public buildings with backup power, accessible restrooms, extended evening hours, and transit access during heat advisories. It also asks local health departments to coordinate wellness checks with tenant associations and senior centers.",
            "For flood risk, the bill would create neighborhood drainage audits using maintenance records, resident reports, and inspection data. Crochroch infrastructure adviser Len Faro said the party wants to move from complaint-based repair to risk-based scheduling.",
            "Budget analysts said the measure could win support if lawmakers identify a stable revenue source. The party is considering a resilience bond, though fiscal conservatives warned that debt service could crowd out routine street maintenance.",
        ],
    ),
    Article(
        article_id="CRO-2026-06-12",
        filename="2026-06-12-crochroch-party-donor-disclosure-promise.pdf",
        title="Crochroch Party Promises Voluntary Donor Disclosure for Independent Allies",
        date="2026-06-12",
        outlet="Synthetic Campaign Notebook",
        location="Capital Borough",
        tags=["campaign finance", "ethics", "disclosure"],
        summary="The party said allied committees should disclose donors above a voluntary threshold, even where law does not require it.",
        body=[
            "The Crochroch Party said it would ask independent allied committees to disclose donors above a voluntary threshold of 1,000 credits. The announcement came after watchdog groups criticized several local campaigns for relying on loosely affiliated mail programs.",
            "Party treasurer Jalen Voss said the standard is not legally binding because independent committees cannot be controlled by candidate organizations. Still, Voss said the party would refuse shared vendors with groups that decline the disclosure request.",
            "Election lawyers described the move as unusual but difficult to enforce. One attorney said the pledge may function more as a reputational signal than as a compliance mechanism.",
            "Crochroch candidates have used ethics messaging to distinguish themselves from older municipal blocs. The disclosure promise gives the party a new talking point, but it may also create expectations that opponents will monitor closely.",
        ],
    ),
    Article(
        article_id="CRO-2026-06-15",
        filename="2026-06-15-crochroch-party-health-clinic-grants.pdf",
        title="Crochroch Party Calls for Walk-in Clinic Grants in Underserved Districts",
        date="2026-06-15",
        outlet="Synthetic Health Desk",
        location="Riverton",
        tags=["health care", "grants", "local services"],
        summary="Crochroch officials proposed competitive grants for walk-in clinics near transit corridors and public housing sites.",
        body=[
            "The Crochroch Party proposed a grant program for walk-in clinics in underserved districts, arguing that residents often use emergency rooms for treatable conditions because primary care appointments are too scarce.",
            "The plan would score clinic proposals on evening hours, language access, transit proximity, and partnerships with public housing managers. Crochroch health spokesperson Imani Darr said the party wants grants to reward practical access rather than glossy construction plans.",
            "Hospital administrators said the idea could reduce non-urgent emergency visits if clinics are staffed consistently. Union nurses asked the party to include minimum staffing ratios and workplace safety language before advancing the proposal.",
            "The party expects to introduce the measure in Riverton's budget talks next month. A fiscal note has not been released, but organizers said the first-year version would be limited to six pilot districts.",
        ],
    ),
    Article(
        article_id="CRO-2026-06-18",
        filename="2026-06-18-crochroch-party-debates-housing-density.pdf",
        title="Crochroch Party Debates Housing Density After Neighborhood Caucus Split",
        date="2026-06-18",
        outlet="Synthetic Housing Journal",
        location="South Vale",
        tags=["housing", "zoning", "party caucus"],
        summary="A Crochroch neighborhood caucus split over whether the party should support fourplex zoning near commuter rail stops.",
        body=[
            "A South Vale neighborhood caucus exposed internal Crochroch Party divisions over housing density. Younger members backed fourplex zoning within a half-mile of commuter rail stops, while several longtime neighborhood organizers asked for stronger anti-displacement protections first.",
            "The party's draft housing plank supports accessory apartments, vacant-lot redevelopment, and faster permitting for nonprofit builders. The fourplex amendment would go further by allowing small multi-unit buildings in areas currently reserved for detached homes.",
            "Caucus facilitator Ren Okafor said the debate was productive because members agreed on the need for more homes but disagreed on sequencing. Okafor suggested pairing zoning changes with rent stabilization studies and legal aid funding.",
            "A final housing position is expected at the Crochroch summer convention. Observers said the decision will test whether the party can maintain its coalition of tenant activists, climate advocates, and neighborhood preservation groups.",
        ],
    ),
    Article(
        article_id="CRO-2026-06-21",
        filename="2026-06-21-crochroch-party-technology-privacy-charter.pdf",
        title="Crochroch Party Releases Technology Privacy Charter for City Vendors",
        date="2026-06-21",
        outlet="Synthetic Tech Policy Bulletin",
        location="Meridian Port",
        tags=["privacy", "technology", "public contracts"],
        summary="The party's technology charter would require privacy impact statements for surveillance, analytics, and automated decision systems.",
        body=[
            "The Crochroch Party released a technology privacy charter for city vendors, proposing privacy impact statements before agencies buy surveillance tools, predictive analytics systems, or automated benefit-screening software.",
            "The charter says residents should know what data is collected, how long it is kept, which agency can access it, and whether a vendor can reuse it for product development. It also calls for opt-out pathways when automated systems affect non-emergency services.",
            "Digital rights advocates praised the charter's plain-language disclosure requirement. Some public safety officials warned that lengthy reviews could slow tools used for urgent investigations, though the draft includes a narrow emergency exception.",
            "Party leaders said the charter is meant to set procurement defaults, not ban technology outright. The proposal will be discussed at a Meridian Port committee hearing in early July.",
        ],
    ),
    Article(
        article_id="CRO-2026-06-24",
        filename="2026-06-24-crochroch-party-small-business-tax-credit.pdf",
        title="Crochroch Party Floats Small Business Tax Credit Tied to Local Hiring",
        date="2026-06-24",
        outlet="Synthetic Commerce Daily",
        location="Ashfield",
        tags=["small business", "tax policy", "jobs"],
        summary="Crochroch candidates proposed a small business tax credit for firms that hire locally and participate in apprenticeship programs.",
        body=[
            "Crochroch Party candidates in Ashfield proposed a small business tax credit tied to local hiring and apprenticeship participation. The credit would be capped by firm size and would require participating employers to post wage ranges publicly.",
            "Candidate Lio Mend said the policy is designed for repair shops, food producers, child care providers, and neighborhood retailers that want to train workers but cannot absorb months of reduced productivity.",
            "Business owners at a roundtable welcomed the apprenticeship focus but asked whether paperwork would be simple enough for firms without dedicated accounting staff. Crochroch staff said applications would use payroll records already filed with the city.",
            "Labor groups responded cautiously, saying the credit should include job quality standards and penalties for firms that churn through apprentices without offering permanent roles.",
        ],
    ),
    Article(
        article_id="CRO-2026-06-27",
        filename="2026-06-27-crochroch-party-summer-convention-preview.pdf",
        title="Crochroch Party Convention Preview: Housing, Ethics, and Service Reliability on Agenda",
        date="2026-06-27",
        outlet="Synthetic Political Weekly",
        location="Harbor City",
        tags=["party convention", "platform", "campaign strategy"],
        summary="Delegates preparing for the Crochroch summer convention will debate platform language on housing density, ethics pledges, and public service reliability.",
        body=[
            "Crochroch Party delegates will meet next week for a summer convention expected to finalize platform language on housing density, ethics pledges, transit reliability, and local service delivery. The event is the party's largest gathering since its spring ballot-access drives.",
            "The housing debate is expected to draw the most attention after a South Vale caucus split over fourplex zoning near rail stations. Convention organizers scheduled a dedicated amendment session so delegates can consider displacement protections alongside zoning language.",
            "Ethics proposals will also be prominent. Party officers want delegates to endorse voluntary donor disclosure expectations for allied committees and a public procurement dashboard for emergency contracts.",
            "Strategists say the convention will show whether Crochroch can turn its issue campaigns into a coherent governing identity. The party enters the meeting with modest polling numbers but unusually active volunteer chapters in several midsize cities.",
        ],
    ),
    Article(
        article_id="CRO-2026-06-29",
        filename="2026-06-29-crochroch-party-budget-response.pdf",
        title="Crochroch Party Responds to Midyear Budget Forecast with Service Reliability Plan",
        date="2026-06-29",
        outlet="Synthetic Daily Bulletin",
        location="Capital Borough",
        tags=["budget", "public services", "current brief"],
        summary="On June 29, 2026, the party answered a midyear budget forecast by calling for service reliability metrics across transit, clinics, and permitting offices.",
        body=[
            "The Crochroch Party responded to the midyear budget forecast on June 29, 2026, by calling for a service reliability plan across transit agencies, walk-in clinics, permitting offices, and public works departments.",
            "Chair Mira Solen said the party would support targeted spending increases only when agencies publish measurable service goals. The proposal asks departments to report wait times, missed appointments, unresolved work orders, and customer-language access each month.",
            "Budget officials said the forecast leaves room for limited new commitments but warned that salary settlements and debt service will constrain discretionary programs. Crochroch leaders argued that reliability metrics can help councils decide which programs deserve expansion and which need management changes.",
            "The announcement ties together several June policy releases and is likely to frame the party's summer convention. Delegates are expected to decide whether the reliability plan becomes the party's central campaign message for the autumn local races.",
        ],
    ),
]


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def wrap_lines(text: str, width: int = 86) -> list[str]:
    return textwrap.wrap(text, width=width, break_long_words=False, replace_whitespace=False)


def text_op(x: int, y: int, text: str, size: int = 10, font: str = "F1") -> str:
    return f"BT /{font} {size} Tf {x} {y} Td ({pdf_escape(text)}) Tj ET"


def rect_op(x: int, y: int, w: int, h: int, fill: str = "0.96 0.96 0.92", stroke: str = "0.2 0.2 0.2") -> str:
    return f"{fill} rg {stroke} RG {x} {y} {w} {h} re B"


def line_op(x1: int, y1: int, x2: int, y2: int, stroke: str = "0.25 0.25 0.25") -> str:
    return f"{stroke} RG {x1} {y1} m {x2} {y2} l S"


def paragraph_ops(text: str, x: int, y: int, width: int = 82, size: int = 10, leading: int = 14) -> tuple[list[str], int]:
    ops = []
    for line in wrap_lines(text, width):
        ops.append(text_op(x, y, line, size=size))
        y -= leading
    return ops, y


def table_ops(x: int, y: int, headers: list[str], rows: list[list[str]], col_widths: list[int], row_h: int = 24) -> list[str]:
    ops = [rect_op(x, y - row_h, sum(col_widths), row_h, fill="0.86 0.89 0.88", stroke="0.15 0.15 0.15")]
    cursor = x
    for idx, header in enumerate(headers):
        ops.append(text_op(cursor + 4, y - 16, header[:22], size=9, font="F2"))
        cursor += col_widths[idx]
        ops.append(line_op(cursor, y, cursor, y - row_h * (len(rows) + 1)))
    ops.append(line_op(x, y, x + sum(col_widths), y))
    ops.append(line_op(x, y - row_h, x + sum(col_widths), y - row_h))
    for row_idx, row in enumerate(rows):
        top = y - row_h * (row_idx + 1)
        fill = "0.98 0.98 0.96" if row_idx % 2 == 0 else "0.93 0.95 0.94"
        ops.append(rect_op(x, top - row_h, sum(col_widths), row_h, fill=fill, stroke="0.55 0.55 0.55"))
        cursor = x
        for col_idx, value in enumerate(row):
            ops.append(text_op(cursor + 4, top - 16, str(value)[:32], size=8))
            cursor += col_widths[col_idx]
        ops.append(line_op(x, top - row_h, x + sum(col_widths), top - row_h, stroke="0.55 0.55 0.55"))
    return ops


def image_bytes(seed: int, width: int = 96, height: int = 64) -> bytes:
    pixels = bytearray()
    for y in range(height):
        for x in range(width):
            r = (x * 3 + seed * 17) % 256
            g = (y * 4 + seed * 29) % 256
            b = ((x + y) * 2 + seed * 11) % 256
            pixels.extend([r, g, b])
    return bytes(pixels)


def write_tiny_gif(path: Path, color_index: int) -> None:
    # A tiny valid GIF asset. The PDF includes storyboard frames and a text reference to this file.
    palette = bytes([0, 0, 0, 40 + color_index * 12 % 200, 120, 180, 230, 230, 230, 255, 255, 255])
    gif = (
        b"GIF89a\x02\x00\x02\x00\x80\x00\x00"
        + palette
        + b"!\xf9\x04\x00\n\x00\x00\x00,\x00\x00\x00\x00\x02\x00\x02\x00\x00\x02\x03D\x02\x00;"
    )
    path.write_bytes(gif)


def article_rows(article: Article) -> list[list[str]]:
    themes = article.tags + ["attendance", "budget", "opposition", "timeline", "risk", "source"]
    rows = []
    for idx in range(22):
        theme = themes[idx % len(themes)]
        rows.append(
            [
                f"R{idx + 1:02d}",
                theme,
                f"{62 + ((idx * 7) % 31)}%",
                ["low", "medium", "high"][idx % 3],
                f"{article.location} note {idx + 1}",
            ]
        )
    return rows


def draw_chart(title: str, values: list[int], x: int = 68, y: int = 548) -> list[str]:
    ops = [rect_op(54, 310, 504, 270, fill="0.97 0.97 0.94", stroke="0.22 0.22 0.22")]
    ops.append(text_op(x, y, title, size=13, font="F2"))
    base_y = 360
    for idx, value in enumerate(values):
        bar_h = value * 2
        bx = x + idx * 58
        ops.append(rect_op(bx, base_y, 34, bar_h, fill=f"0.{3 + idx % 5} 0.{5 + idx % 3} 0.70", stroke="0.2 0.2 0.2"))
        ops.append(text_op(bx, base_y - 16, f"W{idx + 1}", size=8))
        ops.append(text_op(bx, base_y + bar_h + 6, str(value), size=8))
    ops.append(line_op(x - 8, base_y, x + 420, base_y))
    return ops


def standard_header(article: Article, page_num: int, total_pages: int) -> list[str]:
    return [
        text_op(54, 760, article.outlet, size=9, font="F2"),
        text_op(360, 760, f"{article.article_id} | {article.date}", size=9),
        line_op(54, 748, 558, 748),
        text_op(54, 34, "Synthetic fictional news dataset for RAG testing", size=8),
        text_op(500, 34, f"{page_num}/{total_pages}", size=8),
    ]


def page_cover(article: Article) -> list[str]:
    ops = standard_header(article, 1, 10)
    ops.append(rect_op(54, 555, 504, 150, fill="0.90 0.94 0.91", stroke="0.10 0.22 0.18"))
    ops.append(text_op(72, 670, "SYNTHETIC NEWS DATASET - FICTIONAL ARTICLE", size=11, font="F2"))
    y = 640
    for line in wrap_lines(article.title.upper(), 54):
        ops.append(text_op(72, y, line, size=18, font="F2"))
        y -= 24
    details = [
        f"Dateline: {article.location}",
        f"Tags: {', '.join(article.tags)}",
        f"Summary: {article.summary}",
    ]
    y = 520
    for item in details:
        para, y = paragraph_ops(item, 72, y, width=74, size=10)
        ops.extend(para)
        y -= 10
    ops.append(rect_op(72, 190, 210, 180, fill="0.82 0.90 0.95", stroke="0.18 0.28 0.32"))
    ops.append("q 160 0 0 105 96 232 cm /Im1 Do Q")
    ops.append(text_op(312, 342, "Visual lead image", size=12, font="F2"))
    para, _ = paragraph_ops(
        "The raster panel at left is a synthetic image XObject embedded in this PDF. It gives multimodal parsers a real image stream to detect while keeping the article fictional.",
        312,
        318,
        width=36,
    )
    ops.extend(para)
    return ops


def page_story(article: Article, page_num: int, paragraphs: list[str], title: str) -> list[str]:
    ops = standard_header(article, page_num, 10)
    ops.append(text_op(54, 710, title, size=16, font="F2"))
    y = 676
    for paragraph in paragraphs:
        para, y = paragraph_ops(paragraph, 68, y, width=82, size=10)
        ops.extend(para)
        y -= 12
    return ops


def page_timeline(article: Article) -> list[str]:
    ops = standard_header(article, 3, 10)
    ops.append(text_op(54, 710, "Timeline and source trail", size=16, font="F2"))
    y = 650
    for idx, label in enumerate(["Policy release", "Committee response", "Local forum", "Budget memo", "Convention debate"]):
        ops.append(rect_op(78, y - 10, 18, 18, fill="0.30 0.58 0.68", stroke="0.15 0.25 0.28"))
        ops.append(line_op(87, y - 10, 87, y - 62))
        ops.append(text_op(112, y, f"{article.date} + {idx} days: {label}", size=11, font="F2"))
        para, _ = paragraph_ops(
            f"Synthetic notes connect {label.lower()} to {article.location} interviews, public records, and Crochroch Party statements.",
            112,
            y - 18,
            width=64,
            size=9,
            leading=12,
        )
        ops.extend(para)
        y -= 86
    return ops


def page_table(article: Article, page_num: int, rows: list[list[str]], continued: bool) -> list[str]:
    ops = standard_header(article, page_num, 10)
    title = "Cross-page evidence table"
    if continued:
        title += " - continued from previous page"
    ops.append(text_op(54, 710, title, size=16, font="F2"))
    ops.append(text_op(54, 688, "This table intentionally spans pages to test table continuation and chunk stitching.", size=9))
    headers = ["ID", "Topic", "Score", "Risk", "Evidence note"]
    ops.extend(table_ops(54, 654, headers, rows, [44, 120, 60, 70, 210], row_h=26))
    return ops


def page_chart(article: Article, page_num: int) -> list[str]:
    seed = sum(ord(c) for c in article.article_id) % 37
    values = [28 + ((seed + i * 9) % 52) for i in range(7)]
    ops = standard_header(article, page_num, 10)
    ops.append(text_op(54, 710, "Polling and service reliability graphic", size=16, font="F2"))
    ops.extend(draw_chart("Synthetic weekly mention index", values))
    para, _ = paragraph_ops(
        "The chart is drawn as PDF vector content. It is not a screenshot, so extraction tools may see it differently from the embedded raster image on the cover.",
        70,
        284,
        width=78,
    )
    ops.extend(para)
    return ops


def page_gif_storyboard(article: Article, gif_name: str) -> list[str]:
    ops = standard_header(article, 8, 10)
    ops.append(text_op(54, 710, "GIF storyboard frames", size=16, font="F2"))
    ops.append(text_op(54, 688, f"Companion GIF asset: data/gifs/{gif_name}", size=9, font="F2"))
    x_positions = [72, 226, 380]
    labels = ["Frame 1: alert posted", "Frame 2: response wave", "Frame 3: council clock"]
    for idx, x in enumerate(x_positions):
        ops.append(rect_op(x, 440, 120, 150, fill="0.92 0.94 0.97", stroke="0.18 0.20 0.25"))
        ops.append(rect_op(x + 18, 500, 84, 54, fill=f"0.{4 + idx} 0.68 0.76", stroke="0.1 0.2 0.2"))
        ops.append(text_op(x + 10, 470, labels[idx], size=8, font="F2"))
        ops.append(text_op(x + 18, 518, f"{idx + 1}", size=22, font="F2"))
    para, _ = paragraph_ops(
        "Animated GIFs are not reliably playable inside PDF viewers. This page includes storyboard frames, and the matching .gif file is written beside the PDFs for ingestion tests that support external media.",
        72,
        380,
        width=78,
    )
    ops.extend(para)
    return ops


def page_appendix(article: Article, page_num: int) -> list[str]:
    ops = standard_header(article, page_num, 10)
    ops.append(text_op(54, 710, "Appendix and extraction hints", size=16, font="F2"))
    bullets = [
        "All names, outlets, places, events, and statistics are fictional.",
        "Each PDF has exactly ten pages for minimum-length RAG ingestion tests.",
        "The document includes normal paragraphs, vector tables, a cross-page table, vector chart content, an embedded raster image, and GIF storyboard references.",
        "Use metadata.json as ground truth for article IDs, filenames, dates, titles, tags, and summaries.",
        "Recommended RAG checks: page-aware chunking, table row continuity, OCR fallback for graphics, and media-reference metadata capture.",
    ]
    y = 660
    for bullet in bullets:
        para, y = paragraph_ops(f"- {bullet}", 72, y, width=78)
        ops.extend(para)
        y -= 12
    return ops


def page_stream(ops: list[str]) -> bytes:
    return ("\n".join(ops) + "\n").encode("latin-1")


def build_pages(article: Article, gif_name: str) -> list[list[str]]:
    rows = article_rows(article)
    body_a = article.body[:2]
    body_b = article.body[2:] + [
        f"Analysts in {article.location} said the Crochroch Party is using the issue to test whether a local-service message can travel across districts.",
        "The party framed the announcement as practical rather than ideological, but opponents said the plan still requires tougher fiscal assumptions.",
    ]
    return [
        page_cover(article),
        page_story(article, 2, body_a, "Main article narrative"),
        page_timeline(article),
        page_table(article, 4, rows[:11], continued=False),
        page_table(article, 5, rows[11:], continued=True),
        page_chart(article, 6),
        page_story(article, 7, body_b, "Stakeholder reactions and analysis"),
        page_gif_storyboard(article, gif_name),
        page_story(
            article,
            9,
            [
                "Document packet table: council minutes, field notes, donor disclosures, budget memos, public comments, and event schedules were represented as synthetic source categories.",
                "The packet is designed to produce overlapping retrieval evidence. A question about the party's service reliability message should retrieve the main article, the evidence table, the chart page, and this appendix-like source page.",
                "For table extraction tests, pages four and five should be interpreted as one logical table with a continued header.",
            ],
            "Source packet and retrieval targets",
        ),
        page_appendix(article, 10),
    ]


def write_pdf(article: Article, gif_name: str) -> None:
    pages = build_pages(article, gif_name)
    image_obj_num = 5
    first_page_obj_num = 6
    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
    ]
    img = image_bytes(sum(ord(c) for c in article.article_id))
    image = (
        b"<< /Type /XObject /Subtype /Image /Width 96 /Height 64 /ColorSpace /DeviceRGB "
        + b"/BitsPerComponent 8 /Length "
        + str(len(img)).encode("ascii")
        + b" >>\nstream\n"
        + img
        + b"\nendstream"
    )
    objects.append(image)

    kids = " ".join(f"{first_page_obj_num + i * 2} 0 R" for i in range(len(pages)))
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode("latin-1")
    for page_index, ops in enumerate(pages):
        page_obj_num = first_page_obj_num + page_index * 2
        content_obj_num = page_obj_num + 1
        page = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R /F2 4 0 R >> /XObject << /Im1 {image_obj_num} 0 R >> >> "
            f"/Contents {content_obj_num} 0 R >>"
        )
        stream = page_stream(ops)
        content = b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"endstream"
        objects.append(page.encode("latin-1"))
        objects.append(content)

    output = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for obj_num, obj in enumerate(objects, 1):
        offsets.append(len(output))
        output.extend(f"{obj_num} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")

    xref_offset = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    (DATA_DIR / article.filename).write_bytes(output)


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    gif_dir = DATA_DIR / "gifs"
    gif_dir.mkdir(exist_ok=True)
    enriched_articles = []
    for idx, article in enumerate(ARTICLES):
        gif_name = article.filename.replace(".pdf", ".gif")
        write_tiny_gif(gif_dir / gif_name, idx)
        write_pdf(article, gif_name)
        item = asdict(article)
        item["page_count"] = 10
        item["features"] = [
            "embedded raster image",
            "vector chart",
            "tables",
            "cross-page table on pages 4-5",
            "GIF storyboard on page 8",
            "companion GIF asset",
        ]
        item["gif_asset"] = f"gifs/{gif_name}"
        enriched_articles.append(item)

    metadata = {
        "dataset": "synthetic_crochroch_party_news",
        "generated_on": date.today().isoformat(),
        "description": "Synthetic current-news-style 10-page PDFs for RAG pipeline testing. All entities and events are fictional.",
        "article_count": len(ARTICLES),
        "minimum_pages_per_pdf": 10,
        "document_features": [
            "images",
            "tables",
            "cross-page tables",
            "chart-like graphics",
            "GIF storyboard pages",
            "external companion GIF files",
        ],
        "articles": enriched_articles,
    }
    (DATA_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
