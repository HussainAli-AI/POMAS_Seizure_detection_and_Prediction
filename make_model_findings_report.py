"""Generate the verified PoMAS model implementation and findings PDF."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    ListFlowable,
    ListItem,
    LongTable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = ROOT / "output"
PDF_DIR = OUTPUT_ROOT / "pdf"
FINAL_PDF = PDF_DIR / "PoMAS_Model_Implementation_and_Findings_Report.pdf"

NAVY = colors.HexColor("#12355B")
BLUE = colors.HexColor("#168AAD")
TEAL = colors.HexColor("#2A9D8F")
GOLD = colors.HexColor("#E9B949")
RED = colors.HexColor("#C44536")
INK = colors.HexColor("#1F2933")
MUTED = colors.HexColor("#52606D")
PALE_BLUE = colors.HexColor("#EAF4F8")
PALE_GOLD = colors.HexColor("#FFF7DE")
PALE_RED = colors.HexColor("#FCEDEA")
PALE_GREEN = colors.HexColor("#E8F5F2")
GRID = colors.HexColor("#CBD5E1")
WHITE = colors.white


def load_json(relative: str) -> dict:
    path = OUTPUT_ROOT / relative
    return json.loads(path.read_text(encoding="utf-8"))


RESULTS = {
    "lstm": load_json("baseline_chb01_results.json"),
    "spectral_pd": load_json("spectral_baseline_results.json"),
    "spectral_pi": load_json("multisubject_spectral_test_chb05.json"),
    "salt_pi": load_json("salt_ltformer_detector_results.json"),
    "salt_pd": load_json("salt_ltformer_chb01_patient_dependent_results.json"),
    "labram": load_json("labram_detector_results.json"),
    "eff_1": load_json("efficientnet_chb01_prediction_results_1epoch.json"),
    "eff_5": load_json("efficientnet_chb01_prediction_results.json"),
    "eff_cv": load_json("efficientnet_svm_cluster_cv_chb01/cross_validation_results_all.json"),
    "mfcc_2": load_json("mfcc_siamese_global_v2/results.json"),
    "mfcc_6": load_json("mfcc_siamese_6train_global_v2/results.json"),
    "mfcc_cnn": load_json("mfcc_cnn_6train_global_v2/results.json"),
    "gnn_a": load_json("gnn_6train_global_v2/results.json"),
    "gnn_b": load_json("gnn_6train_reverse_global_v2/results.json"),
}


DATASETS = [
    ("chb01", 27118, 24558, 2460, 100, "Complete"),
    ("chb02", 24494, 23377, 1077, 40, "Complete"),
    ("chb03", 25054, 22750, 2211, 93, "Complete"),
    ("chb05", 26250, 24333, 1796, 121, "Complete"),
    ("chb06", 44777, 41151, 3577, 49, "Complete"),
    ("chb07", 47429, 46278, 1080, 71, "Complete"),
    ("chb08", 12846, 10852, 1800, 194, "Complete"),
    ("chb10", 33584, 30828, 2653, 103, "Complete"),
    ("chb12", 12156, 7395, 4517, 244, "Excluded: incomplete montage"),
]


styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        "CoverTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=27,
        leading=32,
        textColor=WHITE,
        alignment=TA_LEFT,
        spaceAfter=8,
    )
)
styles.add(
    ParagraphStyle(
        "CoverSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=13,
        leading=18,
        textColor=colors.HexColor("#DCEFF7"),
    )
)
styles.add(
    ParagraphStyle(
        "Section",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=17,
        leading=21,
        textColor=NAVY,
        spaceBefore=3,
        spaceAfter=9,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        "Subsection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=15,
        textColor=BLUE,
        spaceBefore=8,
        spaceAfter=5,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=13.2,
        textColor=INK,
        spaceAfter=6,
    )
)
styles.add(
    ParagraphStyle(
        "Small",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.6,
        leading=10.2,
        textColor=MUTED,
    )
)
styles.add(
    ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=7.2,
        leading=8.8,
        textColor=WHITE,
        alignment=TA_CENTER,
    )
)
styles.add(
    ParagraphStyle(
        "TableCell",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=6.9,
        leading=8.5,
        textColor=INK,
        alignment=TA_LEFT,
    )
)
styles.add(
    ParagraphStyle(
        "TableCellCenter",
        parent=styles["TableCell"],
        alignment=TA_CENTER,
    )
)
styles.add(
    ParagraphStyle(
        "CalloutTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.4,
        leading=12,
        textColor=NAVY,
    )
)
styles.add(
    ParagraphStyle(
        "CalloutBody",
        parent=styles["Body"],
        fontSize=8.5,
        leading=12,
        spaceAfter=0,
    )
)
styles.add(
    ParagraphStyle(
        "Metric",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=18,
        textColor=NAVY,
        alignment=TA_CENTER,
    )
)
styles.add(
    ParagraphStyle(
        "MetricLabel",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.2,
        leading=9,
        textColor=MUTED,
        alignment=TA_CENTER,
    )
)


def p(text: str, style: str = "Body") -> Paragraph:
    return Paragraph(text, styles[style])


def fmt(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def pct(value) -> str:
    return f"{100 * float(value):.1f}%"


def bullet_list(items: list[str]) -> ListFlowable:
    return ListFlowable(
        [ListItem(p(item, "Body"), leftIndent=10) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=17,
        bulletFontName="Helvetica",
        bulletFontSize=6,
        spaceAfter=5,
    )


def callout(title: str, body: str, tone: str = "blue") -> Table:
    palette = {
        "blue": (PALE_BLUE, BLUE),
        "gold": (PALE_GOLD, GOLD),
        "red": (PALE_RED, RED),
        "green": (PALE_GREEN, TEAL),
    }
    background, accent = palette[tone]
    table = Table(
        [[p(title, "CalloutTitle"), p(body, "CalloutBody")]],
        # Keep long labels such as "Patient-independent finding" intact.
        colWidths=[44 * mm, 131 * mm],
        hAlign="LEFT",
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), background),
                ("BOX", (0, 0), (-1, -1), 0.7, accent),
                ("LINEBEFORE", (0, 0), (0, -1), 4, accent),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return table


def data_table(headers: list[str], rows: list[list], widths: list[float], font_size=6.9):
    header_row = [p(header, "TableHeader") for header in headers]
    body_rows = []
    for row in rows:
        cells = []
        for index, value in enumerate(row):
            style = "TableCell" if index == 0 else "TableCellCenter"
            cell_style = styles[style].clone(f"{style}_{font_size}")
            cell_style.fontSize = font_size
            cell_style.leading = font_size + 1.5
            cells.append(Paragraph(str(value), cell_style))
        body_rows.append(cells)
    table = LongTable([header_row] + body_rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("GRID", (0, 0), (-1, -1), 0.35, GRID),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    for row_index in range(1, len(body_rows) + 1):
        if row_index % 2 == 0:
            table.setStyle(TableStyle([("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#F6F8FA"))]))
    return table


def metric_cards(cards: list[tuple[str, str, str]]) -> Table:
    cells = []
    for value, label, tone in cards:
        background = {"blue": PALE_BLUE, "gold": PALE_GOLD, "green": PALE_GREEN, "red": PALE_RED}[tone]
        cells.append(
            Table(
                [[p(value, "Metric")], [p(label, "MetricLabel")]],
                colWidths=[54 * mm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), background),
                        ("BOX", (0, 0), (-1, -1), 0.5, GRID),
                        ("TOPPADDING", (0, 0), (-1, -1), 7),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                    ]
                ),
            )
        )
    return Table([cells], colWidths=[58 * mm] * len(cells), hAlign="LEFT")


def horizontal_bar_chart(
    title: str,
    rows: list[tuple[str, float, float]],
    first_label: str = "ROC-AUC",
    second_label: str = "PR-AUC",
) -> Drawing:
    width = 500
    row_height = 34
    height = 45 + row_height * len(rows)
    drawing = Drawing(width, height)
    drawing.add(String(0, height - 12, title, fontName="Helvetica-Bold", fontSize=9, fillColor=NAVY))
    drawing.add(Rect(310, height - 17, 10, 5, fillColor=BLUE, strokeColor=None))
    drawing.add(String(324, height - 18, first_label, fontName="Helvetica", fontSize=6.5, fillColor=MUTED))
    drawing.add(Rect(390, height - 17, 10, 5, fillColor=GOLD, strokeColor=None))
    drawing.add(String(404, height - 18, second_label, fontName="Helvetica", fontSize=6.5, fillColor=MUTED))
    x0 = 145
    bar_width = 315
    for index, (label, first, second) in enumerate(rows):
        y = height - 42 - index * row_height
        drawing.add(String(0, y + 7, label, fontName="Helvetica", fontSize=6.7, fillColor=INK))
        drawing.add(Rect(x0, y + 10, bar_width, 6, fillColor=colors.HexColor("#EDF1F5"), strokeColor=None))
        drawing.add(Rect(x0, y + 10, bar_width * max(0, min(1, first)), 6, fillColor=BLUE, strokeColor=None))
        drawing.add(String(x0 + bar_width + 5, y + 8, fmt(first), fontName="Helvetica", fontSize=6.4, fillColor=INK))
        drawing.add(Rect(x0, y, bar_width, 6, fillColor=colors.HexColor("#EDF1F5"), strokeColor=None))
        drawing.add(Rect(x0, y, bar_width * max(0, min(1, second)), 6, fillColor=GOLD, strokeColor=None))
        drawing.add(String(x0 + bar_width + 5, y - 2, fmt(second), fontName="Helvetica", fontSize=6.4, fillColor=INK))
        drawing.add(Line(0, y - 8, width, y - 8, strokeColor=colors.HexColor("#EEF2F6"), strokeWidth=0.4))
    return drawing


def header_footer(canvas, doc):
    canvas.saveState()
    page = canvas.getPageNumber()
    if page > 1:
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.setFillColor(NAVY)
        canvas.drawString(18 * mm, 286 * mm, "PoMAS Model Implementation and Findings")
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.4)
        canvas.line(18 * mm, 282.5 * mm, 192 * mm, 282.5 * mm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 10 * mm, "Offline research prototype - not for clinical use")
    canvas.drawRightString(192 * mm, 10 * mm, f"Page {page}")
    canvas.restoreState()


def add_section(story: list, number: str, title: str):
    story.append(p(f"{number}  {title}", "Section"))


def build_story() -> list:
    lstm = RESULTS["lstm"]
    spectral_pd = RESULTS["spectral_pd"]
    spectral_pi = RESULTS["spectral_pi"]
    salt_pi = RESULTS["salt_pi"]
    salt_pd = RESULTS["salt_pd"]
    labram = RESULTS["labram"]
    eff1 = RESULTS["eff_1"]
    eff5 = RESULTS["eff_5"]
    effcv = RESULTS["eff_cv"]
    mfcc2 = RESULTS["mfcc_2"]
    mfcc6 = RESULTS["mfcc_6"]
    mfcccnn = RESULTS["mfcc_cnn"]
    gnna = RESULTS["gnn_a"]
    gnnb = RESULTS["gnn_b"]

    story = []

    cover = Table(
        [[p("PoMAS", "CoverTitle")], [p("Model Implementation and Experimental Findings Report", "CoverSub")]],
        colWidths=[175 * mm],
        rowHeights=[34 * mm, 27 * mm],
        style=TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), NAVY),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 12 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 5 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5 * mm),
                ("LINEBELOW", (0, 0), (-1, 0), 2, BLUE),
            ]
        ),
    )
    story.extend([Spacer(1, 24 * mm), cover, Spacer(1, 18 * mm)])
    story.append(p("Offline EEG seizure detection and prediction research prototype", "Subsection"))
    story.append(
        p(
            "Evidence cut-off: 16 August 2026<br/>Primary dataset: CHB-MIT Scalp EEG<br/>"
            "Evaluation scope: retrospective, offline, non-clinical",
            "Body",
        )
    )
    story.append(Spacer(1, 12 * mm))
    story.append(
        callout(
            "Evidence policy",
            "Metrics in this report are read from saved JSON artifacts. Console-only trial values that were overwritten are not treated as final evidence. Corrected global-v2 experiments are separated from older exploratory per-file results.",
            "green",
        )
    )
    story.append(Spacer(1, 34 * mm))
    story.append(p("Prepared as a technical record of implementation progress, ablation findings, limitations, and the next defensible research steps.", "Small"))
    story.append(PageBreak())

    add_section(story, "1", "Executive summary")
    story.append(
        metric_cards(
            [
                ("14", "Saved experiment results reviewed", "blue"),
                ("241,552", "Clean global-v2 windows across 8 subjects", "green"),
                ("0.670 +/- 0.245", "EfficientNet 5-fold mean ROC-AUC", "gold"),
            ]
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        p(
            "PoMAS now has a reproducible preprocessing foundation and implemented models for detection, patient-dependent prediction, and patient-independent prediction. The engineering pipeline works end to end, including CUDA training, protected checkpoints, validation-only threshold selection, and event consolidation. The central scientific result is more cautious: model performance changes sharply across recordings and subjects, so generalization has not been established.",
            "Body",
        )
    )
    story.append(
        bullet_list(
            [
                "The leakage-safer patient-dependent EfficientNet-SVM experiment achieved mean test ROC-AUC 0.670 +/- 0.245 across five seizure-cluster folds, but fold ROC-AUC ranged from 0.343 to 0.985.",
                "The best patient-independent result on chb05 was the dual-graph GNN (ROC-AUC 0.637, PR-AUC 0.097), but the reciprocal held-out chb03 result fell to ROC-AUC 0.408 and PR-AUC 0.068.",
                "MFCC-Siamese, seizure-only MFCC, SALT-inspired detection, and LaBraM transfer did not provide robust preictal generalization under the tested splits.",
                "Strong ranking metrics from early single-split experiments were often paired with zero sensitivity or failed validation, demonstrating why validation/test separation and repeated folds are essential.",
                "The current system must be described as an offline research prototype. It is not clinically validated and is not suitable for treatment, EHR, or neurostimulation decisions.",
            ]
        )
    )
    story.append(
        callout(
            "Overall conclusion",
            "Implementation is technically functional and proposal-aligned at prototype level. Patient-dependent prediction shows limited but unstable signal. Patient-independent prediction remains subject-dependent and inconclusive. The next implementation priority is the offline agent/event pipeline, followed by broader subject-level evaluation rather than additional tuning on chb03 or chb05.",
            "gold",
        )
    )
    story.append(PageBreak())

    add_section(story, "2", "Data foundation and evaluation protocol")
    story.append(
        p(
            "The corrected global-v2 archives label each subject on a continuous summary-clock timeline, allowing a 30-minute preictal interval to cross EDF recording boundaries. Windows are 10 seconds long with a 5-second stride, 19 bipolar channels, and 256 Hz sampling. Ictal windows are excluded from early-warning prediction; the 30-minute postictal recovery interval is excluded from ordinary interictal training.",
            "Body",
        )
    )
    dataset_rows = [[s, f"{n:,}", f"{i:,}", f"{pr:,}", f"{ic:,}", status] for s, n, i, pr, ic, status in DATASETS]
    story.append(
        data_table(
            ["Subject", "Windows", "Interictal", "Preictal", "Ictal", "Archive status"],
            dataset_rows,
            [20 * mm, 23 * mm, 27 * mm, 25 * mm, 20 * mm, 60 * mm],
            font_size=7.2,
        )
    )
    story.append(Spacer(1, 7))
    story.append(
        callout(
            "chb12 exclusion",
            "chb12_27.edf, chb12_28.edf, and chb12_29.edf did not support the required montage. Those files contain 13 seizures, so chb12 was preserved as an incomplete archive but excluded from the main six-subject training cohort.",
            "red",
        )
    )
    story.append(p("Primary patient-independent split", "Subsection"))
    story.append(
        data_table(
            ["Role", "Subjects", "Purpose"],
            [
                ["Training", "chb01, chb02, chb06, chb07, chb08, chb10", "Balanced sampling across subjects and classes"],
                ["Validation", "chb03", "Checkpoint and threshold selection only"],
                ["Test", "chb05", "Unseen-subject evaluation"],
                ["Reciprocal", "Validation chb05; test chb03", "Checks whether performance is subject-specific"],
            ],
            [30 * mm, 75 * mm, 70 * mm],
            font_size=7.2,
        )
    )
    story.append(p("Metric interpretation", "Subsection"))
    story.append(
        bullet_list(
            [
                "ROC-AUC evaluates ranking over thresholds; PR-AUC is more informative under preictal class imbalance.",
                "Sensitivity and specificity depend on a threshold selected exclusively on validation data.",
                "False-positive windows/hour counts overlapping windows and is not a clinical alarm rate.",
                "Alarm events are consolidated with a 30-minute refractory period. Preictal-episode sensitivity is not labelled seizure-level sensitivity because adjacent seizures can share one preictal episode.",
            ]
        )
    )
    story.append(PageBreak())

    add_section(story, "3", "Implemented model inventory")
    inventory = [
        ["Temporal LSTM", "Prediction", "Patient-dependent", "Legacy per-file", "Completed baseline"],
        ["Five-band spectral logistic model", "Prediction", "Patient-dependent", "Legacy per-file", "Completed baseline"],
        ["Multisubject spectral logistic model", "Prediction", "Patient-independent", "Legacy per-file", "Completed baseline"],
        ["LTformer-inspired detector", "Detection", "Patient-independent", "Legacy per-file", "Completed; not exact SALT"],
        ["LTformer-inspired patient model", "Detection", "Patient-dependent", "Legacy per-file", "Completed; not exact SALT"],
        ["LaBraM transfer detector", "Detection", "Patient-independent", "LaBraM 200 Hz", "Completed transfer trial"],
        ["EfficientNet-B0 + SVM", "Prediction", "Patient-dependent", "Legacy per-file", "1- and 5-epoch trials"],
        ["EfficientNet seizure-cluster CV", "Prediction", "Patient-dependent", "Global-v2 chb01", "Five folds complete"],
        ["MFCC-Siamese", "Prediction", "Patient-independent", "Global-v2", "2-subject and 6-subject"],
        ["MFCC seizure-only CNN", "Prediction", "Patient-independent", "Global-v2", "Auxiliary-loss ablation"],
        ["Dual-graph GNN", "Prediction", "Patient-independent", "Global-v2", "Forward and reciprocal tests"],
    ]
    story.append(
        data_table(
            ["Model", "Task", "Regime", "Data", "Status"],
            inventory,
            [48 * mm, 28 * mm, 36 * mm, 34 * mm, 29 * mm],
            font_size=6.8,
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        callout(
            "Naming discipline",
            "The custom LSTM-Transformer is reported as LTformer-inspired, not as a faithful SALT reproduction. The MFCC and GNN models are documented as 19-channel PoMAS adaptations because the cited papers used different channel representations and protocols.",
            "blue",
        )
    )
    story.append(p("What has not yet been completed", "Subsection"))
    story.append(
        bullet_list(
            [
                "A production-quality offline agent event store and replay orchestrator.",
                "Validated SHAP, Grad-CAM, or attention explanations connected to final predictions.",
                "A dashboard driven by real replay outputs rather than demonstration values.",
                "Siena external-dataset preprocessing and evaluation.",
                "Exact SALT self-supervised pretraining, exact six-SVM EfficientNet reproduction, or exact cited GNN/MFCC protocols.",
            ]
        )
    )
    story.append(PageBreak())

    add_section(story, "4", "Detection model results")
    detection_rows = [
        ["LTformer-inspired, patient-independent", fmt(salt_pi["accuracy"]), fmt(salt_pi["macro_f1"]), fmt(salt_pi["preictal_recall"]), "chb01/02 train; chb03 val; chb05 test"],
        ["LTformer-inspired, patient-dependent", fmt(salt_pd["accuracy"]), fmt(salt_pd["macro_f1"]), fmt(salt_pd["preictal_recall"]), "Recording split within chb01"],
        ["LaBraM transfer detector", fmt(labram["accuracy"]), fmt(labram["macro_f1"]), fmt(labram["preictal_recall"]), f"{labram['pretrained_tensors_loaded']} pretrained tensors loaded"],
    ]
    story.append(
        data_table(
            ["Model", "Accuracy", "Macro F1", "Preictal recall", "Evaluation note"],
            detection_rows,
            [63 * mm, 23 * mm, 24 * mm, 27 * mm, 38 * mm],
            font_size=7.2,
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        horizontal_bar_chart(
            "Detection models: macro F1 and preictal recall",
            [
                ("LTformer patient-independent", salt_pi["macro_f1"], salt_pi["preictal_recall"]),
                ("LTformer patient-dependent", salt_pd["macro_f1"], salt_pd["preictal_recall"]),
                ("LaBraM transfer", labram["macro_f1"], labram["preictal_recall"]),
            ],
            "Macro F1",
            "Preictal recall",
        )
    )
    story.append(
        callout(
            "Detection finding",
            "Accuracy overstated utility because interictal windows dominate. The patient-independent LTformer reached accuracy 0.788 but preictal recall only 0.020. LaBraM loaded 219 pretrained tensors yet produced zero preictal recall. The patient-dependent LTformer improved macro F1 to 0.540 but preictal recall remained 0.064.",
            "red",
        )
    )
    story.append(
        p(
            "An earlier one-epoch LaBraM console diagnostic was observed but was not retained as a separate result artifact. The saved final LaBraM JSON is therefore the only LaBraM value used in the comparison table.",
            "Small",
        )
    )
    story.append(PageBreak())

    add_section(story, "5", "Patient-dependent prediction results")
    eff_cv_macro = effcv["macro_test_metrics"]
    patient_dependent_rows = [
        ["Temporal LSTM", fmt(lstm["roc_auc"]), fmt(lstm["pr_auc"]), fmt(lstm["sensitivity"]), fmt(lstm["specificity"]), "Threshold selected 0.95; all-negative test decision"],
        ["Spectral logistic", fmt(spectral_pd["roc_auc"]), fmt(spectral_pd["pr_auc"]), fmt(spectral_pd["sensitivity"]), fmt(spectral_pd["specificity"]), "High test ranking; validation ROC-AUC 0.107"],
        ["EfficientNet-SVM, 1 epoch", fmt(eff1["test"]["roc_auc"]), fmt(eff1["test"]["pr_auc"]), fmt(eff1["test"]["sensitivity"]), fmt(eff1["test"]["specificity"]), "Validation ROC-AUC 0.300"],
        ["EfficientNet-SVM, 5 epochs", fmt(eff5["test"]["roc_auc"]), fmt(eff5["test"]["pr_auc"]), fmt(eff5["test"]["sensitivity"]), fmt(eff5["test"]["specificity"]), "Validation ROC-AUC 0.295; no gain from epochs"],
        ["EfficientNet-SVM, 5-fold cluster CV", f"{fmt(eff_cv_macro['roc_auc']['mean'])} +/- {fmt(eff_cv_macro['roc_auc']['std'])}", f"{fmt(eff_cv_macro['pr_auc']['mean'])} +/- {fmt(eff_cv_macro['pr_auc']['std'])}", fmt(eff_cv_macro["sensitivity"]["mean"]), fmt(eff_cv_macro["specificity"]["mean"]), "Primary leakage-safer patient-dependent estimate"],
    ]
    story.append(
        data_table(
            ["Model", "ROC-AUC", "PR-AUC", "Sensitivity", "Specificity", "Interpretation"],
            patient_dependent_rows,
            [48 * mm, 26 * mm, 26 * mm, 24 * mm, 24 * mm, 27 * mm],
            font_size=6.6,
        )
    )
    fold_rows = []
    chart_rows = []
    for fold in effcv["folds"]:
        test = fold["test"]
        fold_rows.append(
            [
                f"Fold {fold['fold']}",
                str(fold["test_cluster"]),
                fmt(test["roc_auc"]),
                fmt(test["pr_auc"]),
                fmt(test["sensitivity"]),
                fmt(test["specificity"]),
                fmt(test["false_positive_windows_per_hour"], 1),
            ]
        )
        chart_rows.append((f"Fold {fold['fold']}", test["roc_auc"], test["pr_auc"]))
    story.append(p("Seizure-cluster cross-validation", "Subsection"))
    story.append(
        data_table(
            ["Fold", "Test cluster", "ROC-AUC", "PR-AUC", "Sensitivity", "Specificity", "FP windows/h"],
            fold_rows,
            [22 * mm, 25 * mm, 24 * mm, 24 * mm, 26 * mm, 26 * mm, 28 * mm],
            font_size=6.9,
        )
    )
    story.append(Spacer(1, 6))
    story.append(horizontal_bar_chart("EfficientNet-SVM cluster folds", chart_rows))
    story.append(
        callout(
            "Patient-dependent finding",
            "Single-split test AUC values above 0.84 were not supported by validation. Five-fold cluster evaluation is more defensible and shows substantial nonstationarity: ROC-AUC ranges from 0.343 to 0.985 and mean false-positive windows/hour is 263 +/- 170.",
            "gold",
        )
    )
    story.append(PageBreak())

    add_section(story, "6", "Patient-independent prediction results")
    def pi_row(name, result, test_subject):
        w = result["test"]["window_metrics"]
        e = result["test"]["event_metrics"]
        return [name, test_subject, fmt(w["roc_auc"]), fmt(w["pr_auc"]), fmt(w["balanced_accuracy"]), fmt(e["preictal_episode_sensitivity"]), fmt(e["false_alarm_events_per_hour"])]

    patient_independent_rows = [
        ["Multisubject spectral logistic", "chb05", fmt(spectral_pi["roc_auc"]), fmt(spectral_pi["pr_auc"]), "-", "-", "-"],
        pi_row("MFCC-Siamese, 2 train subjects", mfcc2, "chb05"),
        pi_row("MFCC-Siamese, 6 train subjects", mfcc6, "chb05"),
        pi_row("MFCC seizure-only ablation", mfcccnn, "chb05"),
        pi_row("Dual-graph GNN, forward", gnna, "chb05"),
        pi_row("Dual-graph GNN, reciprocal", gnnb, "chb03"),
    ]
    story.append(
        data_table(
            ["Model", "Test", "ROC-AUC", "PR-AUC", "Balanced acc.", "Episode sens.", "False alarms/h"],
            patient_independent_rows,
            [48 * mm, 20 * mm, 22 * mm, 22 * mm, 24 * mm, 23 * mm, 25 * mm],
            font_size=6.5,
        )
    )
    pi_chart_rows = [
        ("Spectral logistic - chb05", spectral_pi["roc_auc"], spectral_pi["pr_auc"]),
        ("MFCC-Siamese 2 subjects - chb05", mfcc2["test"]["window_metrics"]["roc_auc"], mfcc2["test"]["window_metrics"]["pr_auc"]),
        ("MFCC-Siamese 6 subjects - chb05", mfcc6["test"]["window_metrics"]["roc_auc"], mfcc6["test"]["window_metrics"]["pr_auc"]),
        ("MFCC seizure-only - chb05", mfcccnn["test"]["window_metrics"]["roc_auc"], mfcccnn["test"]["window_metrics"]["pr_auc"]),
        ("Dual-graph GNN - chb05", gnna["test"]["window_metrics"]["roc_auc"], gnna["test"]["window_metrics"]["pr_auc"]),
        ("Dual-graph GNN - chb03", gnnb["test"]["window_metrics"]["roc_auc"], gnnb["test"]["window_metrics"]["pr_auc"]),
    ]
    story.append(Spacer(1, 7))
    story.append(horizontal_bar_chart("Patient-independent test ranking metrics", pi_chart_rows))
    story.append(
        callout(
            "Patient-independent finding",
            "The GNN is the best available branch on chb05 (ROC-AUC 0.637), but the reciprocal held-out chb03 test falls below chance (ROC-AUC 0.408). This proves subject-specific performance and prevents a claim of stable patient-independent prediction.",
            "red",
        )
    )
    story.append(PageBreak())

    add_section(story, "7", "Ablation study and causal findings")
    story.append(p("Ablation A: training-subject diversity", "Subsection"))
    story.append(
        data_table(
            ["MFCC configuration", "Train subjects", "Test ROC-AUC", "Test PR-AUC", "Outcome"],
            [
                ["Siamese", "2", fmt(mfcc2["test"]["window_metrics"]["roc_auc"]), fmt(mfcc2["test"]["window_metrics"]["pr_auc"]), "Chance-level test ranking and zero sensitivity at validation threshold"],
                ["Siamese", "6", fmt(mfcc6["test"]["window_metrics"]["roc_auc"]), fmt(mfcc6["test"]["window_metrics"]["pr_auc"]), "ROC improved, but specificity collapsed to 0.098"],
            ],
            [38 * mm, 27 * mm, 29 * mm, 28 * mm, 53 * mm],
            font_size=7.0,
        )
    )
    story.append(
        p(
            "Adding four clean training subjects increased MFCC-Siamese test ROC-AUC from 0.501 to 0.598, but it did not create a useful operating point. The model produced 71 false alarm events on chb05 and 2.10 false alarms/hour.",
            "Body",
        )
    )
    story.append(p("Ablation B: Siamese auxiliary objectives", "Subsection"))
    story.append(
        data_table(
            ["Configuration", "Patient loss", "Contrastive loss", "Val ROC", "Test ROC", "Test PR", "False alarms/h"],
            [
                ["MFCC-Siamese", "0.2", "0.1", fmt(mfcc6["validation"]["window_metrics"]["roc_auc"]), fmt(mfcc6["test"]["window_metrics"]["roc_auc"]), fmt(mfcc6["test"]["window_metrics"]["pr_auc"]), fmt(mfcc6["test"]["event_metrics"]["false_alarm_events_per_hour"])],
                ["Seizure-only MFCC", "0", "0", fmt(mfcccnn["validation"]["window_metrics"]["roc_auc"]), fmt(mfcccnn["test"]["window_metrics"]["roc_auc"]), fmt(mfcccnn["test"]["window_metrics"]["pr_auc"]), fmt(mfcccnn["test"]["event_metrics"]["false_alarm_events_per_hour"])],
            ],
            [38 * mm, 23 * mm, 27 * mm, 22 * mm, 22 * mm, 21 * mm, 22 * mm],
            font_size=6.8,
        )
    )
    story.append(
        p(
            "Removing patient-identity and contrastive losses did not materially improve PR-AUC or false alarms. The primary limitation is therefore the cross-subject MFCC representation and domain shift, not just the Siamese loss formulation.",
            "Body",
        )
    )
    story.append(p("Ablation C: reciprocal GNN holdout", "Subsection"))
    story.append(
        data_table(
            ["Validation", "Test", "Test ROC-AUC", "Test PR-AUC", "Balanced acc.", "False alarms/h"],
            [
                ["chb03", "chb05", fmt(gnna["test"]["window_metrics"]["roc_auc"]), fmt(gnna["test"]["window_metrics"]["pr_auc"]), fmt(gnna["test"]["window_metrics"]["balanced_accuracy"]), fmt(gnna["test"]["event_metrics"]["false_alarm_events_per_hour"])],
                ["chb05", "chb03", fmt(gnnb["test"]["window_metrics"]["roc_auc"]), fmt(gnnb["test"]["window_metrics"]["pr_auc"]), fmt(gnnb["test"]["window_metrics"]["balanced_accuracy"]), fmt(gnnb["test"]["event_metrics"]["false_alarm_events_per_hour"])],
            ],
            [31 * mm, 27 * mm, 30 * mm, 28 * mm, 28 * mm, 31 * mm],
            font_size=7.0,
        )
    )
    story.append(
        callout(
            "Causal conclusion",
            "Performance follows the held-out subject more than the model-selection role: chb05 is consistently easier, while chb03 remains difficult. The dominant problem is inter-subject distribution shift.",
            "gold",
        )
    )
    story.append(PageBreak())

    add_section(story, "8", "Cross-experiment findings and limitations")
    story.append(p("Findings supported by the completed experiments", "Subsection"))
    story.append(
        bullet_list(
            [
                "Correct global labels and subject/recording separation materially change the credibility of reported performance. Early single-split scores cannot be treated as final estimates.",
                "More training epochs are not automatically beneficial. EfficientNet test ROC-AUC decreased from 0.858 at one epoch to 0.849 at five epochs while validation remained poor.",
                "More training subjects help some ranking metrics but do not remove subject-domain shift. MFCC and GNN results remain unstable across unseen patients.",
                "Thresholded sensitivity can be misleading. MFCC-Siamese reached test sensitivity 0.970 only by reducing specificity to 0.098 and generating 71 false alarm events.",
                "A zero false-alarm result may represent an all-negative classifier. The two-subject MFCC model had zero test alarms because its validation threshold exceeded every test score.",
                "AUC and PR-AUC must be considered with validation behavior, class prevalence, sensitivity, specificity, and event-level false alarms.",
            ]
        )
    )
    story.append(p("Current limitations", "Subsection"))
    limitations = [
        ["Cohort", "Eight clean global-v2 subjects are available; this remains a small subset of CHB-MIT."],
        ["External validity", "Siena has not yet been implemented or evaluated."],
        ["Protocol fidelity", "SALT, EfficientNet-SVM, MFCC, and GNN are proposal-aligned adaptations, not exact paper reproductions."],
        ["Alarm definition", "The 30-minute refractory event metric is an offline prototype, not a clinically validated SOP/SPH protocol."],
        ["Model selection", "chb03 and chb05 have now both been used as validation/test subjects; further tuning against them risks overfitting."],
        ["Explainability", "Existing XAI modules are not yet validated against the final model outputs."],
        ["Deployment", "No clinical, EHR, wearable, or stimulation integration has been validated."],
    ]
    story.append(data_table(["Area", "Limitation"], limitations, [40 * mm, 135 * mm], font_size=7.4))
    story.append(Spacer(1, 8))
    story.append(
        callout(
            "Interpretation boundary",
            "The project demonstrates implementation feasibility and exposes generalization failure under stricter evaluation. It does not demonstrate reliable seizure forecasting, clinical efficacy, or readiness for deployment.",
            "red",
        )
    )
    story.append(PageBreak())

    add_section(story, "9", "Proposal alignment and implementation status")
    alignment_rows = [
        ["Preprocessing", "19 channels, filtering, normalization, 10-second windows", "Implemented", "Global-v2 labels span EDF boundaries"],
        ["Detection", "SALT-style LSTM-Transformer", "Partially implemented", "LTformer-inspired; no full dual-stream SSL"],
        ["Patient-dependent prediction", "STFT + correlation + EfficientNet-B0 + SVM", "Implemented adaptation", "Single SVM and 19 channels; five-fold cluster CV complete"],
        ["Patient-independent MFCC", "MFCC-CNN/Siamese", "Implemented adaptation", "Temporal MFCC maps, 2- and 6-subject studies, loss ablation"],
        ["Patient-independent GNN", "Physical and subject-specific graph information", "Implemented adaptation", "Shared-electrode physical graph + functional correlation graph"],
        ["Multi-agent/CEP", "Monitoring, prediction, alert agents", "Scaffold only", "Offline persistent replay is next"],
        ["XAI", "SHAP, Grad-CAM, attention", "Scaffold only", "Not yet validated with final prediction path"],
        ["Dashboard", "Risk timeline and tiered alerts", "Demonstration only", "Current dashboard is not connected to real replay"],
        ["External validation", "CHB-MIT and Siena", "CHB-MIT only", "Siena remains optional stretch work"],
    ]
    story.append(
        data_table(
            ["Component", "Proposed method", "Status", "Evidence/qualification"],
            alignment_rows,
            [38 * mm, 52 * mm, 35 * mm, 50 * mm],
            font_size=6.7,
        )
    )
    story.append(Spacer(1, 8))
    story.append(
        callout(
            "Scope decision",
            "PoMAS should remain an offline, retrospective research prototype. Simulated alerts, event logs, and decision-support demonstrations are in scope. Live EHR integration, clinician notification, neurostimulation, and medical-device claims are out of scope.",
            "blue",
        )
    )
    story.append(p("What is defensible to claim now", "Subsection"))
    story.append(
        bullet_list(
            [
                "A modular PoMAS research pipeline has been implemented from raw CHB-MIT EDF files to protected model artifacts and repeatable evaluation.",
                "Patient-dependent and patient-independent modelling branches have been compared under increasingly strict data splits.",
                "Corrected evaluation reveals substantial recording and subject nonstationarity that earlier single-split scores obscured.",
                "Negative and unstable results are retained as formal ablations rather than discarded.",
            ]
        )
    )
    story.append(PageBreak())

    add_section(story, "10", "Conclusions and recommended next work")
    story.append(p("Final conclusions", "Subsection"))
    conclusions = [
        ["1", "The data pipeline is the strongest completed contribution.", "Global timeline labels, protected HDF5 archives, and disjoint splits provide a reproducible experimental base."],
        ["2", "Patient-dependent prediction contains signal but is unstable.", "EfficientNet-SVM five-fold ROC-AUC averages 0.670 with very high fold variance and false-positive windows."],
        ["3", "Patient-independent prediction is not established.", "GNN and MFCC results change materially between chb03 and chb05. Reciprocal GNN testing demonstrates subject-domain shift."],
        ["4", "Accuracy alone is inadequate.", "Macro F1, preictal recall, PR-AUC, validation behavior, and false alarm events expose failures hidden by accuracy or isolated AUC values."],
        ["5", "Further tuning on the same held-out subjects should stop.", "chb03 and chb05 have served as both validation and test targets. Additional tuning would weaken the evidential boundary."],
        ["6", "The remaining thesis value is integration and rigorous reporting.", "Offline agents, persistent events, reproducible replay, and transparent limitations are higher priority than another model search."],
    ]
    story.append(data_table(["#", "Conclusion", "Evidence"], conclusions, [12 * mm, 65 * mm, 98 * mm], font_size=7.2))
    story.append(Spacer(1, 10))
    story.append(p("Recommended implementation order", "Subsection"))
    story.append(
        data_table(
            ["Priority", "Work item", "Acceptance criterion"],
            [
                ["1", "Offline agent replay and persistent event store", "Replays saved predictions in timestamp order; logs monitoring, risk, and simulated alerts reproducibly"],
                ["2", "CEP alarm rules", "Consecutive-window logic, refractory period, thresholds, and alert tiers are tested and auditable"],
                ["3", "Unified comparison artifact", "One machine-readable table covering all model splits, metrics, thresholds, and caveats"],
                ["4", "Validated explanation for selected branch", "Explanation corresponds to the actual final score path and passes sanity checks"],
                ["5", "Dashboard connection", "Displays real offline replay events; no random risk values"],
                ["6", "Optional expanded subject CV or Siena", "Attempt only after core integration and thesis chapters are secure"],
            ],
            [24 * mm, 61 * mm, 90 * mm],
            font_size=7.1,
        )
    )
    story.append(Spacer(1, 9))
    story.append(
        callout(
            "Recommended thesis framing",
            "PoMAS is a proof-of-concept multi-agent framework evaluated retrospectively on public EEG. Its contribution is a reproducible integration and evaluation workflow that identifies when apparently strong window-level models fail under recording-level, seizure-cluster, and subject-level generalization tests.",
            "green",
        )
    )
    story.append(PageBreak())

    add_section(story, "Appendix A", "Saved evidence artifacts")
    artifact_rows = [
        ["Temporal LSTM", "output/baseline_chb01_results.json"],
        ["Spectral patient-dependent", "output/spectral_baseline_results.json"],
        ["Spectral patient-independent", "output/multisubject_spectral_test_chb05.json"],
        ["LTformer patient-independent", "output/salt_ltformer_detector_results.json"],
        ["LTformer patient-dependent", "output/salt_ltformer_chb01_patient_dependent_results.json"],
        ["LaBraM transfer", "output/labram_detector_results.json"],
        ["EfficientNet 1 epoch", "output/efficientnet_chb01_prediction_results_1epoch.json"],
        ["EfficientNet 5 epochs", "output/efficientnet_chb01_prediction_results.json"],
        ["EfficientNet cluster CV", "output/efficientnet_svm_cluster_cv_chb01/cross_validation_results_all.json"],
        ["MFCC-Siamese 2 subjects", "output/mfcc_siamese_global_v2/results.json"],
        ["MFCC-Siamese 6 subjects", "output/mfcc_siamese_6train_global_v2/results.json"],
        ["MFCC seizure-only", "output/mfcc_cnn_6train_global_v2/results.json"],
        ["GNN forward", "output/gnn_6train_global_v2/results.json"],
        ["GNN reciprocal", "output/gnn_6train_reverse_global_v2/results.json"],
    ]
    story.append(data_table(["Experiment", "Saved metric artifact"], artifact_rows, [65 * mm, 110 * mm], font_size=7.0))
    story.append(Spacer(1, 10))
    story.append(p("Primary methodological references", "Subsection"))
    story.append(
        bullet_list(
            [
                "CHB-MIT Scalp EEG Database. PhysioNet: https://physionet.org/content/chbmit/1.0.0/",
                "Saadoon et al. EfficientNet-B0 and SVM seizure prediction. DOI: 10.3390/bioengineering12020109",
                "Dissanayake et al. Patient-independent seizure prediction using scalp EEG. DOI: 10.1109/JSEN.2021.3057076",
                "Dissanayake et al. Geometric deep learning for subject-independent prediction. DOI: 10.1109/JBHI.2021.3100297",
                "Xiao et al. SALT seizure detection. DOI: 10.1142/S0129065726500127",
                "LaBraM official implementation: https://github.com/935963004/LaBraM",
            ]
        )
    )
    story.append(
        p(
            "Report generation note: all quantitative values were loaded from the saved artifacts listed above. Values are rounded for presentation; source JSON files retain full precision.",
            "Small",
        )
    )
    return story


def build_pdf() -> Path:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    if FINAL_PDF.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {FINAL_PDF}")
    document = SimpleDocTemplate(
        str(FINAL_PDF),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title="PoMAS Model Implementation and Experimental Findings Report",
        author="PoMAS Research Project",
        subject="Offline EEG seizure detection and prediction model audit",
    )
    document.build(build_story(), onFirstPage=header_footer, onLaterPages=header_footer)
    return FINAL_PDF


if __name__ == "__main__":
    print(build_pdf())
