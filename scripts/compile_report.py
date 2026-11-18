"""Compile the evidence-backed technical report as a PDF."""
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


def main():
    out = Path("reports")
    out.mkdir(parents=True, exist_ok=True)
    pdf_path = out / "final_report.pdf"

    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                            rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    story = [Paragraph("PathSnap Technical Report", styles["Title"]), Spacer(1, 12)]

    story.append(Paragraph("1. System Summary", styles["Heading1"]))
    story.append(Paragraph(
        "PathSnap combines trip-level candidate generation, a tuned candidate classifier, "
        "directed-network HMM/Viterbi decoding, FastAPI orchestration, and a static evaluator UI.",
        styles["BodyText"],
    ))

    story.append(Paragraph("2. Candidate and Model Evidence", styles["Heading1"]))
    try:
        phase2 = pd.read_csv("results/phase2/coverage.csv")
        story.append(Paragraph(
            f"Training trip coverage: {phase2.coverage_pct.mean():.4f}% mean; "
            f"true candidate recall: {phase2.labelled_candidate_recall_pct.mean():.4f}% mean.",
            styles["BodyText"],
        ))
    except FileNotFoundError:
        story.append(Paragraph("Phase 2 coverage artifact is not available.", styles["BodyText"]))

    try:
        experiments = pd.read_csv("results/phase3/experiment_comparison.csv")
        table_data = [["Experiment", "Accuracy", "Macro-F1"]]
        table_data += [[r.experiment, f"{r.accuracy_mean:.6f}", f"{r.macro_f1_mean:.6f}"]
                       for r in experiments.itertuples()]
        table = Table(table_data, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4c78a8")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ]))
        story += [table, Spacer(1, 12)]
    except FileNotFoundError:
        story.append(Paragraph("Phase 3 comparison artifact is not available.", styles["BodyText"]))

    story.append(Paragraph("3. HMM/Viterbi Route Reconstruction", styles["Heading1"]))
    story.append(Paragraph(
        "Candidate arcs are hidden HMM states. The emission log-score is the saved model's "
        "candidate probability. Transitions use directed road-network shortest-path distance "
        "and compare that distance with GPS displacement and elapsed time.",
        styles["BodyText"],
    ))
    try:
        phase4 = pd.read_csv("results/phase4/student_test_comparison.csv")
        r = phase4.iloc[0]
        story.append(Paragraph(
            f"Permitted student-test mean classifier arc accuracy: "
            f"{r.classifier_arc_accuracy:.6f}; decoded accuracy: "
            f"{r.decoded_arc_accuracy:.6f}; decoded plausible-transition rate: "
            f"{r.decoded_plausible_transition_rate:.6f}.",
            styles["BodyText"],
        ))
    except FileNotFoundError:
        story.append(Paragraph("Phase 4 comparison artifact is not available.", styles["BodyText"]))

    story.append(Paragraph("4. API and UI", styles["Heading1"]))
    story.append(Paragraph(
        "Laptop 1 hosts FastAPI and the complete inference pipeline. The UI displays ten "
        "permitted trip choices, GPS points, predicted road segments, and aggregate "
        "server-calculated diagnostics. Ground-truth arcs and route labels are not returned.",
        styles["BodyText"],
    ))

    story.append(Paragraph("5. Deployment Status", styles["Heading1"]))
    story.append(Paragraph(
        "Local HTTP verification passed for FastAPI health/trip/match endpoints and static UI assets "
        "on 2026-11-18. Cross-laptop network deployment was not physically tested and remains "
        "marked as unverified.",
        styles["BodyText"],
    ))

    story.append(Paragraph("6. Limitations and Conclusion", styles["Heading1"]))
    story.append(Paragraph(
        "The HMM improved sequence continuity but did not improve mean pointwise arc accuracy "
        "for every trip. It also uses bounded 2 km network searches and does not reconstruct "
        "unobserved segments. The held-out evaluator route labels were not used for training "
        "or decoder selection.",
        styles["BodyText"],
    ))

    doc.build(story)
    print(f"saved {pdf_path}")


if __name__ == "__main__":
    main()
