from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

out = Path('output/pdf/PoMAS_Ablation_Study_So_Far.pdf')
out.parent.mkdir(parents=True, exist_ok=True)
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Small', parent=styles['BodyText'], fontSize=8, leading=10))
styles.add(ParagraphStyle(name='Note', parent=styles['BodyText'], fontSize=9, leading=12, textColor=colors.HexColor('#4a5568')))
doc = SimpleDocTemplate(str(out), pagesize=A4, rightMargin=1.6*cm, leftMargin=1.6*cm, topMargin=1.5*cm, bottomMargin=1.5*cm)
def cell(text, header=False):
    return Paragraph(str(text), styles['Small'] if not header else ParagraphStyle('HeaderCell', parent=styles['Small'], textColor=colors.white, fontName='Helvetica-Bold'))
story = [Paragraph('PoMAS Ablation Study - Results So Far', styles['Title']), Spacer(1, 8),
         Paragraph('Scope: exploratory seizure-state and preictal classification experiments conducted on CHB-MIT subjects chb01, chb02, chb03, and chb05. This report records observed results; it does not claim clinical performance.', styles['Note']), Spacer(1, 12)]

experiments = [
('Temporal LSTM baseline', 'Patient-dependent chb01 exploratory: ROC-AUC 0.725; PR-AUC 0.773; sensitivity 46.0%; specificity 86.7%. Threshold selection was not validation-safe.'),
('Spectral + logistic regression (patient-dependent)', 'chb01 exploratory: test ROC-AUC 0.899 and PR-AUC 0.921, but validation ROC-AUC was 0.107. The validation-derived threshold produced no useful alerts.'),
('Spectral + logistic regression (patient-independent)', 'Train chb01+chb02; validate chb03; test chb05: ROC-AUC 0.439; PR-AUC 0.046; preictal sensitivity 5.4%.'),
('SALT LTformer from scratch (patient-independent)', 'Same multi-subject split: accuracy 78.8%; macro-F1 0.319; preictal recall 2.0%.'),
('SALT LTformer from scratch (patient-dependent)', 'chb01 held-out seizure recordings: accuracy 45.7%; macro-F1 0.540; preictal recall 6.4%.'),
('Pretrained LaBraM (patient-independent)', 'Same multi-subject split: accuracy 69.3%; macro-F1 0.278; preictal recall 0.0%.')]
for title, detail in experiments:
    story += [Paragraph(title, styles['Heading3']), Paragraph(detail, styles['BodyText']), Spacer(1, 6)]
story += [Spacer(1, 6), Paragraph('Interpretation', styles['Heading2']),
 Paragraph('The shared patient-independent split was train chb01+chb02, validation chb03, test chb05. None of the tested models generalized reliably to the unseen patient chb05. The spectral model, SALT LTformer, and LaBraM all produced low preictal recall or below-chance ranking on this split.', styles['BodyText']), Spacer(1, 8),
 Paragraph('Why accuracy is insufficient', styles['Heading2']),
 Paragraph('Interictal windows are the dominant class. A model can obtain a high accuracy by predicting mainly interictal, while failing to identify preictal windows. Macro-F1 and preictal recall are therefore the primary indicators in this report.', styles['BodyText']), PageBreak(),
 Paragraph('Ablation Factors and Lessons', styles['Title']), Spacer(1, 10)]

factors = [[cell(x, True) for x in ['Factor', 'Variants tested', 'Observed lesson']],
 ['Training regime', 'Patient-dependent vs patient-independent', 'Cross-patient generalization was substantially harder than within-patient experiments.'],
 ['Representation', 'Raw temporal LSTM, spectral log-power, SALT LTformer, LaBraM', 'No representation produced reliable unseen-patient preictal detection in the current four-subject study.'],
 ['Pretraining', 'LaBraM official pretrained backbone vs from-scratch SALT', 'LaBraM loaded 219 pretrained tensors but did not improve test preictal recall with the current bipolar-channel mapping.'],
 ['Class imbalance', 'Weighted loss / controlled interictal downsampling', 'These mitigations did not yet provide stable held-out seizure-recording performance.'],
 ['Thresholding', 'Validation-derived thresholds', 'Thresholds must never be selected on the test recordings. High test AUC alone did not ensure usable alerts.'],
]
factors = [factors[0]] + [[cell(x) for x in row] for row in factors[1:]]
t2=Table(factors,colWidths=[4.0*cm,7.0*cm,12.0*cm],repeatRows=1)
t2.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1f4e78')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8),('GRID',(0,0),(-1,-1),0.25,colors.HexColor('#a0aec0')),('VALIGN',(0,0),(-1,-1),'TOP'),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#edf2f7')]),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
story += [t2, Spacer(1, 14), Paragraph('Current conclusion', styles['Heading2']), Paragraph('This is an early feasibility study, not final PoMAS performance. The next proposed-model experiment is patient-dependent STFT spectrogram plus inter-channel correlation features with EfficientNet-B0 and an SVM risk classifier. Final claims require recording-level cross-validation across more subjects and event-level alert metrics.', styles['BodyText'])]
doc.build(story)
print(out)
