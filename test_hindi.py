import app, os
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import enums
out='test_hindi.pdf'
font = app.ensure_font_for_language('Hindi') or getattr(app,'APP_FONT_NAME',None)
print('Using font:', font)
styles = getSampleStyleSheet()
normal = styles['Normal']
heading = ParagraphStyle('Heading', parent=styles['Heading1'], alignment=enums.TA_LEFT)
if font:
    try:
        normal.fontName = font
        heading.fontName = font
    except Exception as e:
        print('Failed to set font on styles:', e)
story = [Paragraph('प्रयोगात्मक हिन्दी पाठ', heading), Spacer(1,12), Paragraph('यह एक परीक्षण है।', normal)]
try:
    doc = SimpleDocTemplate(out)
    doc.build(story)
    print('Wrote', out, 'size', os.path.getsize(out))
except Exception as e:
    print('PDF generation failed:', e)
