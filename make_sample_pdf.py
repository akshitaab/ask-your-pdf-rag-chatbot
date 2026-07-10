"""Generates a small sample PDF used to test the RAG pipeline."""
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet

os.makedirs("sample_data", exist_ok=True)
doc = SimpleDocTemplate("sample_data/employee_handbook.pdf", pagesize=letter)
styles = getSampleStyleSheet()
story = []

def h(text):
    story.append(Paragraph(text, styles["Heading1"]))
    story.append(Spacer(1, 10))

def p(text):
    story.append(Paragraph(text, styles["Normal"]))
    story.append(Spacer(1, 10))

h("Nimbus Analytics — Employee Handbook")
p("Welcome to Nimbus Analytics. This handbook explains our leave policy, "
  "reimbursement process, work-from-home guidelines, and code of conduct.")
story.append(PageBreak())

h("1. Leave Policy")
p("Every full-time employee is entitled to 18 paid leaves per calendar year, "
  "comprising 12 casual leaves and 6 sick leaves. Unused casual leave can be "
  "carried forward up to a maximum of 10 days into the next year.")
story.append(PageBreak())

h("2. Work From Home Policy")
p("Employees may work from home up to 2 days per week without prior approval. "
  "For more than 2 days, written approval from the reporting manager is required.")
story.append(PageBreak())

h("3. Reimbursement Process")
p("Travel, internet, and client-meeting expenses are reimbursable up to Rs. 5000 "
  "per month. Employees must submit original bills within 15 days of the expense.")

doc.build(story)
print("Sample PDF created at sample_data/employee_handbook.pdf")