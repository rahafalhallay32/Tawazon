# main.py
from fastapi import FastAPI, Form
from fastapi.responses import FileResponse
from model import generate_saving_plan_pdf

app = FastAPI()

@app.post("/generate")
def generate_plan(
    age: int = Form(...),
    income_source: str = Form(...),
    monthly_income: float = Form(...),
    savings_goal: float = Form(...),
    goal_description: str = Form(...),
    start_month: str = Form(...)
):
    # Prepare input data dictionary
    input_data = {
        "age": age,
        "income_source": income_source,
        "monthly_income": monthly_income,
        "savings_goal": savings_goal,
        "goal_description": goal_description,
        "start_month": start_month
    }

    # Generate the PDF using OpenAI + WeasyPrint
    pdf_path = generate_saving_plan_pdf(input_data)

    # Return the generated PDF
    return FileResponse(path=pdf_path, filename="saving_plan.pdf", media_type="application/pdf")
