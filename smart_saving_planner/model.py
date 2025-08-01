import json
import re
from datetime import datetime
from openai import OpenAI
from weasyprint import HTML

# Initialize OpenAI client with API key
client = OpenAI(
    api_key="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"
)

def generate_saving_plan_pdf(data: dict) -> str:
    try:
        # Build the prompt messages for the assistant
        messages = [
            {
    "role": "system",
    "content": (
    "أنت مستشار مالي محترف.\n"
    "مهمتك إعداد خطة ادخار شهرية تناسب المستخدم.\n"
    "❗️اخاطبه بشكل مباشر باستخدام 'أنت' فقط.\n"
    "❗️اشرح له لماذا هذه الخطة مناسبة له، بدون تبرير حسابي مفصل.\n"
    "✅ أضف نصائح وتوجيهات عملية لتحسين قدرته على الادخار:\n"
    "- ابحث عن دخل إضافي.\n"
    "- قلل من النفقات غير الضرورية.\n"
    "- فكر بخيارات مرنة مثل تقليل الهدف أو تمديد المدة.\n"
    "\n"
    "📌 القيود:\n"
    "- لا يجب أن يزيد مبلغ الادخار عن 40٪ من دخله الشهري.\n"
    "- اترك على الأقل 60٪ من الدخل للنفقات المعيشية.\n"
    "\n"
    "📤 الإخراج يكون بصيغة JSON فقط، ويحتوي على:\n"
    "- 'reason': فقرة بأسلوب مباشر تشرح لماذا الخطة مناسبة لك، مع نصائح عملية.\n"
    "- 'plan': قائمة من الشهور والمبالغ بصيغة:\n"
    "    - 'الشهر'\n"
    "    - 'المبلغ'\n"
    "\n"
    "📛 لا تكتب أي شيء خارج JSON إطلاقاً."
)

            },
            {
                "role": "user",
                "content": (
                    f"العمر: {data['age']}\n"
                    f"مصدر الدخل: {data['income_source']}\n"
                    f"الدخل الشهري: {data['monthly_income']} ريال\n"
                    f"المبلغ المطلوب ادخاره شهرياً: {data['savings_goal']} ريال\n"
                    f"الهدف من الادخار: {data['goal_description']}\n"
                    f"بداية الادخار: {data['start_month']}\n"
                    "أنشئ الخطة بأسلوب مخاطبة مباشر وواقعي."
                )
            }
        ]

        # Get response from OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3,
            timeout=60
        )

        raw_output = response.choices[0].message.content.strip()

        # Extract JSON content using regex
        json_match = re.search(r"\{[\s\S]+\}", raw_output)
        if not json_match:
            raise ValueError("Model response is not valid JSON.")

        parsed = json.loads(json_match.group(0))
        reason = parsed["reason"]
        plan = parsed["plan"]

        # Build the HTML content for the PDF
        html_content = f"""
        <html lang="ar" dir="rtl">
        <head>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: 'Cairo', 'Amiri', sans-serif;
                    direction: rtl;
                    text-align: right;
                    padding: 50px;
                    line-height: 1.8;
                }}
                h1 {{
                    text-align: center;
                    font-size: 28px;
                    margin-bottom: 30px;
                }}
                p {{
                    font-size: 18px;
                    margin-bottom: 15px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 17px;
                    margin-top: 30px;
                }}
                th, td {{
                    border: 1px solid #555;
                    padding: 12px 16px;
                }}
                th {{
                    background-color: #f2f2f2;
                    font-weight: bold;
                    font-size: 18px;
                }}
            </style>
        </head>
        <body>
            <h1>خطة الادخار الشهرية</h1>

            <p><strong>العمر:</strong> {data['age']}</p>
            <p><strong>مصدر الدخل:</strong> {data['income_source']}</p>
            <p><strong>الدخل الشهري:</strong> {data['monthly_income']} ريال</p>
            <p><strong>المبلغ المستهدف للادخار:</strong> {data['savings_goal']} ريال</p>
            <p><strong>الهدف من الادخار:</strong> {data['goal_description']}</p>
            <p><strong>بداية الادخار:</strong> {data['start_month']}</p>

            <p><strong>سبب الخطة:</strong> {reason}</p>

            <h2 style="margin-top: 40px;">📅 جدول الادخار الشهري</h2>
            <table>
                <thead>
                    <tr>
                        <th>الشهر</th>
                        <th>مبلغ الادخار (ريال)</th>
                    </tr>
                </thead>
                <tbody>
        """

        # Generate rows from the plan data
        for item in plan:
            month = item.get("الشهر")
            amount = item.get("المبلغ")
            html_content += f"""
                <tr>
                    <td>{month}</td>
                    <td>{amount}</td>
                </tr>
            """

        html_content += """
                </tbody>
            </table>
        </body>
        </html>
        """

        # Generate PDF from HTML
        filename = f"saving_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        HTML(string=html_content).write_pdf(filename)
        return filename

    except Exception as e:
        print(f"[❌ ERROR] Failed to generate Arabic PDF: {e}")
        raise
