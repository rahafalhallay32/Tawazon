import torch
import os
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText
from pdf2image import convert_from_path
import json

# -------------------
# Paths and settings
# -------------------
MODEL_PATH = "ChatDOC/OCRFlux-3B"
PDF_FOLDER = "../data/pdfs"
TEXT_FOLDER = "../data/json_files"
EXTRACT_PROMPT = (
    "You are analyzing an Arabic presentation slide. Extract all visible text exactly as it appears, preserving the reading direction from right to left. "
    "Detect section headers and any other main titles, and format them as <h1> headers. "
    "Format subpoints, phrases, and descriptive statements under each header using bullet points or paragraphs. "
    "Do not repeat or hallucinate content. Retain the Arabic language, and clearly structure the output in a way that reflects how a person would read and understand the slide visually."
)

# -------------------
# Load OCRFlux model
# -------------------
print("📦 Loading OCRFlux model and processor...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    device_map={"": "cpu"},
)
model.eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

# -------------------
# OCR a single image
# -------------------
def ocr_image(image_path, max_new_tokens=4096):
    image = Image.open(image_path)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"file://{image_path}"},
                {"type": "text", "text": EXTRACT_PROMPT},
            ],
        },
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = inputs.to(model.device)

    output_ids = model.generate(
        **inputs, temperature=0.0, max_new_tokens=max_new_tokens, do_sample=False
    )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return output_text[0]


# -------------------
# Process all PDFs
# -------------------
import json

def process_pdfs():
    os.makedirs(TEXT_FOLDER, exist_ok=True)
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]

    for pdf_file in pdf_files:
        pdf_path = os.path.join(PDF_FOLDER, pdf_file)
        json_output_path = os.path.join(TEXT_FOLDER, pdf_file.replace(".pdf", ".json"))

        # Skip if JSON already exists
        if os.path.exists(json_output_path):
            print(f" Skipping {pdf_file}, already processed.")
            continue

        print(f"📄 Converting {pdf_file} to images...")
        images = convert_from_path(pdf_path)

        pages = []
        for i, image in enumerate(images):
            image_path = f"temp_page_{i}.png"
            image.save(image_path)

            print(f"🔍 OCR page {i+1}/{len(images)}...")
            text = ocr_image(image_path, max_new_tokens=4096)
            pages.append({
                "page": i + 1,
                "content": text.strip()
            })

            os.remove(image_path)

        with open(json_output_path, "w", encoding="utf-8") as f:
            json.dump(pages, f, ensure_ascii=False, indent=2)

        print(f"✅ Saved JSON: {json_output_path}\n")

if __name__ == "__main__":
    process_pdfs()
