import json
import os

from fastapi import FastAPI, HTTPException
from google import genai
from pydantic import BaseModel

app = FastAPI(title="Invoice Intelligence API")

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key) if api_key else None


class ExtractRequest(BaseModel):
    document_id: str
    text: str
    schema: dict


EXPECTED_KEYS = {
    "vendor",
    "currency",
    "total_amount",
    "invoice_date",
    "due_in_days",
    "is_paid",
    "priority",
    "contact_email",
    "line_items",
    "item_count",
}


@app.get("/")
def home():
    return {"status": "Invoice Intelligence API running"}


def validate_output(data: dict):

    if set(data.keys()) != EXPECTED_KEYS:
        raise ValueError(
            f"Expected keys {EXPECTED_KEYS}, got {set(data.keys())}"
        )

    data["contact_email"] = data["contact_email"].lower()

    data["total_amount"] = int(data["total_amount"])
    data["due_in_days"] = int(data["due_in_days"])
    data["item_count"] = int(data["item_count"])

    for item in data["line_items"]:
        item["quantity"] = int(item["quantity"])
        item["unit_price"] = int(item["unit_price"])

    return data


@app.post("/extract")
async def extract(req: ExtractRequest):

    if client is None:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY missing",
        )

    prompt = f"""
You are an invoice extraction engine.

Extract the invoice into JSON.

Return ONLY valid JSON.

It MUST exactly match this JSON Schema:

{json.dumps(req.schema)}

Rules:

- vendor: exactly as written
- currency: ISO4217 code
- total_amount: integer
- invoice_date: YYYY-MM-DD
- due_in_days: integer
- is_paid: boolean
- priority: one of low, normal, high, urgent
- contact_email: lowercase
- line_items: preserve order
- item_count: number of line items

Invoice:

{req.text}
"""

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        text = response.text.strip()

        # remove markdown if Gemini returns it
        text = text.replace("```json", "").replace("```", "").strip()

        result = json.loads(text)

        return validate_output(result)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )
