import json
import os

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI(title="Invoice Intelligence API")


# OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


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

    # Exact key validation
    if set(data.keys()) != EXPECTED_KEYS:
        raise ValueError(f"Unexpected keys: {list(data.keys())}")

    # Normalize email
    if data.get("contact_email"):
        data["contact_email"] = data["contact_email"].lower().strip()

    # Ensure integer values
    data["total_amount"] = int(data["total_amount"])

    data["due_in_days"] = int(data["due_in_days"])

    data["item_count"] = int(data["item_count"])

    for item in data["line_items"]:
        item["quantity"] = int(item["quantity"])
        item["unit_price"] = int(item["unit_price"])

    return data


@app.post("/extract")
async def extract_invoice(req: ExtractRequest):

    if not os.getenv("OPENAI_API_KEY"):
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY missing")

    prompt = f"""
You are an invoice extraction engine.

Extract structured invoice information from the document.

Follow these rules exactly:

- vendor: biller's proper name exactly as written
- currency: ISO 4217 code
- total_amount: integer in main currency unit
- invoice_date: YYYY-MM-DD
- due_in_days: integer
- is_paid: boolean
- priority: one of low, normal, high, urgent
- contact_email: lowercase
- line_items: preserve order
- item_count: number of line items

Return ONLY JSON matching the supplied JSON schema.

Invoice document:

{req.text}
"""

    try:
        response = client.responses.create(
            model="gpt-5.5-mini",
            input=prompt,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "invoice_extraction",
                    "schema": req.schema,
                    "strict": True,
                }
            },
        )

        result = json.loads(response.output_text)

        result = validate_output(result)

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
