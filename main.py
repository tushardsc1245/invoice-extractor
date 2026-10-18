import os
from typing import Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
import instructor
from openai import OpenAI

app = FastAPI(title="Local LLM Invoice Extraction Service")

client = instructor.from_openai(
    OpenAI(
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "dummy-key"),
    ),
    mode=instructor.Mode.JSON,
)

class InvoiceRequest(BaseModel):
    text: Optional[str] = None

class InvoiceResponse(BaseModel):
    vendor: str = Field(..., description="The vendor name as a string.")
    amount: float = Field(..., description="The total due as a number (float or int).")
    currency: str = Field(..., description="3-letter currency code, uppercase (e.g. USD).")
    date: str = Field(..., description="Payment due date as YYYY-MM-DD substring.")

    @field_validator("currency")
    @classmethod
    def validate_currency(cls, v: str) -> str:
        return v.upper().strip()

    @field_validator("amount", mode="before")
    @classmethod
    def parse_amount(cls, v):
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            cleaned = "".join(c for c in v if c.isdigit() or c == ".")
            return float(cleaned)
        return v

@app.post("/extract", response_model=InvoiceResponse, status_code=status.HTTP_200_OK)
def extract_invoice(payload: InvoiceRequest):
    if not payload or not payload.text or not payload.text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Input text cannot be empty or missing."
        )

    try:
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            response_model=InvoiceResponse,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert invoice data extractor. Extract the vendor name, "
                        "total amount due, 3-letter uppercase currency code, and payment due date "
                        "in YYYY-MM-DD format from the provided invoice text. "
                        "Omit reasoning, return valid JSON only."
                    ),
                },
                {"role": "user", "content": payload.text},
            ],
            max_retries=2,
        )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Extraction failed: {str(e)}"
        )

@app.get("/health")
def health_check():
    return {"status": "healthy"}
