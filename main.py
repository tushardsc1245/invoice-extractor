import os
from typing import Optional
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator
import instructor
from openai import OpenAI

app = FastAPI(title="Local LLM Invoice Extraction Service")

# Initialize client (point to your local LLM server e.g. Ollama/vLLM or OpenAI-compatible API)
# For local Ollama: base_url="http://localhost:11434/v1", api_key="ollama"
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
    date: str = Field(..., description="Payment due date as YYYY-MM-DD.")

    @field_validator("currency")
    @classmethod
    iculate_currency(cls, v: str) -> str:
        return v.upper().strip()

@app.post("/extract", response_model=InvoiceResponse, status_code=status.HTTP_200_OK)
def extract_invoice(payload: InvoiceRequest):
    # Handle malformed or empty input gracefully (avoid HTTP 500)
    if not payload or not payload.text or not payload.text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Input text cannot be empty or missing."
        )

    try:
        # Use Instructor + LLM to extract structured data matching InvoiceResponse
        response = client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            response_model=InvoiceResponse,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert invoice data extractor. Extract the vendor name, "
                        "total amount due, 3-letter uppercase currency code, and payment due date "
                        "in YYYY-MM-DD format from the provided invoice text."
                    ),
                },
                {"role": "user", "content": payload.text},
            ],
            max_retries=2,
        Omit reasoning, return valid JSON only."
        )
        return response
    except Exception as e:
        # Fallback error handling to prevent 500 crashes on parsing/model errors
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Failed to extract structured data from text: {str(e)}"
        )

@app.get("/health")
def health_check():
    return {"status": "healthy"}
