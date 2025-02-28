import os
import io
import pytest
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_upload_pdfs():
    # Create a dummy PDF file in memory
    dummy_pdf = io.BytesIO(b"%PDF-1.4 dummy content")
    dummy_pdf.name = "dummy.pdf"
    response = client.post(
        "/upload_pdfs/",
        files={"files": (dummy_pdf.name, dummy_pdf, "application/pdf")}
    )
    # Verify the endpoint responds with a 200 status code and contains the key "Uploaded PDFs"
    assert response.status_code == 200
    json_data = response.json()
    assert "Uploaded PDFs" in json_data

def test_process_pdfs_and_ask_question():
    # Create a dummy PDF for processing
    dummy_pdf = io.BytesIO(b"%PDF-1.4 dummy processing content")
    dummy_pdf.name = "dummy_process.pdf"
    
    # Upload the dummy PDF
    upload_response = client.post(
        "/upload_pdfs/",
        files={"files": (dummy_pdf.name, dummy_pdf, "application/pdf")}
    )
    assert upload_response.status_code == 200

    # Process the uploaded PDF
    process_response = client.post(
        "/process_pdfs/",
        data={"llm_choice": "Phi", "pdf_files": [dummy_pdf.name]}
    )
    assert process_response.status_code == 200
    json_process = process_response.json()
    assert "status" in json_process

    # Ask a question using the processed data
    question_payload = {"question": "What is dummy content?", "llm_choice": "Phi"}
    ask_response = client.post("/ask_question/", json=question_payload)
    assert ask_response.status_code == 200
    json_data = ask_response.json()
    assert "answer" in json_data
    # Ensure the answer is a non-empty string
    assert isinstance(json_data["answer"], str) and len(json_data["answer"]) > 0