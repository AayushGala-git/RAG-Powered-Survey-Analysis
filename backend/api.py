import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List
from .Llama3 import llama3
from .Mixtral import mixtral_llm
from .Phi import phi
from .Vectorstore import get_pdf_text, get_chunks, get_vectorstore
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, consider restricting origins.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory to store uploaded PDFs
PDF_DIR = "uploaded_pdfs"
if not os.path.exists(PDF_DIR):
    os.makedirs(PDF_DIR)

# Custom prompt template
custom_template = (
    "Given the following conversation and a follow-up question, rephrase the follow-up question to be "
    "a standalone question, in its original language. Chat History: {chat_history} Follow Up Input: {question} "
    "Standalone question:"
)
Standalone_Question_Prompt = PromptTemplate.from_template(custom_template)

# Model for the question input
class QuestionInput(BaseModel):
    question: str
    llm_choice: str

def get_llm(llm_choice: str):
    if llm_choice == "Llama 3.1":
        return llama3
    elif llm_choice == "Mixtral":
        return mixtral_llm
    else:
        return phi

def get_conversation_chain(vectorstore, llm_choice: str):
    llm = get_llm(llm_choice)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True, output_key='answer'
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        condense_question_prompt=Standalone_Question_Prompt,
        memory=memory,
        return_source_documents=True
    )
    return conversation_chain

@app.post("/upload_pdfs/")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    saved_files = []
    for file in files:
        file_path = os.path.join(PDF_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        saved_files.append(file_path)
    return {"Uploaded PDFs": saved_files}

@app.post("/process_pdfs/")
async def process_pdfs(llm_choice: str = Form(...), pdf_files: List[str] = Form(...)):
    all_selected_files = [os.path.join(PDF_DIR, pdf) for pdf in pdf_files]
    raw_text = get_pdf_text(all_selected_files)
    text_chunks = get_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    conversation_chain = get_conversation_chain(vectorstore, llm_choice)
    app.state.conversation = conversation_chain
    return {"status": "PDFs processed successfully", "llm": llm_choice}

@app.post("/ask_question/")
async def ask_question(question_input: QuestionInput):
    if not hasattr(app.state, "conversation") or app.state.conversation is None:
        return JSONResponse(status_code=400, content={"error": "No conversation chain found. Process PDFs first."})

    response = app.state.conversation({"question": question_input.question})
    source_docs = response.get("source_documents", [])
    sources = []
    for doc in source_docs:
        page = doc.metadata.get("page", "Unknown")
        file_name = doc.metadata.get("source", "Unknown")
        snippet = doc.page_content[:200]
        sources.append({
            "page": page,
            "file": file_name,
            "snippet": snippet
        })

    return {
        "answer": response["answer"],
        "chat_history": response["chat_history"],
        "sources": sources
    }

@app.post("/compare_reports/")
async def compare_reports(llm_choice: str = Form(...), pdf_files: List[str] = Form(...)):
    if len(pdf_files) != 2:
        return JSONResponse(status_code=400, content={"error": "Please select exactly 2 PDFs for comparison."})

    summaries = {}
    llm = get_llm(llm_choice)
    for pdf in pdf_files:
        file_path = os.path.join(PDF_DIR, pdf)
        raw_text = get_pdf_text([file_path])
        text_chunks = get_chunks(raw_text)
        combined_text = " ".join([chunk.page_content for chunk in text_chunks])
        prompt = f"Summarize the following market research report in a concise paragraph:\n\n{combined_text}"
        summary = llm(prompt)
        summaries[pdf] = summary

    compare_prompt = (
        f"Compare the following two market research reports and highlight their similarities, differences, "
        f"and key insights:\n\nReport 1 Summary:\n{summaries[pdf_files[0]]}\n\nReport 2 Summary:\n{summaries[pdf_files[1]]}\n\nComparison:"
    )
    comparison = llm(compare_prompt)

    return {"comparison": comparison, "summaries": summaries}