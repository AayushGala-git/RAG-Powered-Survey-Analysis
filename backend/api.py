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
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, consider restricting origins.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory to store uploaded PDFs - use /tmp for Render's ephemeral storage
PDF_DIR = "/tmp/uploaded_pdfs"
if not os.path.exists(PDF_DIR):
    os.makedirs(PDF_DIR)
    logger.info(f"Created directory: {PDF_DIR}")

# Keep track of uploaded files in memory since filesystem is ephemeral on Render
uploaded_files_registry = {}

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
    logger.info(f"Received upload request for {len(files)} files")
    saved_files = []
    
    try:
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                return JSONResponse(
                    status_code=400, 
                    content={"error": f"File {file.filename} is not a PDF"}
                )
                
            file_path = os.path.join(PDF_DIR, file.filename)
            
            # Save the file
            try:
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
            finally:
                file.file.close()  # Make sure to close the file after reading
            
            # Check if file was actually saved
            if os.path.exists(file_path):
                logger.info(f"Successfully saved file: {file_path}")
                # Register the file in our in-memory registry
                uploaded_files_registry[file.filename] = file_path
                saved_files.append(file_path)
            else:
                logger.error(f"Failed to save file: {file_path}")
                return JSONResponse(
                    status_code=500, 
                    content={"error": f"Failed to save file: {file.filename}"}
                )
                
        return {"Uploaded PDFs": [os.path.basename(f) for f in saved_files]}
    except Exception as e:
        logger.error(f"Error uploading PDFs: {str(e)}")
        return JSONResponse(
            status_code=500, 
            content={"error": f"Error uploading PDFs: {str(e)}"}
        )

@app.post("/process_pdfs/")
async def process_pdfs(llm_choice: str = Form(...), pdf_files: List[str] = Form(...)):
    logger.info(f"Processing PDFs: {pdf_files} with LLM: {llm_choice}")
    
    # Retrieve full paths from our registry
    all_selected_files = [uploaded_files_registry.get(pdf, os.path.join(PDF_DIR, pdf)) for pdf in pdf_files]
    
    # Check if files exist
    for file_path in all_selected_files:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return JSONResponse(
                status_code=404, 
                content={"error": f"File not found: {os.path.basename(file_path)}. Please upload it again."}
            )
    
    try:
        logger.info("Extracting text from PDFs...")
        raw_text = get_pdf_text(all_selected_files)
        
        logger.info("Chunking text...")
        text_chunks = get_chunks(raw_text)
        
        logger.info("Creating vector store...")
        vectorstore = get_vectorstore(text_chunks)
        
        logger.info("Creating conversation chain...")
        conversation_chain = get_conversation_chain(vectorstore, llm_choice)
        app.state.conversation = conversation_chain
        
        logger.info("PDFs processed successfully")
        return {"status": "PDFs processed successfully", "llm": llm_choice}
    except Exception as e:
        logger.error(f"Error processing PDFs: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Error processing PDFs: {str(e)}"})

@app.post("/ask_question/")
async def ask_question(question_input: QuestionInput):
    logger.info(f"Question received: {question_input.question}")
    
    if not hasattr(app.state, "conversation") or app.state.conversation is None:
        logger.error("No conversation chain found")
        return JSONResponse(status_code=400, content={"error": "No conversation chain found. Process PDFs first."})

    try:
        logger.info("Processing question...")
        response = app.state.conversation({"question": question_input.question})
        
        source_docs = response.get("source_documents", [])
        sources = []
        for doc in source_docs:
            page = doc.metadata.get("page", "Unknown")
            file_path = doc.metadata.get("source", "Unknown")
            file_name = os.path.basename(file_path)
            snippet = doc.page_content[:200]
            sources.append({
                "page": page,
                "file": file_name,
                "snippet": snippet
            })

        logger.info("Question answered successfully")
        return {
            "answer": response["answer"],
            "chat_history": response["chat_history"],
            "sources": sources
        }
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Error processing question: {str(e)}"})

@app.post("/compare_reports/")
async def compare_reports(llm_choice: str = Form(...), pdf_files: List[str] = Form(...)):
    logger.info(f"Comparing reports: {pdf_files} with LLM: {llm_choice}")
    
    if len(pdf_files) != 2:
        logger.error("Invalid number of PDFs selected for comparison")
        return JSONResponse(status_code=400, content={"error": "Please select exactly 2 PDFs for comparison."})

    # Retrieve full paths from our registry
    file_paths = [uploaded_files_registry.get(pdf, os.path.join(PDF_DIR, pdf)) for pdf in pdf_files]
    
    # Check if files exist
    for file_path in file_paths:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return JSONResponse(
                status_code=404, 
                content={"error": f"File not found: {os.path.basename(file_path)}. Please upload it again."}
            )

    try:
        summaries = {}
        llm = get_llm(llm_choice)
        
        for i, pdf in enumerate(pdf_files):
            file_path = file_paths[i]
            logger.info(f"Extracting text from: {file_path}")
            raw_text = get_pdf_text([file_path])
            
            logger.info(f"Chunking text from: {file_path}")
            text_chunks = get_chunks(raw_text)
            
            combined_text = " ".join([chunk.page_content for chunk in text_chunks])
            prompt = f"Summarize the following market research report in a concise paragraph:\n\n{combined_text}"
            
            logger.info(f"Generating summary for: {file_path}")
            summary = llm(prompt)
            summaries[pdf] = summary

        logger.info("Comparing the two reports")
        compare_prompt = (
            f"Compare the following two market research reports and highlight their similarities, differences, "
            f"and key insights:\n\nReport 1 Summary:\n{summaries[pdf_files[0]]}\n\nReport 2 Summary:\n{summaries[pdf_files[1]]}\n\nComparison:"
        )
        comparison = llm(compare_prompt)

        logger.info("Reports compared successfully")
        return {"comparison": comparison, "summaries": summaries}
    except Exception as e:
        logger.error(f"Error comparing reports: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Error comparing reports: {str(e)}"})

# Health check endpoint for Render
@app.get("/health")
async def health_check():
    logger.info("Health check endpoint called")
    return {"status": "healthy"}