import os
# Disable tokenizers parallelism to avoid fork warnings.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from PyPDF2 import PdfReader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
import chromadb
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS

# Updated function to extract text from PDFs with OCR fallback for image-based text.
# Now, each page is converted into its own Document with metadata containing the source and page number.
def get_pdf_text(docs):
    documents_text = []
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        # Process each page individually
        for page_num, page in enumerate(pdf_reader.pages, start=1):
            page_text = page.extract_text()
            final_text = ""
            if page_text and page_text.strip():
                final_text = page_text
            else:
                print(f"Warning: No text extracted from page {page_num} in {pdf}, attempting OCR.")
                try:
                    from pdf2image import convert_from_path
                    import pytesseract
                    from PIL import ImageOps
                    # Convert the current page to an image with high DPI (e.g., 300)
                    images = convert_from_path(pdf, first_page=page_num, last_page=page_num, dpi=300)
                    if images:
                        # Preprocess the image: convert to grayscale and auto-contrast to improve OCR accuracy
                        image = images[0].convert("L")
                        enhanced_image = ImageOps.autocontrast(image)
                        ocr_text = pytesseract.image_to_string(enhanced_image)
                        if ocr_text and ocr_text.strip():
                            print(f"OCR succeeded on page {page_num} in {pdf}.")
                            final_text = ocr_text
                        else:
                            print(f"Warning: OCR returned no text on page {page_num} in {pdf}.")
                    else:
                        print(f"Warning: Could not convert page {page_num} to image for OCR in {pdf}.")
                except Exception as e:
                    print(f"Warning: OCR failed for page {page_num} in {pdf}: {e}")
            if not final_text.strip():
                print(f"Warning: No text extracted from page {page_num} in {pdf}")
            # Create a Document for each page with metadata for source and page number
            document = Document(
                page_content=final_text,
                metadata={"source": pdf, "page": page_num}
            )
            documents_text.append(document)
    return documents_text

# Converting text to chunks
def get_chunks(raw_text_documents):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(raw_text_documents)
    print(f"Number of text chunks generated: {len(chunks)}")
    return chunks

# Using all-MiniLM embeddings model and FAISS to get vectorstore
def get_vectorstore(chunked_text):
    if not chunked_text or len(chunked_text) == 0:
        raise ValueError("No text chunks provided to generate embeddings.")

    # Instantiate the embeddings model
    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # Generate a sample embedding to verify that embeddings are being generated
    try:
        sample_embedding = embeddings_model.embed_documents([chunked_text[0].page_content])
        if sample_embedding and sample_embedding[0]:
            print(f"Sample embedding length: {len(sample_embedding[0])}")
        else:
            print("Warning: Sample embedding is empty.")
    except Exception as e:
        print(f"Error generating sample embedding: {e}")

    # Create the FAISS vectorstore from documents and embeddings
    try:
        vectordb = FAISS.from_documents(documents=chunked_text, embedding=embeddings_model)
    except Exception as e:
        print(f"Error creating FAISS vectorstore: {e}")
        raise e

    return vectordb