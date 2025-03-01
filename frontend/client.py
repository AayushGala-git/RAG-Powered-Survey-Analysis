import streamlit as st
import requests
import os
import time

# Read the backend URL from the environment variable, with a default for local testing
BASE_API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").strip()
# Remove any trailing parentheses that might have been mistakenly added
if BASE_API_URL.endswith(')'):
    BASE_API_URL = BASE_API_URL[:-1]

# Add a health check function to verify backend is available
def check_backend_health():
    try:
        response = requests.get(f"{BASE_API_URL}/health", timeout=10)
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        st.error(f"Backend connection error: {str(e)}")
        return False

# Add a retry mechanism for important requests
def request_with_retry(method, url, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            response = method(url, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                st.error(f"Request failed after {max_retries} attempts: {str(e)}")
                if "files" in kwargs and method == requests.post:
                    st.error("File upload failed. Please try again with smaller files or fewer files.")
                return None
            st.warning(f"Request failed, retrying ({attempt+1}/{max_retries})...")
            time.sleep(2)  # Wait before retrying

# Custom CSS for UI enhancements using given HEX colors
st.markdown(
    """
    <style>
    h1 {
        color: #e14ed2;
        text-align: center;
    }
    h2 {
        color: #3edbda;
    }
    h3 {
        color: #3edbda;
    }
    .stButton>button {
        background-color: #e14ed2;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #3edbda;
    }
    .error-message {
        color: red;
        background-color: #ffeeee;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to upload PDFs
def upload_pdfs(files):
    upload_url = f"{BASE_API_URL}/upload_pdfs/"
    file_data = [("files", (file.name, file, "application/pdf")) for file in files]
    return request_with_retry(requests.post, upload_url, files=file_data)

# Function to process PDFs
def process_pdfs(selected_pdfs, llm_choice):
    process_url = f"{BASE_API_URL}/process_pdfs/"
    data = {"llm_choice": llm_choice, "pdf_files": selected_pdfs}
    return request_with_retry(requests.post, process_url, data=data)

# Function to ask a question
def ask_question(question, llm_choice):
    ask_url = f"{BASE_API_URL}/ask_question/"
    json_data = {"question": question, "llm_choice": llm_choice}
    return request_with_retry(requests.post, ask_url, json=json_data)

# Function to compare reports
def compare_reports(selected_pdfs, llm_choice):
    compare_url = f"{BASE_API_URL}/compare_reports/"
    data = {"llm_choice": llm_choice, "pdf_files": selected_pdfs}
    return request_with_retry(requests.post, compare_url, data=data)

# Add session state to keep track of uploaded files
if 'uploaded_file_names' not in st.session_state:
    st.session_state.uploaded_file_names = []

# Check if the backend is available
backend_available = check_backend_health()
if not backend_available:
    st.warning(f"⚠️ Unable to connect to the backend at {BASE_API_URL}. Some features may not work.")
    st.info("If you're experiencing connection issues, please ensure the backend service is running and properly configured.")
else:
    st.success(f"✅ Connected to backend at {BASE_API_URL}")

# Streamlit UI
st.markdown("<h1>RAG-Power Survey Analysis</h1>", unsafe_allow_html=True)

# Add information about the deployed app
st.markdown("This application uses Retrieval-Augmented Generation (RAG) to analyze and compare market research reports.")

# PDF Upload section
st.markdown("<h2>Upload PDFs</h2>", unsafe_allow_html=True)
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Upload PDFs"):
        with st.spinner("Uploading PDFs..."):
            upload_response = upload_pdfs(uploaded_files)
            if upload_response and "Uploaded PDFs" in upload_response:
                # Update our session state with the new file names
                st.session_state.uploaded_file_names = [f.name for f in uploaded_files]
                st.success(f"Uploaded PDFs successfully!")

# LLM Selection and Process PDFs section
st.markdown("<h2>Process PDFs</h2>", unsafe_allow_html=True)
saved_pdfs = st.multiselect("Select previously uploaded PDFs", st.session_state.uploaded_file_names)
llm_choice = st.selectbox("Select LLM", ["Mixtral", "Phi", "Llama 3.1"])

if st.button("Process PDFs"):
    if saved_pdfs and llm_choice:
        with st.spinner("Processing PDFs... This may take a while for large files."):
            process_response = process_pdfs(saved_pdfs, llm_choice)
            if process_response and "status" in process_response:
                st.success(f"Processed PDFs with {llm_choice}: {process_response.get('status')}")
            elif process_response and "error" in process_response:
                st.error(f"Error: {process_response.get('error')}")
    else:
        st.error("Please select PDFs and an LLM to process")

# Ask Question section
st.markdown("<h2>Ask a Question</h2>", unsafe_allow_html=True)
question = st.text_input("Type your question here")

if st.button("Ask Question"):
    if question and llm_choice:
        with st.spinner("Generating answer... This may take a moment."):
            question_response = ask_question(question, llm_choice)
            if question_response:
                if "error" in question_response:
                    st.error(question_response["error"])
                else:
                    st.markdown("<h3>Answer:</h3>", unsafe_allow_html=True)
                    st.write(question_response.get("answer"))
                    
                    st.markdown("<h3>Sources:</h3>", unsafe_allow_html=True)
                    sources = question_response.get("sources", [])
                    if sources:
                        for idx, src in enumerate(sources, start=1):
                            st.write(f"**Source {idx}** - File: `{src['file']}`, Page: `{src['page']}`")
                            with st.expander("Snippet"):
                                st.write(src["snippet"])
                    else:
                        st.write("No sources returned.")
                    
                    st.markdown("<h3>Chat History:</h3>", unsafe_allow_html=True)
                    chat_history = question_response.get("chat_history", [])
                    if chat_history:
                        for msg in chat_history:
                            st.write(msg.get("content", ""))
                    else:
                        st.write("No chat history available.")
    else:
        st.error("Please enter a question and select an LLM to ask")

# New section: Compare Reports
st.markdown("<h2>Compare Reports</h2>", unsafe_allow_html=True)
compare_pdfs = st.multiselect("Select exactly 2 PDFs to compare", st.session_state.uploaded_file_names, key="compare_select")
llm_choice_compare = st.selectbox("Select LLM for Comparison", ["Mixtral", "Phi", "Llama 3.1"], key="compare")

if st.button("Compare Reports"):
    if compare_pdfs and len(compare_pdfs) == 2 and llm_choice_compare:
        with st.spinner("Comparing reports... This may take a while for detailed analysis."):
            compare_response = compare_reports(compare_pdfs, llm_choice_compare)
            if compare_response:
                if "error" in compare_response:
                    st.error(compare_response["error"])
                else:
                    st.markdown("<h3>Comparison:</h3>", unsafe_allow_html=True)
                    st.write(compare_response.get("comparison"))
                    st.markdown("<h3>Individual Summaries:</h3>", unsafe_allow_html=True)
                    summaries = compare_response.get("summaries", {})
                    for file, summary in summaries.items():
                        st.write(f"**{file}**:")
                        st.write(summary)
    else:
        st.error("Please select exactly 2 PDFs and an LLM to compare")
