import streamlit as st
import requests
import os

# Read the backend URL from the environment variable, with a default for local testing.
BASE_API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

# Debug: show the backend URL on the app
st.write("Using backend URL:", BASE_API_URL)

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
    </style>
    """,
    unsafe_allow_html=True,
)

# Function to upload PDFs
def upload_pdfs(files):
    upload_url = f"{BASE_API_URL}/upload_pdfs/"
    file_data = [("files", (file.name, file, "application/pdf")) for file in files]
    response = requests.post(upload_url, files=file_data)
    st.write("Upload Response Code:", response.status_code)
    st.write("Upload Response Text:", response.text)
    try:
        return response.json()
    except Exception as e:
        st.error(f"Error parsing JSON: {e}")
        return {}

# Function to process PDFs
def process_pdfs(selected_pdfs, llm_choice):
    process_url = f"{BASE_API_URL}/process_pdfs/"
    data = {"llm_choice": llm_choice, "pdf_files": selected_pdfs}
    response = requests.post(process_url, data=data)
    return response.json()

# Function to ask a question
def ask_question(question, llm_choice):
    ask_url = f"{BASE_API_URL}/ask_question/"
    json_data = {"question": question, "llm_choice": llm_choice}
    response = requests.post(ask_url, json=json_data)
    return response.json()

# Function to compare reports
def compare_reports(selected_pdfs, llm_choice):
    compare_url = f"{BASE_API_URL}/compare_reports/"
    data = {"llm_choice": llm_choice, "pdf_files": selected_pdfs}
    response = requests.post(compare_url, data=data)
    return response.json()

# Streamlit UI
st.markdown("<h1>RAG-Power Survey Analysis</h1>", unsafe_allow_html=True)

# PDF Upload section
st.markdown("<h2>Upload PDFs</h2>", unsafe_allow_html=True)
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    if st.button("Upload PDFs"):
        with st.spinner("Uploading PDFs..."):
            upload_response = upload_pdfs(uploaded_files)
        st.success(f"Uploaded PDFs: {upload_response.get('Uploaded PDFs')}")

# LLM Selection and Process PDFs section
st.markdown("<h2>Process PDFs</h2>", unsafe_allow_html=True)
saved_pdfs = st.multiselect("Select previously uploaded PDFs", [file.name for file in uploaded_files])
llm_choice = st.selectbox("Select LLM", ["Mixtral", "Phi", "Llama 3.1"])

if st.button("Process PDFs"):
    if saved_pdfs and llm_choice:
        with st.spinner("Processing PDFs..."):
            process_response = process_pdfs(saved_pdfs, llm_choice)
        st.success(f"Processed PDFs with {llm_choice}: {process_response.get('status')}")
    else:
        st.error("Please select PDFs and an LLM to process")

# Ask Question section
st.markdown("<h2>Ask a Question</h2>", unsafe_allow_html=True)
question = st.text_input("Type your question here")

if st.button("Ask Question"):
    if question and llm_choice:
        with st.spinner("Generating answer..."):
            question_response = ask_question(question, llm_choice)
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
        for msg in question_response.get("chat_history", []):
            st.write(msg.get("content"))
    else:
        st.error("Please enter a question and select an LLM to ask")

# New section: Compare Reports
st.markdown("<h2>Compare Reports</h2>", unsafe_allow_html=True)
compare_pdfs = st.multiselect("Select exactly 2 PDFs to compare", [file.name for file in uploaded_files])
llm_choice_compare = st.selectbox("Select LLM for Comparison", ["Mixtral", "Phi", "Llama 3.1"], key="compare")

if st.button("Compare Reports"):
    if compare_pdfs and len(compare_pdfs) == 2 and llm_choice_compare:
        with st.spinner("Comparing reports..."):
            compare_response = compare_reports(compare_pdfs, llm_choice_compare)
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