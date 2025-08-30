import streamlit as st
from answer_extractor import PDFAnswerExtractor

st.set_page_config(page_title="PDF Q&A System", page_icon="📄")

st.title("📄 PDF Question & Answer System")
st.markdown("Ask questions about the uploaded PDF: *Artificial Intelligence and Librarianship*")

# Initialize the extractor
@st.cache_resource
def load_extractor():
    return PDFAnswerExtractor()

try:
    extractor = load_extractor()
    st.success("✅ PDF processed and ready for questions!")
    
    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! Ask me anything about the AI and Librarianship PDF."}
        ]
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("Ask a question about the PDF..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching PDF..."):
                answer = extractor.extract_answer(prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

except Exception as e:
    st.error(f"❌ Error loading PDF system: {e}")
    st.info("Make sure to run `python pdf_processor.py` first to process the PDF.")