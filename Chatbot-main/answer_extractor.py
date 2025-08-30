from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class PDFAnswerExtractor:
    def __init__(self):
        # Load vectorstore
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectorstore = FAISS.load_local("vectorstore/db_faiss", embeddings, allow_dangerous_deserialization=True)
        
        # Load LLM
        pipe = pipeline("text-generation", model="distilgpt2", max_new_tokens=200, temperature=0.7, pad_token_id=50256)
        self.llm = HuggingFacePipeline(pipeline=pipe)
        
        # Create QA chain
        prompt = PromptTemplate(
            template="Context: {context}\n\nQuestion: {question}\n\nAnswer:",
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt}
        )
    
    def extract_answer(self, question: str):
        """Extract answer from PDF content"""
        result = self.qa_chain.invoke({"query": question})
        return result["result"]

# Example usage
if __name__ == "__main__":
    extractor = PDFAnswerExtractor()
    
    # Test questions
    questions = [
        "What is artificial intelligence?",
        "How does AI relate to librarianship?",
        "What are the main topics discussed in this document?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        answer = extractor.extract_answer(question)
        print(f"A: {answer}")