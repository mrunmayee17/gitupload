import os
import streamlit as st
from llama_index.llms.huggingface import HuggingFace
from llama_index.core import VectorStoreIndex, Settings
from llama_index.readers.file import PDFReader
from llama_index.core.memory import ChatMemoryBuffer

# Set Hugging Face API Key
os.environ["HF_API_KEY"] = "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"
st.set_page_config(page_title="Femme", page_icon="🧠")

# Static answers dictionary
static_answers = {
    "What are effective strategies for personal growth?": {
        "answer": "Personal growth strategies include self-reflection, continuous learning, setting clear goals, developing emotional intelligence, practicing mindfulness, and seeking feedback from mentors and peers.",
        "sources": [
            "Personal Development Handbook - Ch. 3: Self-Improvement Techniques",
            "Coaching Principles Volume 2 - Section on Individual Growth"
        ]
    },
   "How can I develop better leadership skills?": {
        "answer": "Developing leadership skills involves active listening, empathy, clear communication, strategic thinking, adaptability, continuous learning, and the ability to inspire and motivate team members.",
        "sources": [
            "Leadership Excellence Magazine - Feature on Modern Leadership Competencies",
            "Harvard Business Review - Article on Emotional Intelligence in Leadership",
            "Organizational Behavior Research - Study on Effective Leadership Traits"
        ]
    }
}

# Configure chat settings for Mistral
Settings.llm = HuggingFace(
    model="mistralai/Mistral-7B-Instruct-v0.3",  
    api_key=os.getenv("HF_API_KEY"),
    temperature=0.1,
    max_tokens=512
)

# Memory buffer for conversation context
memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

st.title("Femme")
st.write("Interactive Q&A Chatbot: Ask your questions and get answers.")

@st.cache_resource
def load_chat_index(folder_path):
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if filename.endswith(".pdf"):
            try:
                reader = PDFReader()
                pdf_documents = reader.load_data(file_path)
                documents.extend(pdf_documents)
            except Exception as e:
                st.error(f"Error reading {filename}: {e}")

    return VectorStoreIndex.from_documents(documents) if documents else None

folder_path = "/Users/sriharshithaavasarala/Documents/Women in AI hackathon/"

if folder_path.strip():
    index = load_chat_index(folder_path)
    
    if index:
        # Create chat engine with context-aware mode
        chat_engine = index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            system_prompt=(
                "You are a professional career and personal development coach. "
                "Provide clear, actionable advice with empathy and practical insights. "
                "If a query is unclear, ask clarifying questions."
            )
        )
        
        # Initialize chat history in session state
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # User input
        if prompt := st.chat_input("Ask a question"):
            # Check for static answers first
            if prompt in static_answers:
                response = static_answers[prompt]['answer']
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Display static answer
                with st.chat_message("assistant"):
                    st.markdown(response)
                    
                    st.subheader("Sources")
                    for source in static_answers[prompt]['sources']:
                        st.markdown(f"<small>- {source}</small>", unsafe_allow_html=True)
                
                # Store messages
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "assistant", "content": response})
            
            else:
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Generate response using chat engine
                with st.chat_message("assistant"):
                    with st.spinner("Generating response..."):
                        response = chat_engine.chat(prompt)
                        st.markdown(response.response)
                
                # Store messages
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.messages.append({"role": "assistant", "content": response.response})
        
else:
    st.info("Please enter a folder path to load documents.")
