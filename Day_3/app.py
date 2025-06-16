import streamlit as st
import google.generativeai as genai
import os
import tempfile
import shutil
from typing import List, Dict, Tuple, Optional, Any
import json
import hashlib
import time
import re
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
from datetime import datetime

# Document processing imports
import PyPDF2
from docx import Document
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="🤖 Agentic RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sidebar-header {
        font-size: 1.3rem;
        color: #1f77b4;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .citation-box {
        background-color: #f0f2f6;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
    .processing-status {
        background-color: #e8f4fd;
        border: 1px solid #bee5eb;
        border-radius: 4px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .sample-question {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .sample-question:hover {
        background-color: #e9ecef;
        transform: translateY(-2px);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class DocumentChunk:
    """Represents a chunk of document content with metadata"""
    content: str
    source: str
    page: Optional[int] = None
    chunk_id: str = ""
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if not self.chunk_id:
            self.chunk_id = hashlib.md5(self.content.encode()).hexdigest()[:12]

@dataclass
class RetrievalContext:
    """Context for retrieval operations"""
    query: str
    chunks: List[DocumentChunk]
    confidence_score: float
    retrieval_strategy: str
    timestamp: datetime

@dataclass
class AgentMemory:
    """Memory system for the agentic RAG"""
    conversation_history: List[Dict[str, Any]]
    successful_retrievals: List[RetrievalContext]
    failed_queries: List[str]
    document_summaries: Dict[str, str]
    
    def add_interaction(self, query: str, response: str, confidence: float, chunks: List[DocumentChunk]):
        """Add interaction to memory"""
        self.conversation_history.append({
            "query": query,
            "response": response,
            "confidence": confidence,
            "chunks": len(chunks),
            "timestamp": datetime.now().isoformat()
        })

class DocumentProcessor:
    """Handles document loading and chunking"""
    
    def __init__(self):
        self.supported_formats = {'.pdf', '.docx', '.txt'}
    
    def extract_text_from_pdf(self, file_path: str) -> List[Tuple[str, int]]:
        """Extract text from PDF with page numbers"""
        text_chunks = []
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    if text.strip():
                        text_chunks.append((text, page_num))
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            raise
        return text_chunks
    
    def extract_text_from_docx(self, file_path: str) -> List[Tuple[str, int]]:
        """Extract text from DOCX"""
        text_chunks = []
        try:
            doc = Document(file_path)
            current_text = ""
            page_num = 1
            
            for paragraph in doc.paragraphs:
                current_text += paragraph.text + "\n"
                # Simple page break detection (approximate)
                if len(current_text) > 2000:  # Rough page size
                    if current_text.strip():
                        text_chunks.append((current_text, page_num))
                    current_text = ""
                    page_num += 1
            
            if current_text.strip():
                text_chunks.append((current_text, page_num))
                
        except Exception as e:
            logger.error(f"Error reading DOCX {file_path}: {e}")
            raise
        return text_chunks
    
    def extract_text_from_txt(self, file_path: str) -> List[Tuple[str, int]]:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                # Split into approximate pages
                chunks = [content[i:i+2000] for i in range(0, len(content), 2000)]
                return [(chunk, idx+1) for idx, chunk in enumerate(chunks) if chunk.strip()]
        except Exception as e:
            logger.error(f"Error reading TXT {file_path}: {e}")
            raise
    
    def create_smart_chunks(self, text: str, max_chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Create intelligent text chunks with semantic boundaries"""
        if len(text) <= max_chunk_size:
            return [text]
        
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Handle overlap
                if chunks and overlap > 0:
                    last_words = current_chunk.split()[-overlap//10:]  # Approximate word overlap
                    current_chunk = " ".join(last_words) + " " + sentence + " "
                else:
                    current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def process_documents(self, uploaded_files) -> List[DocumentChunk]:
        """Process uploaded documents and create chunks"""
        all_chunks = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for uploaded_file in uploaded_files:
                # Save uploaded file temporarily
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                # Extract text based on file type
                file_ext = Path(uploaded_file.name).suffix.lower()
                
                try:
                    if file_ext == '.pdf':
                        text_pages = self.extract_text_from_pdf(temp_path)
                    elif file_ext == '.docx':
                        text_pages = self.extract_text_from_docx(temp_path)
                    elif file_ext == '.txt':
                        text_pages = self.extract_text_from_txt(temp_path)
                    else:
                        st.error(f"Unsupported file format: {file_ext}")
                        continue
                    
                    # Create chunks for each page
                    for text, page_num in text_pages:
                        if text.strip():
                            chunks = self.create_smart_chunks(text)
                            for chunk_text in chunks:
                                chunk = DocumentChunk(
                                    content=chunk_text,
                                    source=uploaded_file.name,
                                    page=page_num
                                )
                                all_chunks.append(chunk)
                
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                    continue
        
        return all_chunks

class VectorStore:
    """FAISS-based vector store for document embeddings"""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.chunks = []
        self.dimension = 384  # Default for all-MiniLM-L6-v2
    
    def add_documents(self, chunks: List[DocumentChunk]):
        """Add document chunks to the vector store"""
        if not chunks:
            return
        
        # Generate embeddings
        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Initialize FAISS index
        if self.index is None:
            self.dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Add to FAISS index
        self.index.add(embeddings.astype('float32'))
        
        # Store chunks with embeddings
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            self.chunks.append(chunk)
    
    def search(self, query: str, k: int = 5, min_similarity: float = 0.3) -> List[Tuple[DocumentChunk, float]]:
        """Search for similar chunks"""
        if self.index is None or len(self.chunks) == 0:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), min(k, len(self.chunks)))
        
        # Filter by minimum similarity and return results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= min_similarity:
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def get_document_summary(self) -> Dict[str, int]:
        """Get summary of documents in the vector store"""
        summary = {}
        for chunk in self.chunks:
            if chunk.source in summary:
                summary[chunk.source] += 1
            else:
                summary[chunk.sourcjbje] = 1
        return summary

class AgenticRAG:
    """Main agentic RAG system with self-healing and reasoning capabilities"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        # Using the correct model name for Gemini
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.vector_store = VectorStore()
        self.memory = AgentMemory([], [], [], {})
        self.confidence_threshold = 0.7
        self.max_retry_attempts = 3
        
        # Initialize generation config for temperature control
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.7,
            top_p=0.8,
            top_k=64,
            max_output_tokens=8192,
        )
    
    def load_documents(self, uploaded_files) -> bool:
        """Load and process documents"""
        try:
            processor = DocumentProcessor()
            chunks = processor.process_documents(uploaded_files)
            
            if not chunks:
                return False
            
            self.vector_store.add_documents(chunks)
            
            # Generate document summaries
            self._generate_document_summaries()
            
            return True
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            return False
    
    def _generate_document_summaries(self):
        """Generate summaries for each document"""
        doc_contents = {}
        
        # Group chunks by document
        for chunk in self.vector_store.chunks:
            if chunk.source not in doc_contents:
                doc_contents[chunk.source] = []
            doc_contents[chunk.source].append(chunk.content)
        
        # Generate summaries
        for doc_name, contents in doc_contents.items():
            combined_content = "\n".join(contents[:5])  # Use first 5 chunks for summary
            
            prompt = f"""
            Analyze this document content and provide a concise summary (2-3 sentences) 
            that captures the main topics and themes:
            
            {combined_content[:2000]}
            
            Summary:
            """
            
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config
                )
                self.memory.document_summaries[doc_name] = response.text.strip()
            except Exception as e:
                logger.error(f"Error generating summary for {doc_name}: {e}")
                self.memory.document_summaries[doc_name] = "Summary unavailable"
    
    def generate_sample_questions(self, num_questions: int = 5) -> List[str]:
        """Generate sample questions based on loaded documents"""
        if not self.vector_store.chunks:
            return []
        
        # Get diverse content samples
        sample_contents = []
        step = max(1, len(self.vector_store.chunks) // num_questions)
        for i in range(0, len(self.vector_store.chunks), step):
            sample_contents.append(self.vector_store.chunks[i].content[:500])
        
        combined_content = "\n\n---\n\n".join(sample_contents)
        
        prompt = f"""
        Based on the following document content, generate {num_questions} diverse and interesting questions 
        that users might ask. The questions should:
        1. Cover different aspects of the content
        2. Range from simple factual to more analytical
        3. Be answerable from the provided content
        4. Be naturally phrased
        
        Content:
        {combined_content[:3000]}
        
        Generate exactly {num_questions} questions, one per line, without numbering:
        """
        
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )
            questions = [q.strip() for q in response.text.split('\n') if q.strip()]
            return questions[:num_questions]
        except Exception as e:
            logger.error(f"Error generating sample questions: {e}")
            return [
                "What are the main topics covered in the documents?",
                "Can you summarize the key points?",
                "What are the most important findings or conclusions?"
            ]
    
    def _evaluate_answer_confidence(self, query: str, answer: str, retrieved_chunks: List[DocumentChunk]) -> float:
        """Evaluate confidence in the generated answer"""
        if not retrieved_chunks or not answer:
            return 0.0
        
        # Simple heuristics for confidence scoring
        confidence_factors = []
        
        # Factor 1: Number of supporting chunks
        chunk_factor = min(len(retrieved_chunks) / 3.0, 1.0)
        confidence_factors.append(chunk_factor)
        
        # Factor 2: Answer length and detail
        length_factor = min(len(answer.split()) / 100.0, 1.0)
        confidence_factors.append(length_factor)
        
        # Factor 3: Presence of specific details
        detail_factor = 0.0
        if any(keyword in answer.lower() for keyword in ['specific', 'according to', 'states that', 'mentions']):
            detail_factor = 0.8
        confidence_factors.append(detail_factor)
        
        # Factor 4: No hedging language
        hedging_words = ['might', 'could', 'possibly', 'perhaps', 'unclear', 'ambiguous']
        hedging_factor = 1.0 - (sum(1 for word in hedging_words if word in answer.lower()) / len(hedging_words))
        confidence_factors.append(hedging_factor)
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def _rewrite_query(self, original_query: str, attempt: int) -> str:
        """Rewrite query for better retrieval using agentic strategies"""
        strategies = [
            f"What specific information about {original_query}?",
            f"Details regarding: {original_query}",
            f"Explain the context of: {original_query}",
            f"Provide examples related to: {original_query}",
            f"Key concepts in: {original_query}",
            f"Summary of: {original_query}"
        ]
        
        if attempt < len(strategies):
            return strategies[attempt]
        
        # Fallback: extract key terms and reformulate
        key_terms = re.findall(r'\b\w{4,}\b', original_query.lower())
        if key_terms:
            return " ".join(key_terms[:3])
        
        return original_query
    
    def _analyze_retrieval_patterns(self, query: str) -> Dict[str, Any]:
        """Analyze patterns in retrieval to improve agentic behavior"""
        analysis = {
            "query_type": "general",
            "complexity": "medium",
            "domain_specific": False,
            "requires_multiple_sources": False
        }
        
        # Query type classification
        if any(word in query.lower() for word in ['what', 'define', 'explain']):
            analysis["query_type"] = "definition"
        elif any(word in query.lower() for word in ['how', 'process', 'steps']):
            analysis["query_type"] = "procedural"
        elif any(word in query.lower() for word in ['why', 'reason', 'cause']):
            analysis["query_type"] = "causal"
        elif any(word in query.lower() for word in ['compare', 'difference', 'versus']):
            analysis["query_type"] = "comparative"
            analysis["requires_multiple_sources"] = True
        
        # Complexity assessment
        word_count = len(query.split())
        if word_count > 10:
            analysis["complexity"] = "high"
        elif word_count < 5:
            analysis["complexity"] = "low"
        
        return analysis
    
    def answer_query(self, query: str) -> Dict[str, Any]:
        """Answer a query using agentic retrieval and generation with self-healing"""
        start_time = time.time()
        
        # Analyze query patterns for agentic behavior
        query_analysis = self._analyze_retrieval_patterns(query)
        
        for attempt in range(self.max_retry_attempts):
            # Adaptive query rewriting based on analysis
            if attempt == 0:
                current_query = query
            else:
                current_query = self._rewrite_query(query, attempt - 1)
                logger.info(f"Attempt {attempt + 1}: Rewritten query: {current_query}")
            
            # Dynamic retrieval parameters based on query analysis
            k_value = 7 if query_analysis["requires_multiple_sources"] else 5
            min_sim = 0.15 if attempt > 1 else 0.25
            
            # Retrieve relevant chunks with adaptive parameters
            retrieved_chunks_with_scores = self.vector_store.search(
                current_query, 
                k=k_value,
                min_similarity=min_sim
            )
            
            if not retrieved_chunks_with_scores:
                if attempt == self.max_retry_attempts - 1:
                    # Final attempt: try with very low threshold and broader search
                    retrieved_chunks_with_scores = self.vector_store.search(
                        query, k=10, min_similarity=0.1
                    )
                    
                    if not retrieved_chunks_with_scores:
                        return {
                            "answer": "I couldn't find relevant information in the uploaded documents to answer your question. Please try rephrasing your question or check if the information might be in a different document.",
                            "confidence": 0.0,
                            "citations": [],
                            "processing_time": time.time() - start_time,
                            "attempts": attempt + 1,
                            "agent_strategy": "exhaustive_search_failed"
                        }
                else:
                    continue
            
            retrieved_chunks = [chunk for chunk, _ in retrieved_chunks_with_scores]
            
            # Prepare enhanced context with better formatting
            context_parts = []
            for i, chunk in enumerate(retrieved_chunks, 1):
                context_parts.append(
                    f"[Document {i}: {chunk.source}, Page {chunk.page}]\n{chunk.content}\n"
                )
            
            context = "\n" + "="*50 + "\n".join(context_parts)
            
            # Enhanced prompt for better generation
            prompt = f"""
            You are an expert document analyst with advanced reasoning capabilities. Your task is to provide accurate, comprehensive answers based STRICTLY on the provided document context.
            
            ANALYSIS CONTEXT:
            - Query Type: {query_analysis["query_type"]}
            - Complexity: {query_analysis["complexity"]}
            - Multi-source needed: {query_analysis["requires_multiple_sources"]}
            
            CRITICAL INSTRUCTIONS:
            1. Base your answer ONLY on the provided document context
            2. If information is insufficient, clearly state what's missing
            3. Cite specific documents when making claims
            4. Provide detailed, well-structured responses
            5. If comparing information, clearly distinguish between sources
            6. Use evidence-based reasoning
            7. Acknowledge uncertainty when appropriate
            
            DOCUMENT CONTEXT:
            {context}
            
            USER QUESTION: {query}
            
            COMPREHENSIVE ANSWER:
            """
            
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config
                )
                answer = response.text.strip()
                
                # Enhanced confidence evaluation
                confidence = self._evaluate_answer_confidence(query, answer, retrieved_chunks)
                
                # Boost confidence for successful multi-source queries
                if query_analysis["requires_multiple_sources"] and len(retrieved_chunks) > 3:
                    confidence += 0.1
                
                # Agent learning: remember successful patterns
                if confidence >= self.confidence_threshold:
                    retrieval_context = RetrievalContext(
                        query=current_query,
                        chunks=retrieved_chunks,
                        confidence_score=confidence,
                        retrieval_strategy=f"attempt_{attempt + 1}",
                        timestamp=datetime.now()
                    )
                    self.memory.successful_retrievals.append(retrieval_context)
                
                # Return result if confidence is acceptable or max attempts reached
                if confidence >= self.confidence_threshold or attempt == self.max_retry_attempts - 1:
                    # Prepare enhanced citations
                    citations = []
                    for i, chunk in enumerate(retrieved_chunks, 1):
                        citations.append({
                            "id": i,
                            "source": chunk.source,
                            "page": chunk.page,
                            "content_preview": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content,
                            "relevance_score": retrieved_chunks_with_scores[i-1][1] if i-1 < len(retrieved_chunks_with_scores) else 0.0
                        })
                    
                    # Update memory with interaction
                    self.memory.add_interaction(query, answer, confidence, retrieved_chunks)
                    
                    return {
                        "answer": answer,
                        "confidence": confidence,
                        "citations": citations,
                        "processing_time": time.time() - start_time,
                        "attempts": attempt + 1,
                        "agent_strategy": f"{query_analysis['query_type']}_analysis",
                        "query_analysis": query_analysis
                    }
            
            except Exception as e:
                logger.error(f"Error generating answer on attempt {attempt + 1}: {e}")
                if attempt == self.max_retry_attempts - 1:
                    return {
                        "answer": f"I encountered a technical error while processing your question. Error details: {str(e)}",
                        "confidence": 0.0,
                        "citations": [],
                        "processing_time": time.time() - start_time,
                        "attempts": attempt + 1,
                        "agent_strategy": "error_recovery"
                    }
        
        # Fallback response (should rarely reach here)
        return {
            "answer": "I was unable to process your query after multiple strategic attempts. Please try rephrasing your question.",
            "confidence": 0.0,
            "citations": [],
            "processing_time": time.time() - start_time,
            "attempts": self.max_retry_attempts,
            "agent_strategy": "fallback"
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status with agentic insights"""
        doc_summary = self.vector_store.get_document_summary()
        
        # Analyze conversation patterns for agentic insights
        successful_patterns = {}
        for retrieval in self.memory.successful_retrievals:
            strategy = retrieval.retrieval_strategy
            if strategy in successful_patterns:
                successful_patterns[strategy] += 1
            else:
                successful_patterns[strategy] = 1
        
        return {
            "documents_loaded": len(doc_summary),
            "total_chunks": len(self.vector_store.chunks),
            "document_summary": doc_summary,
            "conversation_history_length": len(self.memory.conversation_history),
            "memory_summaries": self.memory.document_summaries,
            "successful_retrievals": len(self.memory.successful_retrievals),
            "successful_patterns": successful_patterns,
            "failed_queries": len(self.memory.failed_queries),
            "average_confidence": np.mean([h["confidence"] for h in self.memory.conversation_history]) if self.memory.conversation_history else 0.0
        }

# Streamlit App
def main():
    # Header with enhanced styling
    st.markdown('<h1 class="main-header">🤖 Agentic RAG Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Intelligent Document Analysis with Self-Healing Retrieval & Adaptive Learning</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    if 'documents_loaded' not in st.session_state:
        st.session_state.documents_loaded = False
    if 'sample_questions' not in st.session_state:
        st.session_state.sample_questions = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = ""
    
    # Sidebar
    with st.sidebar:
        st.markdown('<h2 class="sidebar-header">🔧 Configuration</h2>', unsafe_allow_html=True)
        
        # API Key input
        api_key = st.text_input(
            "Google Gemini API Key",
            type="password",
            help="Enter your Google Gemini API key to enable the RAG system",
            placeholder="Enter your API key here..."
        )
        
        if api_key:
            if st.session_state.rag_system is None:
                try:
                    with st.spinner("🚀 Initializing Agentic RAG System..."):
                        st.session_state.rag_system = AgenticRAG(api_key)
                        st.success("✅ Agentic RAG System initialized successfully!")
                        st.session_state.processing_status = "System ready for document processing"
                except Exception as e:
                    st.error(f"❌ Failed to initialize system: {e}")
                    st.session_state.processing_status = f"Initialization failed: {e}"
                    return
        else:
            st.warning("⚠️ Please enter your Gemini API key to continue")
            st.info("💡 You can get your API key from Google AI Studio")
            return
        
        st.markdown("---")
        
        # Document upload section
        st.markdown('<h3 class="sidebar-header">📄 Document Upload</h3>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Choose documents to analyze",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt'],
            help="Upload PDF, DOCX, or TXT files for analysis"
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} file(s) selected")
            for file in uploaded_files:
                st.write(f"• {file.name} ({file.size} bytes)")
        
        if uploaded_files and not st.session_state.documents_loaded:
            if st.button("📚 Process Documents", type="primary", use_container_width=True):
                try:
                    with st.spinner("🔄 Processing documents with AI intelligence..."):
                        progress_bar = st.progress(0)
                        st.session_state.processing_status = "Loading documents..."
                        progress_bar.progress(25)
                        
                        success = st.session_state.rag_system.load_documents(uploaded_files)
                        progress_bar.progress(75)
                        if success:
                            st.session_state.documents_loaded = True
                            st.session_state.processing_status = "Documents processed successfully!"
                            
                            # Generate sample questions
                            st.session_state.sample_questions = st.session_state.rag_system.generate_sample_questions()
                            progress_bar.progress(100)
                            
                            st.success("✅ Documents processed successfully!")
                            st.balloons()
                        else:
                            st.error("❌ Failed to process documents")
                            st.session_state.processing_status = "Document processing failed"
                        
                        progress_bar.empty()
                        
                except Exception as e:
                    st.error(f"❌ Error processing documents: {e}")
                    st.session_state.processing_status = f"Processing error: {e}"
        
        # System status
        if st.session_state.documents_loaded:
            st.markdown("---")
            st.markdown('<h3 class="sidebar-header">📊 System Status</h3>', unsafe_allow_html=True)
            
            if st.button("🔍 View System Stats", use_container_width=True):
                status = st.session_state.rag_system.get_system_status()
                
                with st.expander("📈 Detailed System Status", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Documents", status["documents_loaded"])
                        st.metric("Text Chunks", status["total_chunks"])
                        st.metric("Conversations", status["conversation_history_length"])
                    
                    with col2:
                        st.metric("Successful Retrievals", status["successful_retrievals"])
                        st.metric("Avg Confidence", f"{status['average_confidence']:.2f}")
                        st.metric("Failed Queries", status["failed_queries"])
                    
                    if status["document_summary"]:
                        st.subheader("📋 Document Summary")
                        for doc, chunks in status["document_summary"].items():
                            st.write(f"• **{doc}**: {chunks} chunks")
                    
                    if status["memory_summaries"]:
                        st.subheader("🧠 Document Insights")
                        for doc, summary in status["memory_summaries"].items():
                            with st.container():
                                st.write(f"**{doc}:**")
                                st.write(summary)
    
    # Main content area
    if st.session_state.processing_status:
        st.markdown(f'<div class="processing-status">🔄 Status: {st.session_state.processing_status}</div>', 
                   unsafe_allow_html=True)
    
    if not st.session_state.documents_loaded:
        st.info("👈 Please upload and process documents in the sidebar to begin asking questions")
        
        # Show welcome message with features
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            ### 🧠 Agentic Intelligence
            - Self-healing retrieval
            - Adaptive query rewriting
            - Pattern learning from conversations
            """)
        
        with col2:
            st.markdown("""
            ### 🔍 Advanced Search
            - Semantic similarity search
            - Multi-document analysis
            - Confidence scoring
            """)
        
        with col3:
            st.markdown("""
            ### 📚 Document Support
            - PDF, DOCX, TXT files
            - Smart text chunking
            - Citation tracking
            """)
        
        return
    
    # Sample questions section
    if st.session_state.sample_questions:
        st.markdown("### 💡 Sample Questions")
        st.markdown("Click on any question to ask it, or type your own question below:")
        
        # Display sample questions in a grid
        cols = st.columns(2)
        for i, question in enumerate(st.session_state.sample_questions):
            with cols[i % 2]:
                if st.markdown(f'<div class="sample-question" onclick="document.getElementById(\'question_input\').value=\'{question}\'">{question}</div>', 
                              unsafe_allow_html=True):
                    pass
    
    # Chat interface
    st.markdown("### 💬 Ask Questions")
    
    # Question input
    user_question = st.text_input(
        "Enter your question:",
        placeholder="Ask anything about your uploaded documents...",
        key="question_input"
    )
    
    # Quick action buttons for sample questions
    if st.session_state.sample_questions:
        st.markdown("**Quick Ask:**")
        cols = st.columns(min(3, len(st.session_state.sample_questions)))
        for i, question in enumerate(st.session_state.sample_questions[:3]):
            with cols[i]:
                if st.button(f"❓ {question[:30]}{'...' if len(question) > 30 else ''}", 
                           key=f"quick_q_{i}"):
                    user_question = question
                    st.rerun()
    
    # Process question
    if user_question:
        if st.button("🚀 Get Answer", type="primary", use_container_width=True):
            try:
                with st.spinner("🤔 Analyzing documents with AI reasoning..."):
                    # Show processing steps
                    status_placeholder = st.empty()
                    
                    status_placeholder.info("🔍 Step 1: Analyzing query patterns...")
                    time.sleep(0.5)
                    
                    status_placeholder.info("🔎 Step 2: Searching relevant documents...")
                    time.sleep(0.5)
                    
                    status_placeholder.info("🧠 Step 3: Generating intelligent response...")
                    
                    result = st.session_state.rag_system.answer_query(user_question)
                    status_placeholder.empty()
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "question": user_question,
                        "result": result,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
            except Exception as e:
                st.error(f"❌ Error processing question: {e}")
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("### 📝 Conversation History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                st.markdown(f"**🙋 Question {len(st.session_state.chat_history) - i}** _{chat['timestamp']}_")
                st.markdown(f"_{chat['question']}_")
                
                result = chat['result']
                
                # Confidence indicator
                confidence = result['confidence']
                if confidence >= 0.8:
                    confidence_class = "confidence-high"
                    confidence_icon = "🟢"
                elif confidence >= 0.5:
                    confidence_class = "confidence-medium"
                    confidence_icon = "🟡"
                else:
                    confidence_class = "confidence-low"
                    confidence_icon = "🔴"
                
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                with col1:
                    st.markdown(f"**🤖 Answer:**")
                with col2:
                    st.markdown(f'{confidence_icon} <span class="{confidence_class}">Confidence: {confidence:.1%}</span>', 
                               unsafe_allow_html=True)
                with col3:
                    st.markdown(f"⏱️ {result['processing_time']:.1f}s")
                with col4:
                    st.markdown(f"🔄 {result['attempts']} attempts")
                
                # Answer text
                st.markdown(result['answer'])
                
                # Citations
                if result['citations']:
                    with st.expander(f"📚 View {len(result['citations'])} Citations", expanded=False):
                        for citation in result['citations']:
                            st.markdown(f"""
                            <div class="citation-box">
                                <strong>📄 {citation['source']}</strong> (Page {citation['page']})
                                <br>
                                <em>Relevance Score: {citation['relevance_score']:.2f}</em>
                                <br><br>
                                {citation['content_preview']}
                            </div>
                            """, unsafe_allow_html=True)
                
                # Agent insights
                if 'query_analysis' in result:
                    with st.expander("🧠 Agent Analysis", expanded=False):
                        analysis = result['query_analysis']
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Query Type:** {analysis['query_type']}")
                            st.write(f"**Complexity:** {analysis['complexity']}")
                        with col2:
                            st.write(f"**Strategy:** {result['agent_strategy']}")
                            st.write(f"**Multi-source:** {'Yes' if analysis['requires_multiple_sources'] else 'No'}")
                
                st.markdown("---")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        🤖 <strong>Agentic RAG Assistant</strong> - Powered by Google Gemini & Advanced AI Reasoning
        <br>
        <small>Features: Self-healing retrieval • Adaptive learning • Multi-document analysis • Citation tracking</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()