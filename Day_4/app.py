import streamlit as st
import google.generativeai as genai
import PyPDF2
from datetime import datetime, timedelta
import os
import time
import json
import re
from typing import List, Dict, Any

# Configuration
PDF_PATH = r"E:\Agentic_AI_workshop\Day_4\insurance_policies.pdf"
GEMINI_API_KEY = "AIzaSyADul5IZjW7U9XR26VLYeItcn0vaUvjx9Q"

# Configure Gemini
@st.cache_resource
def configure_gemini(api_key: str) -> genai.GenerativeModel:
    """Configure Gemini API with caching"""
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-2.0-flash-exp')
    except Exception as e:
        st.error(f"Failed to configure Gemini: {e}")
        return None

# Extract text from PDF with better error handling
@st.cache_data
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF with improved error handling"""
    if not os.path.exists(pdf_path):
        st.error(f"PDF file not found at: {pdf_path}")
        return ""
    
    try:
        text_content = ""
        with open(pdf_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            total_pages = len(pdf_reader.pages)
            
            # Progress bar for PDF extraction
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text_content += page_text + "\n"
                
                # Update progress
                progress = (i + 1) / total_pages
                progress_bar.progress(progress)
                status_text.text(f"Processing page {i + 1} of {total_pages}")
            
            progress_bar.empty()
            status_text.empty()
            
        return text_content
    except Exception as e:
        st.error(f"Error extracting PDF: {e}")
        return ""

# Intelligent PDF parsing using Gemini
def parse_pdf_with_gemini(pdf_text: str, model: genai.GenerativeModel) -> List[Dict[str, Any]]:
    """Parse PDF content using Gemini to extract policy information"""
    if not pdf_text or not model:
        return []
    
    parsing_prompt = f"""
    Analyze the following insurance policy document and extract structured information about each policy.
    
    For each policy found, extract:
    - Policy Name
    - Coverage Types (list)
    - Age Range (min and max)
    - Policy Type (individual/family/group)
    - Special Features (list)
    - Price Range/Tier
    - Eligibility Criteria
    - Benefits
    - Exclusions (if any)
    
    Return the information in JSON format as an array of policy objects.
    
    Document Content:
    {pdf_text[:8000]}  # Limit to avoid token limits
    
    Please provide a valid JSON response with the extracted policy information.
    """
    
    try:
        response = model.generate_content(parsing_prompt)
        # Try to extract JSON from response
        json_match = re.search(r'\[.*\]', response.text, re.DOTALL)
        if json_match:
            policies_data = json.loads(json_match.group())
            return policies_data
        else:
            st.warning("Could not parse policies from PDF. Using fallback method.")
            return fallback_parse_policies(pdf_text)
    except Exception as e:
        st.warning(f"Gemini parsing failed: {e}. Using fallback method.")
        return fallback_parse_policies(pdf_text)

# Fallback parsing method
def fallback_parse_policies(text: str) -> List[Dict[str, Any]]:
    """Fallback method to parse policies from text"""
    policies = []
    
    # Split text into potential policy sections
    sections = re.split(r'\n\s*\n', text)
    
    for section in sections:
        if len(section.strip()) < 50:  # Skip very short sections
            continue
            
        policy = {}
        lines = section.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Extract policy name
            if any(keyword in line.lower() for keyword in ['policy', 'plan', 'coverage']):
                if 'name' not in policy:
                    policy['name'] = line
            
            # Extract coverage information
            if any(keyword in line.lower() for keyword in ['covers', 'coverage', 'benefits']):
                if 'coverage' not in policy:
                    policy['coverage'] = [line]
            
            # Extract age information
            age_match = re.search(r'(\d+)\s*[-to]+\s*(\d+)', line)
            if age_match:
                policy['min_age'] = int(age_match.group(1))
                policy['max_age'] = int(age_match.group(2))
            
            # Extract price information
            if any(keyword in line.lower() for keyword in ['price', 'cost', 'premium']):
                policy['price_info'] = line
        
        if policy and 'name' in policy:
            policies.append(policy)
    
    return policies

# Rate-limited Gemini query with better error handling
def safe_gemini_query(model: genai.GenerativeModel, prompt: str, retries: int = 3) -> str:
    """Make a safe Gemini query with rate limiting and error handling"""
    if not model:
        return "Gemini model not available"
    
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            error_str = str(e).lower()
            if "quota" in error_str or "429" in error_str:
                wait_time = (attempt + 1) * 30
                st.warning(f"Rate limit reached. Waiting {wait_time} seconds... (Attempt {attempt + 1}/{retries})")
                time.sleep(wait_time)
                continue
            elif "400" in error_str:
                st.error("Invalid request. Please check your input.")
                return ""
            else:
                st.error(f"API Error: {e}")
                if attempt == retries - 1:
                    return ""
                time.sleep(5)  # Brief pause before retry
    
    return "Unable to process request after multiple attempts"

def main():
    # Page configuration
    st.set_page_config(
        page_title="Healthcare Policy Sales Agent",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .policy-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Healthcare Policy Sales Agent</h1>
        <p>Find the perfect insurance policy tailored to your needs</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'policies' not in st.session_state:
        st.session_state.policies = []
        st.session_state.pdf_text = ""
        st.session_state.last_query_time = datetime.min
        st.session_state.query_count = 0
        st.session_state.history = []
    
    # Sidebar for system status and controls
    with st.sidebar:
        st.header("📊 System Status")
        
        # API Key status
        if GEMINI_API_KEY:
            st.success("✅ API Key Configured")
        else:
            st.error("❌ API Key Missing")
            st.info("Please add your Gemini API key to the configuration")
        
        # PDF status
        pdf_exists = os.path.exists(PDF_PATH)
        if pdf_exists:
            st.success("✅ PDF File Found")
        else:
            st.error("❌ PDF File Not Found")
            st.info(f"Please ensure your PDF is at: {PDF_PATH}")
        
        # Model status
        if 'model' not in st.session_state and GEMINI_API_KEY:
            with st.spinner("Initializing Gemini..."):
                st.session_state.model = configure_gemini(GEMINI_API_KEY)
        
        if 'model' in st.session_state and st.session_state.model:
            st.success("✅ Gemini Model Ready")
        else:
            st.error("❌ Gemini Model Not Ready")
        
        # Load policies button
        if pdf_exists and st.button("🔄 Reload Policy Data"):
            with st.spinner("Extracting and parsing PDF..."):
                st.session_state.pdf_text = extract_text_from_pdf(PDF_PATH)
                if st.session_state.pdf_text and 'model' in st.session_state:
                    st.session_state.policies = parse_pdf_with_gemini(
                        st.session_state.pdf_text, 
                        st.session_state.model
                    )
                    st.success(f"Loaded {len(st.session_state.policies)} policies")
                else:
                    st.error("Failed to load policies")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Load policies if not already loaded
        if not st.session_state.policies and pdf_exists:
            if st.button("📄 Load Policy Data", type="primary"):
                with st.spinner("Loading and parsing policy data..."):
                    st.session_state.pdf_text = extract_text_from_pdf(PDF_PATH)
                    if st.session_state.pdf_text and 'model' in st.session_state:
                        st.session_state.policies = parse_pdf_with_gemini(
                            st.session_state.pdf_text, 
                            st.session_state.model
                        )
                        st.success(f"Successfully loaded {len(st.session_state.policies)} policies!")
                    else:
                        st.error("Failed to load policies. Please check your PDF and API configuration.")
        
        # User input form
        if st.session_state.policies:
            st.header("👤 Your Information")
            
            with st.form("customer_form", clear_on_submit=False):
                col_a, col_b = st.columns(2)
                
                with col_a:
                    name = st.text_input("Full Name", value="John Doe")
                    age = st.number_input("Age", min_value=0, max_value=120, value=30)
                    family_type = st.radio("Family Type", ["Individual", "Family"], horizontal=True)
                
                with col_b:
                    dependents = st.number_input(
                        "Number of Dependents", 
                        min_value=0, 
                        max_value=10, 
                        value=0,
                        disabled=(family_type == "Individual")
                    )
                    budget = st.select_slider(
                        "Budget Range", 
                        options=["Low", "Medium", "High", "Premium"],
                        value="Medium"
                    )
                
                st.subheader("Special Requirements")
                requirements = st.multiselect(
                    "Select your healthcare needs:",
                    options=[
                        "Senior Health", "Dental Care", "Vision Care", 
                        "Maternity Care", "Chronic Conditions", "Mental Health", 
                        "Prescription Drugs", "Preventive Care", "Emergency Care"
                    ],
                    default=[]
                )
                
                additional_info = st.text_area(
                    "Additional Information", 
                    placeholder="Any specific health concerns or requirements..."
                )
                
                submit_button = st.form_submit_button("🔍 Get Personalized Recommendations", type="primary")
                
                if submit_button:
                    if not ('model' in st.session_state and st.session_state.model):
                        st.error("Gemini model not available. Please check your API configuration.")
                    else:
                        # Rate limiting check
                        now = datetime.now()
                        if (now - st.session_state.last_query_time).total_seconds() < 60:
                            if st.session_state.query_count >= 2:
                                st.warning("⏳ Rate limit reached (2 requests per minute). Please wait...")
                                time_left = 60 - int((now - st.session_state.last_query_time).total_seconds())
                                st.info(f"Try again in {time_left} seconds")
                            else:
                                st.session_state.query_count += 1
                        else:
                            st.session_state.query_count = 1
                            st.session_state.last_query_time = now
                        
                        if st.session_state.query_count <= 2:
                            # Generate recommendations
                            recommendation_prompt = f"""
                            As a healthcare insurance expert, provide personalized policy recommendations based on:
                            
                            Customer Profile:
                            - Name: {name}
                            - Age: {age}
                            - Family Type: {family_type}
                            - Dependents: {dependents}
                            - Budget: {budget}
                            - Special Needs: {', '.join(requirements) if requirements else 'None'}
                            - Additional Info: {additional_info if additional_info else 'None'}
                            
                            Available Policies:
                            {json.dumps(st.session_state.policies, indent=2)}
                            
                            Please provide:
                            1. A brief analysis of the customer's needs
                            2. Top 3 most suitable policies with detailed explanations
                            3. Comparison of benefits and drawbacks
                            4. Recommendations for additional coverage if needed
                            5. Next steps for the customer
                            
                            Format your response with clear headings and bullet points for easy reading.
                            """
                            
                            with st.spinner("🤖 Analyzing your needs and generating recommendations..."):
                                response = safe_gemini_query(st.session_state.model, recommendation_prompt)
                                
                                if response and response != "Unable to process request after multiple attempts":
                                    st.success("✅ Recommendations Generated!")
                                    st.markdown("---")
                                    st.markdown(response)
                                    
                                    # Save to history
                                    st.session_state.history.append({
                                        'timestamp': datetime.now(),
                                        'customer': {
                                            'name': name,
                                            'age': age,
                                            'family_type': family_type,
                                            'dependents': dependents,
                                            'budget': budget,
                                            'requirements': requirements
                                        },
                                        'response': response
                                    })
                                else:
                                    st.error("Failed to generate recommendations. Please try again later.")
    
    with col2:
        # Policy summary
        st.header("📋 Policy Summary")
        
        if st.session_state.policies:
            st.metric("Total Policies Available", len(st.session_state.policies))
            
            # Display policy overview
            with st.expander("View All Policies", expanded=False):
                for i, policy in enumerate(st.session_state.policies, 1):
                    st.markdown(f"""
                    <div class="policy-card">
                        <h4>Policy {i}: {policy.get('name', 'Unnamed Policy')}</h4>
                        <p><strong>Coverage:</strong> {', '.join(policy.get('coverage', ['Not specified']))}</p>
                        <p><strong>Age Range:</strong> {policy.get('min_age', 'N/A')} - {policy.get('max_age', 'N/A')}</p>
                        <p><strong>Price:</strong> {policy.get('price_info', 'Contact for pricing')}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No policies loaded yet. Please load your PDF data first.")
        
        # Query history
        if st.session_state.history:
            st.header("📝 Recent Consultations")
            
            for i, entry in enumerate(reversed(st.session_state.history[-5:]), 1):
                with st.expander(f"Consultation {i} - {entry['customer']['name']}"):
                    st.write(f"**Date:** {entry['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                    st.write(f"**Age:** {entry['customer']['age']}")
                    st.write(f"**Family:** {entry['customer']['family_type']}")
                    st.write(f"**Budget:** {entry['customer']['budget']}")
                    if st.button(f"View Full Recommendation {i}", key=f"view_{i}"):
                        st.markdown(entry['response'])

if __name__ == "__main__":
    main()