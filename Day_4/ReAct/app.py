import streamlit as st
import google.generativeai as genai
from tavily import TavilyClient
import time

# Configure APIs
genai.configure(api_key="AIzaSyADul5IZjW7U9XR26VLYeItcn0vaUvjx9Q")  # Gemini API Key
tavily_api_key = "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"  # Get yours at https://app.tavily.com/

# Initialize clients
try:
    tavily = TavilyClient(api_key=tavily_api_key)
except:
    st.error("Invalid Tavily API key. Please sign up at https://app.tavily.com/ and replace the key in the code.")
    st.stop()

# Initialize session state
if 'research_data' not in st.session_state:
    st.session_state.research_data = {
        'questions': [],
        'search_results': {},
        'report': ""
    }

# Generate research questions using Gemini
def generate_research_questions(topic):
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    As a research assistant, generate 5-6 well-structured research questions about: {topic}
    The questions should cover different aspects of the topic and be suitable for web research.
    Output ONLY a numbered list of questions without any additional text.
    """
    response = model.generate_content(prompt)
    return [q.split('. ', 1)[1] for q in response.text.split('\n') if q.strip() and q[0].isdigit()]

# Perform web search using Tavily
def search_web(question, max_results=3):
    search_result = tavily.search(query=question, max_results=max_results)
    return [
        {
            'title': result['title'],
            'content': result['content'],
            'url': result['url']
        }
        for result in search_result['results']
    ]

# Generate research report using Gemini
def generate_research_report(topic, questions, search_results):
    model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Format research data for the prompt
    research_context = f"# Research Topic: {topic}\n\n"
    for i, question in enumerate(questions):
        research_context += f"## Question {i+1}: {question}\n"
        for j, result in enumerate(search_results[question]):
            research_context += (
                f"### Source {j+1}: {result['title']}\n"
                f"{result['content']}\n\n"
            )
    
    prompt = f"""
    You are a research assistant. Compile a structured report using the following research data:
    
    {research_context}
    
    Report Structure:
    1. Title (centered, bold, large font)
    2. Introduction: Overview of the topic and research purpose
    3. Research Findings: One section per research question with:
       - Concise summary of findings
       - Integration of information from multiple sources
    4. Conclusion: Key insights and synthesis of information
    
    Use markdown formatting with appropriate headings and sections.
    """
    response = model.generate_content(prompt)
    return response.text

# Streamlit UI
st.title("🧠 ReAct Research Agent")
st.caption("Building AI Agents from Scratch - Gemini + Tavily Integration")

topic = st.text_input("Enter your research topic:", "Climate change impacts on biodiversity")

if st.button("Start Research"):
    with st.status("🚀 Starting research process...", expanded=True) as status:
        # Step 1: Generate questions
        st.write("🔍 Planning research questions...")
        st.session_state.research_data['questions'] = generate_research_questions(topic)
        st.success(f"Generated {len(st.session_state.research_data['questions'])} research questions")
        time.sleep(1)
        
        # Step 2: Web search
        st.write("🌐 Searching the web for answers...")
        st.session_state.research_data['search_results'] = {}
        for question in st.session_state.research_data['questions']:
            st.session_state.research_data['search_results'][question] = search_web(question)
            st.info(f"Found {len(st.session_state.research_data['search_results'][question])} sources for: {question}")
            time.sleep(0.5)
        
        # Step 3: Generate report
        st.write("📝 Compiling research report...")
        st.session_state.research_data['report'] = generate_research_report(
            topic,
            st.session_state.research_data['questions'],
            st.session_state.research_data['search_results']
        )
        status.update(label="✅ Research complete!", state="complete", expanded=False)

# Show research questions if available
if st.session_state.research_data['questions']:
    st.subheader("Research Questions")
    for i, question in enumerate(st.session_state.research_data['questions']):
        st.markdown(f"{i+1}. {question}")

# Show search results if available
if st.session_state.research_data['search_results']:
    st.subheader("Web Search Results")
    for question, results in st.session_state.research_data['search_results'].items():
        with st.expander(f"Sources for: {question}"):
            for i, result in enumerate(results):
                st.markdown(f"**Source {i+1}:** [{result['title']}]({result['url']})")
                st.caption(result['content'])
                st.divider()

# Display final report
if st.session_state.research_data['report']:
    st.subheader("Research Report")
    st.markdown(st.session_state.research_data['report'], unsafe_allow_html=True)