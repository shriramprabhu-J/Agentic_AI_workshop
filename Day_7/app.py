import streamlit as st
import pandas as pd
import json
import datetime
from datetime import datetime, timedelta
import plotly.express as px
from typing import Dict, List, Any, Optional
import google.generativeai as genai
import re
import tempfile
import os
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dataclasses import dataclass, asdict, field
from PyPDF2 import PdfReader
import base64

# Configure Streamlit
st.set_page_config(
    page_title="🎯 Life Event-Aware Study Planner",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
for key in ['study_data', 'events', 'chat_history', 'vector_store', 'qa_chain', 'rag_initialized']:
    if key not in st.session_state:
        st.session_state[key] = None

# Data Models
@dataclass
class StudyModule:
    name: str
    priority: int
    estimated_hours: float
    difficulty: int
    topics: List[str]
    deadline: datetime
    status: str = "pending"

@dataclass
class LifeEvent:
    name: str
    date: str
    end_date: str
    event_type: str
    impact_level: int
    description: str = ""
    
    def __post_init__(self):
        # Convert date strings to datetime objects
        if not isinstance(self.date, datetime):
            self.date = datetime.strptime(self.date, '%Y-%m-%d')
        if not isinstance(self.end_date, datetime):
            if self.end_date:
                self.end_date = datetime.strptime(self.end_date, '%Y-%m-%d')
            else:
                self.end_date = self.date

@dataclass
class FatigueMetrics:
    mental_load: float
    stress_level: float
    recommended_intensity: float
    weekly_capacity: float

class RAGStudyAdvisor:
    def __init__(self, api_key: str):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        self.llm = GoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.7
        )
        self.vector_store = None
        self.qa_chain = None
        
    def initialize_knowledge_base(self, pdf_files: List):
        """Initialize RAG with uploaded PDF documents"""
        if not pdf_files:
            st.error("Please upload at least one PDF file for the knowledge base")
            return None
            
        documents = []
        
        for pdf_file in pdf_files:
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_file.read())
                tmp_path = tmp.name
            
            try:
                # Extract text from PDF
                text = ""
                reader = PdfReader(tmp_path)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                
                # Create document
                documents.append(Document(page_content=text))
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)
        
        if not documents:
            st.error("No valid documents extracted from PDFs")
            return None
            
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(documents)
        
        # Create vector store
        try:
            self.vector_store = FAISS.from_documents(splits, self.embeddings)
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            return None
        
        # Create QA chain
        template = """Use the following context to answer the question about study planning:

Context: {context}

Question: {question}

Provide specific, actionable advice based on the research. Be concise and practical.

Answer:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        try:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(search_kwargs={"k": 5}),
                chain_type_kwargs={"prompt": prompt}
            )
            return self.qa_chain
        except Exception as e:
            st.error(f"Error creating QA chain: {str(e)}")
            return None

class LifeEventStudyPlanner:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    def extract_json(self, text: str) -> str:
        """Robust JSON extraction from AI response"""
        # Clean response text
        text = text.strip().replace('\\"', '"').replace('\\n', '')
        
        # First try to parse the entire text as JSON
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            pass
            
        # Try to find JSON objects using multiple strategies
        patterns = [
            r'```json\s*({.*?})\s*```',  # Markdown code block
            r'```\s*({.*?})\s*```',       # Generic code block
            r'({.*})',                     # Simple curly braces
            r'(\[.*\])'                    # Array format
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]  # Handle capture groups
                try:
                    # Try to parse as JSON
                    parsed = json.loads(match)
                    return json.dumps(parsed)  # Return valid JSON string
                except json.JSONDecodeError:
                    # Attempt to fix common issues
                    try:
                        # Remove trailing commas
                        fixed = re.sub(r',\s*([}\]])', r'\1', match)
                        # Replace single quotes with double quotes
                        fixed = fixed.replace("'", '"')
                        # Fix unescaped quotes
                        fixed = re.sub(r'(?<!\\)"', r'\"', fixed)
                        parsed = json.loads(fixed)
                        return json.dumps(parsed)
                    except:
                        continue
        
        # Final fallback - return empty schedule
        st.warning("Could not extract valid JSON from response. Using empty schedule.")
        return '{"schedule": []}'

    def generate_study_modules(self, profile: Dict) -> List[StudyModule]:
        """Generate AI-powered study modules"""
        prompt = f"""
        Generate study modules for: {profile['field']} (Level: {profile['level']})
        Available: {profile['hours_per_week']} hours/week for {profile['duration_weeks']} weeks
        Goals: {profile['goals']}
        
        Create 3-5 modules in JSON format:
        [
            {{
                "name": "Module Name",
                "priority": 1-10,
                "estimated_hours": float,
                "difficulty": 1-5,
                "topics": ["topic1", "topic2", "topic3"],
                "deadline_days": integer
            }}
        ]
        
        Ensure realistic time allocation and progressive difficulty.
        Return ONLY the JSON array.
        """
        
        try:
            response = self.model.generate_content(prompt)
            json_data = self.extract_json(response.text)
            modules_data = json.loads(json_data)
            
            modules = []
            for data in modules_data:
                module = StudyModule(
                    name=data['name'],
                    priority=data['priority'],
                    estimated_hours=data['estimated_hours'],
                    difficulty=data['difficulty'],
                    topics=data['topics'],
                    deadline=datetime.now() + timedelta(days=data['deadline_days'])
                )
                modules.append(module)
            
            return modules
        except Exception as e:
            st.error(f"Error generating modules: {str(e)}")
            return []
    
    def generate_life_events(self, profile: Dict) -> List[Dict]:
        """Generate realistic life events"""
        prompt = f"""
        Generate realistic life events for a {profile['level']} {profile['field']} student
        over {profile['duration_weeks']} weeks.
        
        Include academic, personal, and professional events in JSON format:
        [
            {{
                "name": "Event Name",
                "date": "YYYY-MM-DD",
                "end_date": "YYYY-MM-DD",
                "event_type": "exam/internship/vacation/festival/work/family",
                "impact_level": 1-5,
                "description": "Brief description"
            }}
        ]
        
        Make events realistic and varied in impact.
        Start from: {datetime.now().strftime('%Y-%m-%d')}
        Return ONLY the JSON array.
        """
        
        try:
            response = self.model.generate_content(prompt)
            json_data = self.extract_json(response.text)
            return json.loads(json_data)
        except Exception as e:
            st.error(f"Error generating events: {str(e)}")
            return []
    
    def calculate_fatigue_metrics(self, events: List[LifeEvent], study_hours: float) -> FatigueMetrics:
        """AI-powered fatigue assessment"""
        high_impact_events = [e for e in events if e.impact_level >= 4]
        upcoming_stress = len([e for e in high_impact_events if e.date <= datetime.now() + timedelta(days=14)])
        
        base_load = min(study_hours / 40.0, 1.0)
        stress_multiplier = 1 + (upcoming_stress * 0.2)
        
        mental_load = min(base_load * stress_multiplier, 1.0)
        stress_level = min(mental_load + (upcoming_stress * 0.1), 1.0)
        
        if stress_level > 0.8:
            intensity = 0.3
            capacity = 0.5
        elif stress_level > 0.6:
            intensity = 0.6
            capacity = 0.7
        else:
            intensity = 0.8
            capacity = 1.0
        
        return FatigueMetrics(
            mental_load=mental_load,
            stress_level=stress_level,
            recommended_intensity=intensity,
            weekly_capacity=capacity
        )
    
    def create_adaptive_schedule(self, modules: List[StudyModule], events: List[LifeEvent], 
                                fatigue: FatigueMetrics) -> List[Dict]:
        """Create event-aware schedule"""
        prompt = f"""
        Create an adaptive study schedule considering:
        
        Modules: {[{'name': m.name, 'hours': m.estimated_hours, 'priority': m.priority} for m in modules]}
        
        Upcoming Events: {[{'name': e.name, 'date': e.date.strftime('%Y-%m-%d'), 'impact': e.impact_level} for e in events[:10]]}
        
        Fatigue Metrics:
        - Mental Load: {fatigue.mental_load:.2f}
        - Stress Level: {fatigue.stress_level:.2f}
        - Recommended Intensity: {fatigue.recommended_intensity:.2f}
        
        Generate 30-day schedule in JSON format (ONLY respond with valid JSON):
        {{
            "schedule": [
                {{
                    "date": "YYYY-MM-DD",
                    "module": "Module Name",
                    "duration_hours": float,
                    "intensity": float,
                    "focus_topics": "Specific topics for this session",
                    "scheduling_reason": "Why scheduled this way",
                    "conflict_avoidance": "How conflicts were avoided"
                }}
            ]
        }}
        
        Rules:
        - Avoid high-impact event dates
        - Reduce intensity during stress periods
        - Prioritize modules by deadline and importance
        - Use spaced repetition principles
        
        Start from: {datetime.now().strftime('%Y-%m-%d')}
        """
        
        try:
            response = self.model.generate_content(prompt)
            json_data = self.extract_json(response.text)
            
            # Parse JSON response
            parsed = json.loads(json_data)
            
            # Handle different response formats
            if 'schedule' in parsed:
                schedule_data = parsed['schedule']
            elif isinstance(parsed, list):
                schedule_data = parsed
            else:
                raise ValueError("Unexpected schedule format")
            
            schedule = []
            for session in schedule_data:
                try:
                    schedule.append({
                        'date': datetime.strptime(session['date'], '%Y-%m-%d'),
                        'module': session['module'],
                        'duration_hours': float(session['duration_hours']),
                        'intensity': float(session['intensity']),
                        'focus_topics': session['focus_topics'],
                        'scheduling_reason': session['scheduling_reason'],
                        'conflict_avoidance': session['conflict_avoidance']
                    })
                except (KeyError, ValueError) as e:
                    st.error(f"Error parsing session: {str(e)}")
                    continue
            
            return sorted(schedule, key=lambda x: x['date'])
        except Exception as e:
            st.error(f"Error creating schedule: {str(e)}")
            if 'response' in locals():
                st.text(f"Response content: {response.text[:500]}...")
            return []
    
    def generate_adaptive_alerts(self, events: List[LifeEvent], schedule: List[Dict]) -> List[str]:
        """Generate context-aware study alerts"""
        alerts = []
        upcoming_events = [e for e in events if 0 <= (e.date - datetime.now()).days <= 7]
        
        for event in upcoming_events:
            if event.impact_level >= 4:
                if event.event_type in ['exam', 'test', 'assignment']:
                    alerts.append(f"🚨 {event.name} in {(event.date - datetime.now()).days} days - Switch to review mode, avoid new topics")
                elif event.event_type in ['vacation', 'internship']:
                    alerts.append(f"⚠️ {event.name} approaching - Complete current modules early, adjust schedule")
                elif event.event_type in ['family', 'festival']:
                    alerts.append(f"📅 {event.name} this week - Reduce study intensity, focus on light review")
        
        return alerts

def parse_uploaded_events(uploaded_file) -> List[Dict]:
    """Parse uploaded CSV/JSON events"""
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
            # Ensure required columns exist
            required_cols = ['name', 'date', 'event_type', 'impact_level']
            for col in required_cols:
                if col not in df.columns:
                    st.error(f"Missing required column: {col}")
                    return []
                    
            # Convert dates to string format if they're Timestamps
            df['date'] = df['date'].astype(str)
            if 'end_date' in df.columns:
                df['end_date'] = df['end_date'].astype(str)
            else:
                df['end_date'] = df['date']
                
            # Fill missing description
            if 'description' not in df.columns:
                df['description'] = ""
                
            return df.to_dict('records')
            
        elif uploaded_file.name.endswith('.json'):
            events = json.load(uploaded_file)
            # Ensure proper format
            for event in events:
                if 'end_date' not in event:
                    event['end_date'] = event['date']
                if 'description' not in event:
                    event['description'] = ""
            return events
            
    except Exception as e:
        st.error(f"Error parsing file: {str(e)}")
    return []

def main():
    st.title("🎯 Life Event-Aware Study Planner")
    st.markdown("### Adaptive AI Study Planning with RAG-Powered Insights")
    
    # API Key
    api_key = st.secrets.get("GEMINI_API_KEY", None) or st.sidebar.text_input("Gemini API Key", type="password")
    if not api_key:
        st.warning("Please provide Gemini API key to continue")
        st.stop()
    
    # Initialize planner
    planner = LifeEventStudyPlanner(api_key)
    
    # Sidebar Configuration
    st.sidebar.header("📚 Study Profile")
    field = st.sidebar.text_input("Field of Study", "Data Science")
    level = st.sidebar.selectbox("Level", ["Beginner", "Intermediate", "Advanced"])
    hours_per_week = st.sidebar.slider("Hours/Week", 5, 50, 20)
    duration_weeks = st.sidebar.slider("Duration (weeks)", 4, 20, 12)
    goals = st.sidebar.text_area("Goals", "Master key concepts and build projects")
    
    # Knowledge Base Upload
    st.sidebar.header("🧠 Knowledge Base")
    uploaded_kb = st.sidebar.file_uploader(
        "Upload PDF Knowledge Base", 
        type=['pdf'], 
        accept_multiple_files=True,
        help="Upload PDFs with study techniques and productivity research"
    )
    
    # Event Upload
    st.sidebar.header("📅 Life Events")
    uploaded_events = st.sidebar.file_uploader(
        "Upload Events (CSV/JSON)", 
        type=['csv', 'json'], 
        accept_multiple_files=False
    )
    
    # Sample CSV download
    sample_data = pd.DataFrame({
        'name': ['Midterm Exams', 'Diwali Festival', 'Internship Interview'],
        'date': [(datetime.now() + timedelta(days=14)).strftime('%Y-%m-%d'),
                (datetime.now() + timedelta(days=21)).strftime('%Y-%m-%d'),
                (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')],
        'end_date': [(datetime.now() + timedelta(days=16)).strftime('%Y-%m-%d'),
                    (datetime.now() + timedelta(days=21)).strftime('%Y-%m-%d'),
                    (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')],
        'event_type': ['exam', 'festival', 'work'],
        'impact_level': [5, 3, 4],
        'description': ['Important semester exams', 'Family celebration time', 'Final round interview']
    })
    
    st.sidebar.download_button(
        "📥 Download Sample CSV",
        sample_data.to_csv(index=False),
        "sample_events.csv",
        "text/csv"
    )
    
    # Initialize RAG if not already done
    if uploaded_kb and st.session_state.rag_initialized is None:
        with st.spinner("🧠 Initializing RAG knowledge base..."):
            rag_advisor = RAGStudyAdvisor(api_key)
            qa_chain = rag_advisor.initialize_knowledge_base(uploaded_kb)
            if qa_chain:
                st.session_state.rag_initialized = True
                st.session_state.rag_advisor = rag_advisor
                st.session_state.qa_chain = qa_chain
                st.success("✅ RAG knowledge base initialized!")
            else:
                st.error("❌ Failed to initialize knowledge base")
    
    # Generate Plan
    if st.sidebar.button("🚀 Generate Adaptive Plan", type="primary", use_container_width=True):
        # Check RAG initialization
        if not st.session_state.get('rag_initialized'):
            st.error("Please upload knowledge base PDFs first")
            return
            
        profile = {
            'field': field,
            'level': level,
            'hours_per_week': hours_per_week,
            'duration_weeks': duration_weeks,
            'goals': goals
        }
        
        with st.spinner("🤖 AI generating adaptive study plan..."):
            # Generate modules
            modules = planner.generate_study_modules(profile)
            
            # Get events
            events_data = []
            if uploaded_events:
                events_data = parse_uploaded_events(uploaded_events)
            else:
                events_data = planner.generate_life_events(profile)
            
            events = []
            for event in events_data:
                try:
                    events.append(LifeEvent(**event))
                except Exception as e:
                    st.error(f"Error creating event: {str(e)}")
            
            # Calculate fatigue
            fatigue = planner.calculate_fatigue_metrics(events, hours_per_week)
            
            # Create schedule
            schedule = planner.create_adaptive_schedule(modules, events, fatigue)
            
            # Generate alerts
            alerts = planner.generate_adaptive_alerts(events, schedule)
            
            st.session_state.study_data = {
                'modules': modules,
                'events': events,
                'schedule': schedule,
                'fatigue': fatigue,
                'alerts': alerts
            }
            
            st.success("✅ Adaptive study plan generated!")
    
    # Main Interface
    if st.session_state.get('study_data'):
        tab1, tab2, tab3, tab4 = st.tabs(["📅 Schedule", "⚡ Alerts", "📊 Analytics", "🤖 RAG Assistant"])
        
        with tab1:
            st.header("📅 Event-Aware Schedule")
            
            data = st.session_state.study_data
            
            # Current alerts
            if data['alerts']:
                st.subheader("🚨 Current Alerts")
                for alert in data['alerts']:
                    st.warning(alert)
            
            # Schedule table
            if data['schedule']:
                schedule_df = pd.DataFrame([
                    {
                        'Date': s['date'].strftime('%Y-%m-%d'),
                        'Module': s['module'],
                        'Hours': s['duration_hours'],
                        'Intensity': f"{s['intensity']:.0%}",
                        'Topics': s['focus_topics'],
                        'Reason': s['scheduling_reason']
                    }
                    for s in data['schedule'][:14]
                ])
                st.dataframe(schedule_df, use_container_width=True, hide_index=True)
                
                # Timeline chart
                fig = px.timeline(
                    schedule_df.head(10),
                    x_start="Date",
                    x_end="Date", 
                    y="Module",
                    color="Hours",
                    title="Adaptive Study Timeline"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No schedule generated")
        
        with tab2:
            st.header("⚡ Smart Alerts & Adjustments")
            
            data = st.session_state.study_data
            
            # Fatigue metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mental Load", f"{data['fatigue'].mental_load:.0%}", help="Current cognitive workload")
            with col2:
                st.metric("Stress Level", f"{data['fatigue'].stress_level:.0%}", help="Current stress indicators")
            with col3:
                st.metric("Capacity", f"{data['fatigue'].weekly_capacity:.0%}", help="Available weekly study capacity")
            
            # Upcoming events impact
            st.subheader("📅 Upcoming Events Impact")
            for event in data['events'][:5]:
                days_until = (event.date - datetime.now()).days
                if days_until >= 0:
                    impact_color = ["🟢", "🟡", "🟠", "🔴", "🚨"][min(event.impact_level-1, 4)]
                    st.write(f"{impact_color} **{event.name}** - {days_until} days away (Impact: {event.impact_level}/5)")
            
            # Dynamic recommendations
            st.subheader("📝 Adaptive Recommendations")
            if data['fatigue'].stress_level > 0.7:
                st.error("🚨 High stress detected - Consider lighter study methods and more breaks")
                st.markdown("- Focus on review rather than new material")
                st.markdown("- Use Pomodoro technique with shorter sessions")
                st.markdown("- Prioritize sleep and recovery activities")
            elif data['fatigue'].stress_level > 0.5:
                st.warning("⚠️ Moderate stress - Focus on review over new learning")
                st.markdown("- Use active recall techniques")
                st.markdown("- Take frequent short breaks")
                st.markdown("- Consider light exercise between sessions")
            else:
                st.success("✅ Good capacity for deep learning sessions")
                st.markdown("- Tackle difficult topics during peak hours")
                st.markdown("- Try interleaving different subjects")
                st.markdown("- Experiment with new learning techniques")
        
        with tab3:
            st.header("📊 Learning Analytics")
            
            data = st.session_state.study_data
            
            # Module priorities
            modules_df = pd.DataFrame([
                {
                    'Module': m.name,
                    'Priority': m.priority,
                    'Hours': m.estimated_hours,
                    'Difficulty': m.difficulty,
                    'Deadline': m.deadline.strftime('%Y-%m-%d')
                }
                for m in data['modules']
            ])
            
            if not modules_df.empty:
                fig = px.scatter(
                    modules_df,
                    x='Hours',
                    y='Priority',
                    size='Difficulty',
                    color='Module',
                    title="Module Priority vs Time Investment",
                    hover_data=['Deadline']
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Weekly workload
                if data['schedule']:
                    schedule_df = pd.DataFrame(data['schedule'])
                    schedule_df['date'] = pd.to_datetime(schedule_df['date'])
                    schedule_df['week'] = schedule_df['date'].dt.isocalendar().week
                    weekly_hours = schedule_df.groupby('week')['duration_hours'].sum().reset_index()
                    
                    fig = px.bar(weekly_hours, x='week', y='duration_hours', 
                                title="Weekly Study Hours", labels={'duration_hours': 'Hours'})
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No modules data available")
        
        with tab4:
            st.header("🤖 RAG-Powered Study Assistant")
            
            # Initialize chat history
            if st.session_state.chat_history is None:
                st.session_state.chat_history = []
            
            # Display chat history
            for msg in st.session_state.chat_history:
                if msg['role'] == 'user':
                    st.chat_message("user").write(msg['content'])
                else:
                    st.chat_message("assistant").write(msg['content'])
            
            # Chat input with RAG
            if question := st.chat_input("Ask about study strategies, scheduling, or productivity..."):
                st.chat_message("user").write(question)
                st.session_state.chat_history.append({"role": "user", "content": question})
                
                with st.spinner("🧠 Consulting research database..."):
                    try:
                        # Get RAG response
                        answer = st.session_state.qa_chain.run(question)
                        
                        st.chat_message("assistant").write(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        error_msg = f"I apologize, there was an error: {str(e)}"
                        st.chat_message("assistant").write(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
            
            # Quick questions
            st.subheader("💡 Quick Questions")
            quick_questions = [
                "How should I adjust my schedule during exam week?",
                "What study techniques work best under stress?",
                "How to maintain focus during busy periods?",
                "Best practices for spaced repetition?"
            ]
            
            cols = st.columns(2)
            for i, q in enumerate(quick_questions):
                with cols[i % 2]:
                    if st.button(q, key=f"quick_{q[:20]}", use_container_width=True):
                        with st.spinner("Searching research..."):
                            try:
                                answer = st.session_state.qa_chain.run(q)
                                st.info(f"**Q:** {q}\n\n**A:** {answer}")
                            except:
                                st.error("Error retrieving answer")
    
    else:
        # Welcome screen
        st.markdown("""
        ## 🎯 Welcome to Life Event-Aware Study Planner
        
        ### Features:
        - 📚 **AI-Generated Study Modules**: Tailored to your field and level
        - 📅 **Event-Aware Scheduling**: Automatically adjusts for life events
        - ⚡ **Smart Alerts**: Proactive notifications for schedule conflicts
        - 🧠 **RAG Assistant**: Research-backed study advice (PDF-based)
        - 📊 **Fatigue Monitoring**: Adaptive workload management
        
        ### How It Works:
        1. **Upload PDFs** with study techniques/research
        2. **Configure** your study profile in the sidebar
        3. **Upload** your calendar events (optional) 
        4. **Generate** your adaptive study plan
        5. **Receive** smart alerts and adjustments
        6. **Chat** with RAG assistant for personalized advice
        
        **Get started by uploading knowledge base PDFs and clicking 'Generate Adaptive Plan'!**
        """)

if __name__ == "__main__":
    main()