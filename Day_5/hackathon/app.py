import streamlit as st
import pandas as pd
import json
import datetime
from datetime import datetime, timedelta
import plotly.express as px
from typing import Dict, List, Any
import google.generativeai as genai
import re
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dataclasses import dataclass, asdict

# Configure Streamlit
st.set_page_config(
    page_title="🎯 Life Event-Aware Study Planner",
    page_icon="📚",
    layout="wide"
)

# Initialize session state
for key in ['study_data', 'events', 'chat_history', 'vector_store', 'qa_chain']:
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
    date: datetime
    end_date: datetime
    event_type: str
    impact_level: int
    description: str = ""

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
        
    def initialize_knowledge_base(self):
        """Initialize RAG with productivity and learning research"""
        productivity_docs = [
            "Spaced Repetition: Research shows reviewing material at increasing intervals (1 day, 3 days, 1 week, 2 weeks) improves long-term retention by 200-400%. Optimal for factual learning and vocabulary.",
            
            "Pomodoro Technique: 25-minute focused work sessions followed by 5-minute breaks. Studies show this reduces mental fatigue and maintains consistent performance. Particularly effective for high-concentration tasks.",
            
            "Interleaving Practice: Mixing different types of problems or topics in a single study session improves problem-solving skills by 60%. More effective than blocked practice for complex subjects.",
            
            "Active Recall: Testing yourself without looking at notes activates retrieval pathways, strengthening memory. 40% more effective than passive reading for long-term retention.",
            
            "Cognitive Load Theory: Human working memory can handle 7±2 items simultaneously. Break complex topics into smaller chunks and use visual aids to reduce cognitive overload.",
            
            "Peak Performance Timing: Most people have 2-3 hour windows of peak cognitive performance. Schedule difficult tasks during these periods, typically 2-4 hours after waking.",
            
            "Fatigue Management: Mental fatigue accumulates after 45-90 minutes of focused work. Signs include decreased accuracy, slower processing, and reduced motivation. Recovery requires 15-30 minute breaks.",
            
            "Stress-Performance Curve: Moderate stress enhances performance, but high stress impairs learning. During high-stress periods (exams, deadlines), reduce cognitive load and focus on review rather than new learning.",
            
            "Context-Dependent Learning: Studying in varied environments improves recall flexibility. Change locations, times, and study methods to strengthen memory pathways.",
            
            "Sleep Consolidation: Memory consolidation occurs primarily during sleep. Studying before sleep and getting 7-9 hours improves retention by 20-40%.",
            
            "Exercise and Cognition: 20-30 minutes of moderate exercise before studying increases BDNF (brain-derived neurotrophic factor), improving learning capacity for 2-3 hours.",
            
            "Time Blocking: Dedicated time blocks for specific subjects reduce task-switching costs. Minimum 90-minute blocks for deep work, 30-minute blocks for review.",
            
            "Attention Restoration Theory: Natural environments restore directed attention capacity. 5-10 minute nature breaks between study sessions improve subsequent focus.",
            
            "Flow State Conditions: Clear goals, immediate feedback, and balanced challenge-skill ratio create flow states. Flow sessions are 3-5x more productive than regular study.",
            
            "Metacognitive Strategies: Self-questioning and reflection improve learning efficiency. Ask 'What do I know?', 'What don't I understand?', 'How can I apply this?' during study."
        ]
        
        # Create documents
        documents = [Document(page_content=doc) for doc in productivity_docs]
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        splits = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        
        # Create QA chain
        template = """Use the following productivity research to answer the question about study planning:

Context: {context}

Question: {question}

Provide specific, actionable advice based on the research. Be concise and practical.

Answer:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt}
        )
        
        return self.qa_chain

class LifeEventStudyPlanner:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.rag_advisor = RAGStudyAdvisor(api_key)
        
    def extract_json(self, text: str) -> str:
        """Extract JSON from AI response"""
        try:
            json.loads(text)
            return text
        except:
            pattern = r'```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```'
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1)
            
            for char in ['{', '[']:
                start = text.find(char)
                if start != -1:
                    end = text.rfind('}' if char == '{' else ']')
                    if end > start:
                        try:
                            candidate = text[start:end+1]
                            json.loads(candidate)
                            return candidate
                        except:
                            continue
            return "[]"
    
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
            st.error(f"Error generating modules: {e}")
            return []
    
    def generate_life_events(self, profile: Dict) -> List[Dict]:
        """Generate realistic life events"""
        prompt = f"""
        Generate realistic life events for a {profile['level']} {profile['field']} student
        over {profile['duration_weeks']} weeks.
        
        Include academic, personal, and professional events in JSON:
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
        """
        
        try:
            response = self.model.generate_content(prompt)
            json_data = self.extract_json(response.text)
            return json.loads(json_data)
        except Exception as e:
            st.error(f"Error generating events: {e}")
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
        
        Generate 30-day schedule in JSON:
        [
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
            schedule_data = json.loads(json_data)
            
            schedule = []
            for session in schedule_data:
                schedule.append({
                    'date': datetime.strptime(session['date'], '%Y-%m-%d'),
                    'module': session['module'],
                    'duration_hours': session['duration_hours'],
                    'intensity': session['intensity'],
                    'focus_topics': session['focus_topics'],
                    'scheduling_reason': session['scheduling_reason'],
                    'conflict_avoidance': session['conflict_avoidance']
                })
            
            return sorted(schedule, key=lambda x: x['date'])
        except Exception as e:
            st.error(f"Error creating schedule: {e}")
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
            # Convert date strings to datetime
            df['date'] = pd.to_datetime(df['date'])
            df['end_date'] = pd.to_datetime(df['end_date'])
            return df.to_dict('records')
        elif uploaded_file.name.endswith('.json'):
            events = json.load(uploaded_file)
            # Convert date strings to datetime
            for event in events:
                event['date'] = datetime.strptime(event['date'], '%Y-%m-%d')
                event['end_date'] = datetime.strptime(event['end_date'], '%Y-%m-%d')
            return events
    except Exception as e:
        st.error(f"Error parsing file: {e}")
    return []
def main():
    st.title("🎯 Life Event-Aware Study Planner")
    st.markdown("### Adaptive AI Study Planning with RAG-Powered Insights")
    
    # API Key
    api_key = st.secrets.get("GEMINI_API_KEY") or st.sidebar.text_input("Gemini API Key", type="password")
    if not api_key:
        st.error("Please provide Gemini API key")
        st.stop()
    
    # Initialize planner
    @st.cache_resource
    def get_planner():
        planner = LifeEventStudyPlanner(api_key)
        planner.rag_advisor.initialize_knowledge_base()
        return planner
    
    planner = get_planner()
    
    # Sidebar Configuration
    st.sidebar.header("📚 Study Profile")
    field = st.sidebar.text_input("Field of Study", "Data Science")
    level = st.sidebar.selectbox("Level", ["Beginner", "Intermediate", "Advanced"])
    hours_per_week = st.sidebar.slider("Hours/Week", 5, 50, 20)
    duration_weeks = st.sidebar.slider("Duration (weeks)", 4, 20, 12)
    goals = st.sidebar.text_area("Goals", "Master key concepts and build projects")
    
    # Event Upload
    st.sidebar.header("📅 Life Events")
    uploaded_file = st.sidebar.file_uploader("Upload Events (CSV/JSON)", type=['csv', 'json'])
    
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
        'impact_level': [5, 3, 4]
    })
    
    st.sidebar.download_button(
        "📥 Download Sample CSV",
        sample_data.to_csv(index=False),
        "sample_events.csv",
        "text/csv"
    )
    
    # Generate Plan
    if st.sidebar.button("🚀 Generate Adaptive Plan", type="primary"):
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
            if uploaded_file:
                events_data = parse_uploaded_events(uploaded_file)
            else:
                events_data = planner.generate_life_events(profile)
            
            events = [LifeEvent(**event) for event in events_data]
            
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
    if st.session_state.study_data:
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
                st.dataframe(schedule_df, use_container_width=True)
                
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
        
        with tab2:
            st.header("⚡ Smart Alerts & Adjustments")
            
            data = st.session_state.study_data
            
            # Fatigue metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mental Load", f"{data['fatigue'].mental_load:.0%}")
            with col2:
                st.metric("Stress Level", f"{data['fatigue'].stress_level:.0%}")
            with col3:
                st.metric("Capacity", f"{data['fatigue'].weekly_capacity:.0%}")
            
            # Upcoming events impact
            st.subheader("📅 Upcoming Events Impact")
            for event in data['events'][:5]:
                days_until = (event.date - datetime.now()).days
                if days_until >= 0:
                    impact_color = ["🟢", "🟡", "🟠", "🔴", "🚨"][min(event.impact_level-1, 4)]
                    st.write(f"{impact_color} **{event.name}** - {days_until} days away (Impact: {event.impact_level}/5)")
            
            # Dynamic recommendations
            if data['fatigue'].stress_level > 0.7:
                st.error("🚨 High stress detected - Consider lighter study methods and more breaks")
            elif data['fatigue'].stress_level > 0.5:
                st.warning("⚠️ Moderate stress - Focus on review over new learning")
            else:
                st.success("✅ Good capacity for deep learning sessions")
        
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
            
            fig = px.scatter(
                modules_df,
                x='Hours',
                y='Priority',
                size='Difficulty',
                color='Module',
                title="Module Priority vs Time Investment"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Weekly workload
            if data['schedule']:
                schedule_df = pd.DataFrame(data['schedule'])
                schedule_df['week'] = schedule_df['date'].dt.isocalendar().week
                weekly_hours = schedule_df.groupby('week')['duration_hours'].sum().reset_index()
                
                fig = px.bar(weekly_hours, x='week', y='duration_hours', title="Weekly Study Hours")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.header("🤖 RAG-Powered Study Assistant")
            
            # Display chat history
            if st.session_state.chat_history:
                for msg in st.session_state.chat_history:
                    st.chat_message(msg['role']).write(msg['content'])
            
            # Chat input with RAG
            if question := st.chat_input("Ask about study strategies, scheduling, or productivity..."):
                st.chat_message("user").write(question)
                
                if st.session_state.chat_history is None:
                    st.session_state.chat_history = []
                
                st.session_state.chat_history.append({"role": "user", "content": question})
                
                with st.spinner("🧠 Consulting research database..."):
                    try:
                        # Get RAG response
                        answer = planner.rag_advisor.qa_chain.run(question)
                        
                        st.chat_message("assistant").write(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        error_msg = f"I apologize, there was an error: {e}"
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
            
            for q in quick_questions:
                if st.button(q, key=f"quick_{q[:20]}"):
                    with st.spinner("Searching research..."):
                        answer = planner.rag_advisor.qa_chain.run(q)
                        st.info(f"**Q:** {q}\n\n**A:** {answer}")
    
    else:
        # Welcome screen
        st.markdown("""
        ## 🎯 Welcome to Life Event-Aware Study Planner
        
        ### Features:
        - 📚 **AI-Generated Study Modules**: Tailored to your field and level
        - 📅 **Event-Aware Scheduling**: Automatically adjusts for life events
        - ⚡ **Smart Alerts**: Proactive notifications for schedule conflicts
        - 🧠 **RAG Assistant**: Research-backed study advice
        - 📊 **Fatigue Monitoring**: Adaptive workload management
        
        ### How It Works:
        1. **Configure** your study profile in the sidebar
        2. **Upload** your calendar events (optional) 
        3. **Generate** your adaptive study plan
        4. **Receive** smart alerts and adjustments
        5. **Chat** with RAG assistant for personalized advice
        
        **Get started by filling out your study profile and clicking "Generate Adaptive Plan"!**
        """)

if __name__ == "__main__":
    main()