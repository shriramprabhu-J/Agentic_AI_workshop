import streamlit as st
import pandas as pd
import json
import datetime
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Tuple
import numpy as np
from dataclasses import dataclass, asdict
import google.generativeai as genai
import re
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="🧠 NeuroPlanner | AI Study Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main styling */
    .stApp {
        background-color: #f8f9fa;
    }
    .main-title {
        text-align: center;
        color: #2c3e50;
        padding: 20px 0;
        border-bottom: 1px solid #e0e0e0;
        margin-bottom: 30px;
    }
    .card {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        padding: 20px;
        margin-bottom: 20px;
        border-left: 4px solid #6366f1;
    }
    .card-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
    }
    .card-header i {
        margin-right: 10px;
        font-size: 1.4rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f0f4ff 0%, #e6f7ff 100%);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        border: none;
    }
    .metric-title {
        font-size: 0.9rem;
        color: #4b5563;
        margin-bottom: 8px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1e293b;
    }
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        padding: 10px 15px;
    }
    .progress-bar {
        height: 8px;
        background-color: #e2e8f0;
        border-radius: 4px;
        margin: 15px 0;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #6366f1 0%, #4f46e5 100%);
        border-radius: 4px;
    }
    .session-card {
        background-color: #f9fafb;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        border-left: 3px solid #6366f1;
    }
    .chat-message-user {
        background-color: #e0f2fe;
        border-radius: 15px 15px 0 15px;
        padding: 12px 15px;
        margin: 5px 0;
        max-width: 80%;
        margin-left: auto;
    }
    .chat-message-assistant {
        background-color: white;
        border: 1px solid #e2e8f0;
        border-radius: 15px 15px 15px 0;
        padding: 12px 15px;
        margin: 5px 0;
        max-width: 80%;
    }
    .sidebar .sidebar-content {
        background-color: #f8fafc;
    }
    .sidebar-header {
        padding: 20px 15px 10px;
        border-bottom: 1px solid #e2e8f0;
        margin-bottom: 15px;
    }
    .tab-container {
        background-color: white;
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-top: 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 12px 20px;
        border-radius: 8px 8px 0 0 !important;
        background-color: #f1f5f9 !important;
        margin: 0 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        font-weight: 600;
        box-shadow: 0 -2px 0 #4f46e5 inset;
    }
    .st-emotion-cache-1qg05tj {
        padding: 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agents_initialized' not in st.session_state:
    st.session_state.agents_initialized = False
if 'study_plan' not in st.session_state:
    st.session_state.study_plan = None
if 'events' not in st.session_state:
    st.session_state.events = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'modules' not in st.session_state:
    st.session_state.modules = []

# Data Models
@dataclass
class StudyModule:
    name: str
    priority: int  # 1-10
    estimated_hours: float
    difficulty: int  # 1-5
    dependencies: List[str]
    topics: List[str]
    deadline: datetime
    status: str = "pending"  # pending, in_progress, completed
    
@dataclass
class LifeEvent:
    name: str
    date: datetime
    end_date: datetime
    event_type: str  # exam, vacation, internship, sick_leave, festival
    impact_level: int  # 1-5 (how much it affects study capacity)
    
@dataclass
class FatigueMetrics:
    mental_load: float  # 0-1
    weekly_hours: float
    missed_sessions: int
    stress_level: float  # 0-1
    recommended_intensity: float  # 0-1

# Helper function to extract JSON from Gemini response
def extract_json(text: str) -> str:
    """Extract JSON content from markdown code blocks or raw text with improved robustness."""
    # First try to find complete JSON object/array
    try:
        # Try parsing the entire text first (might be pure JSON)
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    
    # Pattern to match ```json ... ``` blocks
    json_block_match = re.search(r'```(?:json)?\s*({.*}|\[.*\])\s*```', text, re.DOTALL)
    if json_block_match:
        extracted = json_block_match.group(1).strip()
        # Validate it's at least somewhat JSON-like
        if extracted.startswith(('{', '[')) and extracted.endswith(('}', ']')):
            return extracted
    
    # Try to find first complete JSON object or array in text
    stack = []
    start_idx = -1
    for i, char in enumerate(text):
        if char in '{[':
            if not stack:
                start_idx = i
            stack.append(char)
        elif char in '}]':
            if stack:
                stack.pop()
                if not stack and start_idx != -1:
                    # Found complete JSON
                    extracted = text[start_idx:i+1]
                    try:
                        json.loads(extracted)  # Validate
                        return extracted
                    except json.JSONDecodeError:
                        continue
    
    # Fallback: find first { or [ and try to parse from there
    for char in ['{', '[']:
        start_idx = text.find(char)
        if start_idx != -1:
            try:
                # Attempt to parse from start index to end
                json.loads(text[start_idx:])
                return text[start_idx:]
            except json.JSONDecodeError:
                # Try to find matching closing character
                end_idx = text.rfind('}' if char == '{' else ']')
                if end_idx > start_idx:
                    try:
                        json.loads(text[start_idx:end_idx+1])
                        return text[start_idx:end_idx+1]
                    except json.JSONDecodeError:
                        continue
    
    # Final fallback: return empty array if nothing works
    return "[]"
def safe_json_parse(json_text: str, default=None):
    """Safely parse JSON with multiple fallback strategies."""
    if default is None:
        default = []
    
    # First try direct parse
    try:
        return json.loads(json_text)
    except json.JSONDecodeError as e:
        pass
    
    # Try cleaning common issues
    try:
        # Remove trailing commas
        cleaned = re.sub(r',\s*([}\]])', r'\1', json_text)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    # Try extracting just the first complete object/array
    try:
        match = re.search(r'(\{.*?\}|\[.*?\])', json_text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
    except json.JSONDecodeError:
        pass
    
    return default

# AI-Powered Data Generation
class GeminiDataGenerator:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def generate_study_modules(self, field_of_study: str, skill_level: str, available_hours_per_week: int, 
                              study_duration_weeks: int, specific_goals: str = "") -> List[StudyModule]:
        """Generate study modules dynamically based on user input"""
        prompt = f"""
        Generate a comprehensive study plan for a {skill_level} level student in {field_of_study}.
        
        Context:
        - Available study time: {available_hours_per_week} hours per week
        - Study duration: {study_duration_weeks} weeks
        - Specific goals: {specific_goals}
        
        Generate 3-6 study modules in JSON format with the following structure:
        [
            {{
                "name": "Module Name",
                "priority": 1-10 (10 being highest priority),
                "estimated_hours": float (total hours needed),
                "difficulty": 1-5 (5 being most difficult),
                "dependencies": ["prerequisite module names"],
                "topics": ["topic1", "topic2", "topic3", "topic4", "topic5"],
                "deadline_days_from_now": integer (days from today)
            }}
        ]
        
        Make sure:
        - Total estimated hours don't exceed {available_hours_per_week * study_duration_weeks} hours
        - Modules are realistic and progressively build upon each other
        - Topics are specific and actionable
        - Deadlines are spread appropriately
        - Priorities reflect learning sequence and importance
        
        Return only the JSON array, no additional text. Do not wrap in markdown code blocks.
        """
        
        try:
            response = self.model.generate_content(prompt)
            raw_text = response.text.strip()
            json_text = extract_json(raw_text)
            modules_data = json.loads(json_text)
            
            modules = []
            for module_data in modules_data:
                module = StudyModule(
                    name=module_data['name'],
                    priority=module_data['priority'],
                    estimated_hours=module_data['estimated_hours'],
                    difficulty=module_data['difficulty'],
                    dependencies=module_data['dependencies'],
                    topics=module_data['topics'],
                    deadline=datetime.now() + timedelta(days=module_data['deadline_days_from_now'])
                )
                modules.append(module)
            
            return modules
            
        except Exception as e:
            st.error(f"Error generating modules: {e}")
            return []
    
    def generate_life_events(self, student_profile: str, study_period_weeks: int) -> List[Dict]:
        """Generate realistic life events for a student"""
        prompt = f"""
        Generate realistic life events for a student with this profile: {student_profile}
        Study period: {study_period_weeks} weeks from now
        
        Generate 4-8 life events that could affect study schedule in JSON format:
        [
            {{
                "name": "Event Name",
                "date": "YYYY-MM-DD" (within the next {study_period_weeks * 7} days),
                "end_date": "YYYY-MM-DD",
                "event_type": "exam/vacation/internship/sick_leave/festival/family/work/social",
                "impact_level": 1-5 (how much it affects study capacity)
            }}
        ]
        
        Make events realistic for a student:
        - Include common academic events (exams, assignments, presentations)
        - Add personal/social events (festivals, family gatherings, trips)
        - Consider work/internship commitments
        - Include some low-impact and high-impact events
        - Spread events throughout the time period
        
        Start dates from: {datetime.now().strftime('%Y-%m-%d')}
        Return only the JSON array, no additional text. Do not wrap in markdown code blocks.
        """
        
        try:
            response = self.model.generate_content(prompt)
            raw_text = response.text.strip()
            json_text = extract_json(raw_text)
            events_data = json.loads(json_text)
            return events_data
            
        except Exception as e:
            st.error(f"Error generating events: {e}")
            return []
    
    def generate_productivity_knowledge(self) -> List[str]:
        """Generate comprehensive productivity and learning knowledge base"""
        prompt = """
        Generate 15-20 comprehensive productivity and learning techniques that are evidence-based and scientifically proven.
        
        Format each as a detailed explanation including:
        - Technique name
        - How it works
        - When to use it
        - Specific implementation steps
        
        Cover areas like:
        - Memory and retention techniques
        - Focus and concentration methods
        - Time management strategies
        - Stress management techniques
        - Cognitive load management
        - Motivation and habit formation
        - Study environment optimization
        - Active learning methods
        
        Return as a JSON array of strings, each string being a complete explanation.
        Example format:
        ["Technique Name: Detailed explanation of how it works, when to use it, and implementation steps..."]
        
        Return only the JSON array, no additional text. Do not wrap in markdown code blocks.
        """
        
        try:
            response = self.model.generate_content(prompt)
            raw_text = response.text.strip()
            json_text = extract_json(raw_text)
            knowledge_list = json.loads(json_text)
            return knowledge_list
            
        except Exception as e:
            st.error(f"Error generating knowledge base: {e}")
            return [
                "Pomodoro Technique: Work for 25 minutes, then take a 5-minute break. This leverages the brain's natural attention spans.",
                "Spaced Repetition: Review material at increasing intervals to improve long-term retention.",
                "Active Recall: Test yourself instead of passive reading to strengthen memory pathways."
            ]
    
    def generate_study_recommendations(self, context: Dict) -> str:
        """Generate personalized study recommendations"""
        prompt = f"""
        You are an expert study coach. Based on the following student context, provide personalized study advice:
        
        Student Context:
        - Current fatigue level: {context.get('fatigue_level', 'Unknown')}
        - Weekly study hours: {context.get('weekly_hours', 'Unknown')}
        - Upcoming events: {context.get('upcoming_events', 'None')}
        - Current focus: {context.get('current_focus', 'General study')}
        - Field of study: {context.get('field_of_study', 'General')}
        - Skill level: {context.get('skill_level', 'Intermediate')}
        - Recent challenges: {context.get('challenges', 'None specified')}
        - Goals: {context.get('goals', 'Academic success')}
        
        Question: {context.get('question', 'How can I improve my study effectiveness?')}
        
        Provide specific, actionable advice that:
        - Addresses their current situation
        - Considers their fatigue and workload
        - Suggests concrete techniques
        - Includes timing and implementation details
        - Considers their field of study
        - Is realistic and achievable
        
        Keep response conversational but informative, around 150-200 words.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"I'd recommend focusing on your highest priority tasks while managing your energy levels. Consider taking breaks and adjusting your study intensity based on your current workload."

# Core Agent Classes (Updated to use AI generation)
class CognitiveStudyPlannerAgent:
    def __init__(self, gemini_generator: GeminiDataGenerator):
        self.name = "Cognitive Planner"
        self.current_plan = []
        self.gemini = gemini_generator
        self.cognitive_principles = {
            "spaced_repetition": True,
            "interleaving": True,
            "cognitive_load_management": True,
            "peak_performance_scheduling": True
        }
    
    def create_study_plan(self, modules: List[StudyModule], events: List[LifeEvent], 
                         fatigue_metrics: FatigueMetrics) -> List[Dict]:
        """Create intelligent study schedule using AI-powered optimization"""
        # Get AI-generated optimal scheduling
        context = {
            'modules': [{'name': m.name, 'hours': m.estimated_hours, 'difficulty': m.difficulty, 'priority': m.priority} for m in modules],
            'events': [{'name': e.name, 'date': e.date.strftime('%Y-%m-%d'), 'impact': e.impact_level} for e in events],
            'fatigue': {
                'mental_load': fatigue_metrics.mental_load,
                'stress_level': fatigue_metrics.stress_level,
                'weekly_hours': fatigue_metrics.weekly_hours
            }
        }
        
        plan = self._generate_ai_schedule(context, modules, events, fatigue_metrics)
        return sorted(plan, key=lambda x: x['date'])
    
    def _generate_ai_schedule(self, context: Dict, modules: List[StudyModule], 
                             events: List[LifeEvent], fatigue_metrics: FatigueMetrics) -> List[Dict]:
        """Generate study schedule using AI"""
        prompt = f"""
        Create an optimal study schedule based on cognitive science principles.
        
        Modules to schedule:
        {json.dumps(context['modules'], indent=2)}
        
        Upcoming events:
        {json.dumps(context['events'], indent=2)}
        
        Current fatigue metrics:
        - Mental load: {fatigue_metrics.mental_load}
        - Stress level: {fatigue_metrics.stress_level}
        - Weekly hours: {fatigue_metrics.weekly_hours}
        
        Generate a study schedule for the next 30 days that:
        - Follows spaced repetition principles
        - Considers cognitive load management
        - Avoids conflicts with high-impact events
        - Adjusts session intensity based on fatigue
        - Optimizes for peak performance times
        - Interleaves different subjects
        
        Return JSON format:
        [
            {{
                "date": "YYYY-MM-DD",
                "module": "Module Name",
                "duration": float (hours),
                "intensity": float (0.2-1.0),
                "topics": "Specific topic for this session",
                "cognitive_strategy": "Strategy name",
                "reasoning": "Why this session is scheduled this way"
            }}
        ]
        
        Start from: {datetime.now().strftime('%Y-%m-%d')}
        Return only JSON array, no additional text. Do not wrap in markdown code blocks.
        """
        
        try:
            response = self.gemini.model.generate_content(prompt)
            raw_text = response.text.strip()
            json_text = extract_json(raw_text)
            schedule_data = json.loads(json_text)
            
            # Convert to proper format
            plan = []
            for session in schedule_data:
                plan.append({
                    'date': datetime.strptime(session['date'], '%Y-%m-%d'),
                    'module': session['module'],
                    'duration': session['duration'],
                    'intensity': session['intensity'],
                    'topics': session['topics'],
                    'cognitive_strategy': session['cognitive_strategy'],
                    'reasoning': session['reasoning']
                })
            
            return plan
            
        except Exception as e:
            st.error(f"Error generating AI schedule: {e}")
            # Fallback to basic scheduling
            return self._create_basic_schedule(modules, events, fatigue_metrics)
    
    def _create_basic_schedule(self, modules: List[StudyModule], events: List[LifeEvent], 
                              fatigue_metrics: FatigueMetrics) -> List[Dict]:
        """Fallback basic scheduling"""
        plan = []
        current_date = datetime.now()
        
        for module in sorted(modules, key=lambda x: (x.deadline, -x.priority)):
            sessions_needed = int(np.ceil(module.estimated_hours / 2.0))
            
            for i in range(sessions_needed):
                while self._has_conflict(current_date, events):
                    current_date += timedelta(days=1)
                
                plan.append({
                    'date': current_date,
                    'module': module.name,
                    'duration': min(2.0, module.estimated_hours - i * 2.0),
                    'intensity': 0.7 * (1 - fatigue_metrics.stress_level * 0.3),
                    'topics': module.topics[i % len(module.topics)] if module.topics else "General study",
                    'cognitive_strategy': "Focused Study",
                    'reasoning': "Scheduled to avoid conflicts and manage workload"
                })
                
                current_date += timedelta(days=1)
        
        return plan
    
    def _has_conflict(self, date: datetime, events: List[LifeEvent]) -> bool:
        """Check if date conflicts with high-impact events"""
        for event in events:
            if (event.date.date() <= date.date() <= event.end_date.date() and 
                event.impact_level >= 4):
                return True
        return False

class LifeEventInterpreterAgent:
    def __init__(self, gemini_generator: GeminiDataGenerator):
        self.name = "Life Event Interpreter"
        self.gemini = gemini_generator
    
    def interpret_events(self, events_data: List[Dict]) -> List[LifeEvent]:
        """Convert raw event data to structured LifeEvent objects using AI interpretation"""
        if not events_data:
            return []
        
        prompt = f"""
        Analyze these events and classify them with proper impact levels:
        
        Events:
        {json.dumps(events_data, indent=2)}
        
        For each event, determine:
        - event_type: Choose from [exam, vacation, internship, sick_leave, festival, family, work, social, academic, personal]
        - impact_level: 1-5 scale where:
          1 = No impact on study capacity
          2 = Minor impact, can still study effectively
          3 = Moderate impact, reduced study capacity
          4 = Major impact, significantly affects study
          5 = Complete impact, no study possible
        
        Return JSON array with same events plus analysis:
        [
            {{
                "name": "original name",
                "date": "original date",
                "end_date": "original end_date or same as date",
                "event_type": "classified type",
                "impact_level": integer,
                "analysis_reasoning": "why this impact level was chosen"
            }}
        ]
        
        Return only JSON array, no additional text. Do not wrap in markdown code blocks.
        """
        
        try:
            response = self.gemini.model.generate_content(prompt)
            raw_text = response.text.strip()
            json_text = extract_json(raw_text)
            analyzed_events = json.loads(json_text)
            
            interpreted_events = []
            for event_data in analyzed_events:
                event = LifeEvent(
                    name=event_data['name'],
                    date=datetime.strptime(event_data['date'], '%Y-%m-%d'),
                    end_date=datetime.strptime(event_data.get('end_date', event_data['date']), '%Y-%m-%d'),
                    event_type=event_data['event_type'],
                    impact_level=event_data['impact_level']
                )
                interpreted_events.append(event)
            
            return interpreted_events
            
        except Exception as e:
            st.error(f"Error interpreting events: {e}")
            # Fallback to basic interpretation
            return self._basic_interpret_events(events_data)
    
    def _basic_interpret_events(self, events_data: List[Dict]) -> List[LifeEvent]:
        """Fallback basic event interpretation"""
        interpreted_events = []
        
        for event_data in events_data:
            # Basic classification
            name_lower = event_data['name'].lower()
            if any(word in name_lower for word in ['exam', 'test', 'quiz']):
                event_type, impact = 'exam', 5
            elif any(word in name_lower for word in ['vacation', 'trip', 'travel']):
                event_type, impact = 'vacation', 4
            elif any(word in name_lower for word in ['work', 'job', 'internship']):
                event_type, impact = 'work', 3
            elif any(word in name_lower for word in ['family', 'wedding', 'funeral']):
                event_type, impact = 'family', 3
            elif any(word in name_lower for word in ['sick', 'ill', 'medical']):
                event_type, impact = 'sick_leave', 5
            else:
                event_type, impact = 'personal', 2
            
            event = LifeEvent(
                name=event_data['name'],
                date=datetime.strptime(event_data['date'], '%Y-%m-%d'),
                end_date=datetime.strptime(event_data.get('end_date', event_data['date']), '%Y-%m-%d'),
                event_type=event_type,
                impact_level=impact
            )
            interpreted_events.append(event)
        
        return interpreted_events

class FatigueWorkloadMonitorAgent:
    def __init__(self, gemini_generator: GeminiDataGenerator):
        self.name = "Fatigue Monitor"
        self.gemini = gemini_generator
        self.fatigue_history = []
        
    def calculate_fatigue_metrics(self, study_history: List[Dict], 
                                 current_workload: float, student_profile: Dict = None) -> FatigueMetrics:
        """Calculate comprehensive fatigue metrics using AI analysis"""
        if not study_history:
            return FatigueMetrics(
                mental_load=0.2,
                weekly_hours=0,
                missed_sessions=0,
                stress_level=0.1,
                recommended_intensity=0.7
            )
        
        # Get AI analysis of fatigue patterns
        prompt = f"""
        Analyze this student's study pattern and calculate fatigue metrics:
        
        Recent study history:
        {json.dumps(study_history[-10:], indent=2, default=str)}
        
        Current workload: {current_workload} hours/week
        
        Student profile: {json.dumps(student_profile or {}, indent=2)}
        
        Calculate and return JSON:
        {{
            "mental_load": float (0-1, where 1 is completely overwhelmed),
            "weekly_hours": float (actual weekly study hours),
            "missed_sessions": integer (estimated missed sessions),
            "stress_level": float (0-1, overall stress assessment),
            "recommended_intensity": float (0.2-1.0, optimal study intensity),
            "analysis": "Brief explanation of the assessment"
        }}
        
        Consider:
        - Session completion rates
        - Duration vs planned time
        - Frequency of study
        - Quality ratings if available
        - Workload sustainability
        
        Return only JSON, no additional text. Do not wrap in markdown code blocks.
        """
        
        try:
            response = self.gemini.model.generate_content(prompt)
            raw_text = response.text.strip()
            json_text = extract_json(raw_text)
            metrics_data = json.loads(json_text)
            
            return FatigueMetrics(
                mental_load=metrics_data['mental_load'],
                weekly_hours=metrics_data['weekly_hours'],
                missed_sessions=metrics_data['missed_sessions'],
                stress_level=metrics_data['stress_level'],
                recommended_intensity=metrics_data['recommended_intensity']
            )
            
        except Exception as e:
            st.error(f"Error calculating fatigue metrics: {e}")
            # Fallback calculation
            return self._basic_fatigue_calculation(study_history, current_workload)
    
    def _basic_fatigue_calculation(self, study_history: List[Dict], current_workload: float) -> FatigueMetrics:
        """Fallback basic fatigue calculation"""
        last_week = datetime.now() - timedelta(days=7)
        recent_sessions = [s for s in study_history if s.get('date', datetime.now()) > last_week]
        
        weekly_hours = sum(s.get('duration', 0) for s in recent_sessions)
        missed_sessions = sum(1 for s in recent_sessions if not s.get('completed', True))
        
        mental_load = min(weekly_hours / 40.0, 1.0)
        stress_level = (missed_sessions / max(len(recent_sessions), 1)) * 0.5 + mental_load * 0.5
        
        if stress_level > 0.7:
            recommended_intensity = 0.4
        elif stress_level > 0.5:
            recommended_intensity = 0.6
        else:
            recommended_intensity = 0.8
        
        return FatigueMetrics(
            mental_load=mental_load,
            weekly_hours=weekly_hours,
            missed_sessions=missed_sessions,
            stress_level=stress_level,
            recommended_intensity=recommended_intensity
        )
    
    def suggest_recovery_actions(self, fatigue_metrics: FatigueMetrics, student_context: Dict = None) -> List[str]:
        """Generate personalized recovery suggestions using AI"""
        prompt = f"""
        Generate personalized recovery suggestions for a student with these fatigue metrics:
        
        Fatigue Metrics:
        - Mental load: {fatigue_metrics.mental_load}
        - Weekly hours: {fatigue_metrics.weekly_hours}
        - Missed sessions: {fatigue_metrics.missed_sessions}
        - Stress level: {fatigue_metrics.stress_level}
        - Recommended intensity: {fatigue_metrics.recommended_intensity}
        
        Student context: {json.dumps(student_context or {}, indent=2)}
        
        Generate 4-6 specific, actionable recovery suggestions that are:
        - Personalized to their situation
        - Realistic and achievable
        - Evidence-based
        - Varied (covering physical, mental, and study technique aspects)
        
        Return as JSON array of strings:
        ["🎯 Suggestion 1 with emoji and specific action", "💡 Suggestion 2...", ...]
        
        Return only JSON array, no additional text. Do not wrap in markdown code blocks.
        """
        
        try:
            response = self.gemini.model.generate_content(prompt)
            raw_text = response.text.strip()
            json_text = extract_json(raw_text)
            suggestions = json.loads(json_text)
            return suggestions
            
        except Exception as e:
            st.error(f"Error generating recovery suggestions: {e}")
            # Fallback suggestions
            if fatigue_metrics.stress_level > 0.8:
                return [
                    "🛑 Take a 2-day study break to reset your mental state",
                    "🧘 Practice 10-minute daily meditation to reduce stress",
                    "🚶 Take a 30-minute walk in nature daily",
                    "😴 Prioritize 8+ hours of sleep for mental recovery"
                ]
            elif fatigue_metrics.stress_level > 0.6:
                return [
                    "⏰ Reduce daily study hours by 25% this week",
                    "🎯 Focus only on highest priority topics",
                    "🔄 Switch to lighter study methods (videos, discussions)",
                    "💆 Take 15-minute breaks every hour"
                ]
            else:
                return [
                    "📚 Mix challenging and easy topics in each session",
                    "🎵 Try background music or white noise while studying",
                    "🥗 Maintain regular, healthy meals",
                    "💧 Keep a water bottle nearby and stay hydrated"
                ]

class RAGProductivityResearchAgent:
    def __init__(self, gemini_generator: GeminiDataGenerator):
        self.name = "RAG Research Agent"
        self.gemini = gemini_generator
        self.knowledge_base = []
        
        # Initialize with AI-generated knowledge base
        self._initialize_ai_knowledge_base()
    
    def _initialize_ai_knowledge_base(self):
        """Initialize knowledge base with AI-generated content"""
        try:
            knowledge_items = self.gemini.generate_productivity_knowledge()
            for item in knowledge_items:
                self.add_knowledge(item, "ai_generated_research")
        except Exception as e:
            st.error(f"Error initializing knowledge base: {e}")
    
    def add_knowledge(self, text: str, source: str = "manual"):
        """Add knowledge to the system"""
        self.knowledge_base.append({
            'text': text,
            'source': source,
            'timestamp': datetime.now()
        })
    
    def get_personalized_advice(self, context: Dict, query: str) -> str:
        """Get personalized advice using AI and knowledge base"""
        # Retrieve relevant knowledge
        relevant_knowledge = self._retrieve_relevant_knowledge(query)
        
        # Enhanced context for AI
        enhanced_context = context.copy()
        enhanced_context['question'] = query
        enhanced_context['relevant_research'] = relevant_knowledge
        
        return self.gemini.generate_study_recommendations(enhanced_context)
    
    def _retrieve_relevant_knowledge(self, query: str) -> str:
        """Retrieve relevant knowledge from knowledge base"""
        relevant_items = []
        query_words = set(query.lower().split())
        
        # Score each knowledge item based on relevance
        scored_items = []
        for item in self.knowledge_base:
            text_words = set(item['text'].lower().split())
            relevance_score = len(query_words.intersection(text_words))
            if relevance_score > 0:
                scored_items.append((relevance_score, item['text']))
        
        # Return top relevant items
        scored_items.sort(key=lambda x: x[0], reverse=True)
        return "\n\n".join([item[1] for item in scored_items[:3]])

# Updated Multi-Agent System Orchestrator
class StudyPlannerOrchestrator:
    def __init__(self, gemini_api_key: str):
        self.gemini_generator = GeminiDataGenerator(gemini_api_key)
        self.cognitive_planner = CognitiveStudyPlannerAgent(self.gemini_generator)
        self.event_interpreter = LifeEventInterpreterAgent(self.gemini_generator)
        self.fatigue_monitor = FatigueWorkloadMonitorAgent(self.gemini_generator)
        self.rag_agent = RAGProductivityResearchAgent(self.gemini_generator)
    
    def generate_complete_study_setup(self, user_input: Dict) -> Dict:
        """Generate complete study setup from user input using AI"""
        # Generate study modules
        modules = self.gemini_generator.generate_study_modules(
            field_of_study=user_input.get('field_of_study', 'General Studies'),
            skill_level=user_input.get('skill_level', 'Intermediate'),
            available_hours_per_week=user_input.get('hours_per_week', 20),
            study_duration_weeks=user_input.get('study_weeks', 12),
            specific_goals=user_input.get('goals', '')
        )
        
        # Generate life events
        events_data = self.gemini_generator.generate_life_events(
            student_profile=user_input.get('student_profile', 'College student'),
            study_period_weeks=user_input.get('study_weeks', 12)
        )
        
        # Create comprehensive plan
        comprehensive_plan = self.create_comprehensive_plan(modules, events_data, [])
        
        return {
            'modules': modules,
            'events_data': events_data,
            'comprehensive_plan': comprehensive_plan
        }
    
    def create_comprehensive_plan(self, modules: List[StudyModule], events_data: List[Dict], 
                                 study_history: List[Dict] = None) -> Dict:
        """Create comprehensive plan using all AI agents"""
        # Step 1: Interpret life events
        events = self.event_interpreter.interpret_events(events_data)
        
        # Step 2: Calculate fatigue metrics
        fatigue_metrics = self.fatigue_monitor.calculate_fatigue_metrics(
            study_history or [], 0
        )
        
        # Step 3: Create AI-powered study plan
        study_plan = self.cognitive_planner.create_study_plan(modules, events, fatigue_metrics)
        
        # Step 4: Get recovery suggestions
        recovery_suggestions = self.fatigue_monitor.suggest_recovery_actions(fatigue_metrics)
        
        return {
            'study_plan': study_plan,
            'events': events,
            'fatigue_metrics': asdict(fatigue_metrics),
            'recovery_suggestions': recovery_suggestions
        }
    
    def get_daily_recommendation(self, date: datetime, comprehensive_plan: Dict) -> Dict:
        """Get AI-powered daily study recommendation"""
        study_plan = comprehensive_plan['study_plan']
        
        # Find sessions for the specific date
        daily_sessions = [s for s in study_plan if s['date'].date() == date.date()]
        
        if not daily_sessions:
            return {
                'message': "📅 No study sessions scheduled for today. Consider light review or planning tomorrow's work.",
                'sessions': [],
                'total_hours': 0
            }
        
        total_hours = sum(s['duration'] for s in daily_sessions)
        
        return {
            'message': f"📚 You have {len(daily_sessions)} study session(s) planned for today ({total_hours:.1f} hours total)",
            'sessions': daily_sessions,
            'total_hours': total_hours
        }

def main():
    # Title and header
    st.markdown('<h1 class="main-title">🧠 NeuroPlanner | AI-Powered Study Assistant</h1>', unsafe_allow_html=True)
    
    # API Key Configuration
    with st.sidebar:
        st.markdown('<div class="sidebar-header"><h3>⚙️ Configuration</h3></div>', unsafe_allow_html=True)
        GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or st.text_input("Enter Gemini API Key", type="password")
    
    if not GEMINI_API_KEY:
        st.warning("🔑 Please provide your Gemini API key to use the AI-powered features.")
        st.stop()
    
    # Initialize orchestrator
    if not st.session_state.agents_initialized:
        try:
            st.session_state.orchestrator = StudyPlannerOrchestrator(GEMINI_API_KEY)
            st.session_state.agents_initialized = True
        except Exception as e:
            st.error(f"❌ Error initializing AI agents: {e}")
            st.stop()
    
    # Sidebar Configuration
    with st.sidebar:
        st.markdown("### 🎓 Study Configuration")
        
        # User input for study setup
        field_of_study = st.text_input("Field of Study", "Computer Science")
        skill_level = st.selectbox("Current Skill Level", 
                                  ["Beginner", "Intermediate", "Advanced"])
        hours_per_week = st.slider("Available Hours per Week", 5, 60, 20)
        study_weeks = st.slider("Study Duration (weeks)", 4, 24, 12)
        goals = st.text_area("Specific Goals", 
                            "Improve problem-solving skills and prepare for interviews")
        
        student_profile = st.text_input("Student Profile", 
                                      "College student with part-time work")
        
        # Generate Study Plan Button
        if st.button("🚀 Generate AI Study Plan", type="primary", use_container_width=True):
            with st.spinner("🤖 AI agents are analyzing and creating your personalized study plan..."):
                user_input = {
                    'field_of_study': field_of_study,
                    'skill_level': skill_level,
                    'hours_per_week': hours_per_week,
                    'study_weeks': study_weeks,
                    'goals': goals,
                    'student_profile': student_profile
                }
                
                try:
                    result = st.session_state.orchestrator.generate_complete_study_setup(user_input)
                    st.session_state.modules = result['modules']
                    st.session_state.events = result['events_data']
                    st.session_state.study_plan = result['comprehensive_plan']
                    st.success("✅ AI study plan generated successfully!")
                except Exception as e:
                    st.error(f"❌ Error generating study plan: {e}")
    
    # Main content area
    if st.session_state.study_plan:
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📅 Study Schedule", "📊 Analytics", "🎯 Daily Focus", 
            "📈 Progress Tracking", "🤖 AI Assistant"
        ])
        
        with tab1:
            st.markdown('<div class="tab-container">', unsafe_allow_html=True)
            st.header("📅 AI-Generated Study Schedule")
            
            # Display study modules
            st.markdown('<div class="card"><div class="card-header">🎓 Study Modules</div>', unsafe_allow_html=True)
            modules_df = pd.DataFrame([
                {
                    'Module': module.name,
                    'Priority': module.priority,
                    'Hours': module.estimated_hours,
                    'Difficulty': '⭐' * module.difficulty,
                    'Deadline': module.deadline.strftime('%Y-%m-%d'),
                    'Topics': ', '.join(module.topics[:3]) + ('...' if len(module.topics) > 3 else '')
                }
                for module in st.session_state.modules
            ])
            st.dataframe(modules_df, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display schedule
            st.markdown('<div class="card"><div class="card-header">📊 Weekly Schedule</div>', unsafe_allow_html=True)
            study_plan = st.session_state.study_plan['study_plan']
            
            if study_plan:
                schedule_df = pd.DataFrame([
                    {
                        'Date': session['date'].strftime('%Y-%m-%d'),
                        'Day': session['date'].strftime('%A'),
                        'Module': session['module'],
                        'Duration (hrs)': session['duration'],
                        'Intensity': f"{session['intensity']:.1%}",
                        'Focus': session['topics'],
                        'Strategy': session['cognitive_strategy']
                    }
                    for session in study_plan[:14]  # Show next 2 weeks
                ])
                st.dataframe(schedule_df, use_container_width=True)
                
                # Calendar visualization
                st.markdown('<div class="card-header">📅 Calendar View</div>', unsafe_allow_html=True)
                fig = px.timeline(
                    schedule_df.head(10),
                    x_start="Date",
                    x_end="Date",
                    y="Module",
                    color="Duration (hrs)",
                    title="Study Sessions Timeline"
                )
                st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)  # Close tab-container
        
        with tab2:
            st.markdown('<div class="tab-container">', unsafe_allow_html=True)
            st.header("📊 Study Analytics & Insights")
            
            # Fatigue metrics
            fatigue_metrics = st.session_state.study_plan['fatigue_metrics']
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-title">Mental Load</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{fatigue_metrics["mental_load"]:.0%}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-title">Weekly Hours</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{fatigue_metrics["weekly_hours"]:.1f}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-title">Stress Level</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{fatigue_metrics["stress_level"]:.0%}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown('<div class="metric-title">Recommended Intensity</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="metric-value">{fatigue_metrics["recommended_intensity"]:.0%}</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Workload distribution
            st.markdown('<div class="card"><div class="card-header">📈 Workload Distribution</div>', unsafe_allow_html=True)
            if study_plan:
                workload_data = []
                for session in study_plan:
                    week_start = session['date'] - timedelta(days=session['date'].weekday())
                    workload_data.append({
                        'Week': week_start.strftime('%Y-%m-%d'),
                        'Hours': session['duration'],
                        'Module': session['module']
                    })
                
                if workload_data:
                    workload_df = pd.DataFrame(workload_data)
                    weekly_hours = workload_df.groupby('Week')['Hours'].sum().reset_index()
                    
                    fig = px.bar(weekly_hours, x='Week', y='Hours', 
                               title="Weekly Study Hours Distribution")
                    st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Recovery suggestions
            st.markdown('<div class="card"><div class="card-header">💡 AI Recovery Suggestions</div>', unsafe_allow_html=True)
            recovery_suggestions = st.session_state.study_plan.get('recovery_suggestions', [])
            for suggestion in recovery_suggestions:
                st.info(suggestion)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)  # Close tab-container
        
        with tab3:
            st.markdown('<div class="tab-container">', unsafe_allow_html=True)
            st.header("🎯 Today's Focus")
            
            today = datetime.now()
            daily_rec = st.session_state.orchestrator.get_daily_recommendation(
                today, st.session_state.study_plan
            )
            
            st.markdown(f'<div class="card"><h3>{daily_rec["message"]}</h3></div>', unsafe_allow_html=True)
            
            if daily_rec['sessions']:
                for i, session in enumerate(daily_rec['sessions']):
                    st.markdown(f"""
                    <div class="session-card">
                        <h4>Session {i+1}: {session['module']} ({session['duration']:.1f}h)</h4>
                        <p><strong>Focus Topic:</strong> {session['topics']}</p>
                        <p><strong>Strategy:</strong> {session['cognitive_strategy']}</p>
                        <p><strong>Intensity Level:</strong> {session['intensity']:.1%}</p>
                        <p><strong>Reasoning:</strong> {session['reasoning']}</p>
                        <div class="stButton">
                            <button onclick="alert('Session {i+1} started!')">▶️ Start Session</button>
                            <button onclick="alert('Session {i+1} completed!')" style="margin-left:10px;">✅ Mark Complete</button>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)  # Close tab-container
        
        with tab4:
            st.markdown('<div class="tab-container">', unsafe_allow_html=True)
            st.header("📈 Progress Tracking")
            
            # Progress input
            st.markdown('<div class="card"><div class="card-header">📝 Log Study Session</div>', unsafe_allow_html=True)
            with st.form("progress_form"):
                session_date = st.date_input("Session Date", datetime.now().date())
                module_completed = st.selectbox("Module", [m.name for m in st.session_state.modules])
                hours_studied = st.slider("Hours Studied", 0.5, 8.0, 2.0, 0.5)
                quality_rating = st.slider("Session Quality (1-5)", 1, 5, 4)
                notes = st.text_area("Session Notes")
                
                if st.form_submit_button("📊 Log Session", use_container_width=True):
                    # Add to session state (in real app, save to database)
                    if 'study_history' not in st.session_state:
                        st.session_state.study_history = []
                    
                    st.session_state.study_history.append({
                        'date': session_date,
                        'module': module_completed,
                        'duration': hours_studied,
                        'quality': quality_rating,
                        'notes': notes,
                        'completed': True
                    })
                    st.success("Session logged successfully!")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Progress visualization
            if 'study_history' in st.session_state and st.session_state.study_history:
                st.markdown('<div class="card"><div class="card-header">📊 Study Progress</div>', unsafe_allow_html=True)
                history_df = pd.DataFrame(st.session_state.study_history)
                
                # Progress over time
                fig = px.line(history_df, x='date', y='duration', 
                            title="Study Hours Over Time")
                st.plotly_chart(fig, use_container_width=True)
                
                # Module progress
                module_progress = history_df.groupby('module')['duration'].sum().reset_index()
                fig2 = px.pie(module_progress, values='duration', names='module',
                            title="Time Distribution by Module")
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)  # Close tab-container
        
        with tab5:
            st.markdown('<div class="tab-container">', unsafe_allow_html=True)
            st.header("🤖 AI Study Assistant")
            
            # Chat interface
            st.markdown('<div class="card"><div class="card-header">💬 Ask Your AI Study Coach</div>', unsafe_allow_html=True)
            
            # Display chat history
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f'<div class="chat-message-user">{message["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message-assistant">{message["content"]}</div>', unsafe_allow_html=True)
            
            # Chat input
            if user_question := st.chat_input("Ask about study strategies, time management, or specific challenges..."):
                # Add user message to history
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_question
                })
                st.markdown(f'<div class="chat-message-user">{user_question}</div>', unsafe_allow_html=True)
                
                # Generate AI response
                context = {
                    'field_of_study': field_of_study,
                    'skill_level': skill_level,
                    'weekly_hours': hours_per_week,
                    'fatigue_level': st.session_state.study_plan['fatigue_metrics']['stress_level'] if st.session_state.study_plan else 0.5,
                    'current_focus': st.session_state.modules[0].name if st.session_state.modules else "General",
                    'goals': goals,
                    'upcoming_events': len(st.session_state.events) if hasattr(st.session_state, 'events') else 0,
                    'challenges': user_question
                }
                
                with st.spinner("🤔 AI is thinking..."):
                    try:
                        ai_response = st.session_state.orchestrator.rag_agent.get_personalized_advice(
                            context, user_question
                        )
                        
                        # Add AI response to history
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': ai_response
                        })
                        st.markdown(f'<div class="chat-message-assistant">{ai_response}</div>', unsafe_allow_html=True)
                    except Exception as e:
                        error_msg = f"I apologize, but I encountered an error: {e}. Please try rephrasing your question."
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': error_msg
                        })
                        st.markdown(f'<div class="chat-message-assistant">{error_msg}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Quick action buttons
            st.markdown('<div class="card"><div class="card-header">🎯 Quick Actions</div>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("💡 Study Tips", use_container_width=True):
                    tips_context = {
                        'field_of_study': field_of_study,
                        'skill_level': skill_level,
                        'weekly_hours': hours_per_week,
                        'fatigue_level': st.session_state.study_plan['fatigue_metrics']['stress_level'] if st.session_state.study_plan else 0.5,
                        'current_focus': st.session_state.modules[0].name if st.session_state.modules else "General",
                        'goals': goals,
                        'upcoming_events': len(st.session_state.events) if hasattr(st.session_state, 'events') else 0,
                        'challenges': "Requesting general study tips"
                    }
                    with st.spinner("Generating tips..."):
                        tips = st.session_state.orchestrator.rag_agent.get_personalized_advice(
                            tips_context, "Give me personalized study tips"
                        )
                        st.info(tips)
            
            with col2:
                if st.button("⚡ Motivation Boost", use_container_width=True):
                    motivation_context = {
                        'field_of_study': field_of_study,
                        'skill_level': skill_level,
                        'weekly_hours': hours_per_week,
                        'fatigue_level': st.session_state.study_plan['fatigue_metrics']['stress_level'] if st.session_state.study_plan else 0.5,
                        'current_focus': st.session_state.modules[0].name if st.session_state.modules else "General",
                        'goals': goals,
                        'upcoming_events': len(st.session_state.events) if hasattr(st.session_state, 'events') else 0,
                        'challenges': "Need motivation boost"
                    }
                    with st.spinner("Generating motivation..."):
                        motivation = st.session_state.orchestrator.rag_agent.get_personalized_advice(
                            motivation_context, "Give me motivation to continue studying"
                        )
                        st.success(motivation)
            
            with col3:
                if st.button("🔄 Plan Adjustment", use_container_width=True):
                    adjust_context = {
                        'field_of_study': field_of_study,
                        'skill_level': skill_level,
                        'weekly_hours': hours_per_week,
                        'fatigue_level': st.session_state.study_plan['fatigue_metrics']['stress_level'] if st.session_state.study_plan else 0.5,
                        'current_focus': st.session_state.modules[0].name if st.session_state.modules else "General",
                        'goals': goals,
                        'upcoming_events': len(st.session_state.events) if hasattr(st.session_state, 'events') else 0,
                        'challenges': "Requesting plan adjustment advice"
                    }
                    with st.spinner("Analyzing plan..."):
                        adjustment = st.session_state.orchestrator.rag_agent.get_personalized_advice(
                            adjust_context, "How can I optimize my current study plan?"
                        )
                        st.warning(adjustment)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)  # Close tab-container
    
    else:
        # Welcome screen
        st.markdown("""
        <div class="card" style="text-align:center;">
            <h2>Welcome to NeuroPlanner! 🧠</h2>
            <p>Your AI-powered study assistant that creates personalized learning plans using cognitive science</p>
        </div>
        
        <div class="card">
            <h3>✨ How It Works</h3>
            <div class="stColumns">
                <div class="stColumn">
                    <h4>1. Configure Your Study</h4>
                    <p>Set your field, skill level, and availability in the sidebar</p>
                </div>
                <div class="stColumn">
                    <h4>2. Generate AI Plan</h4>
                    <p>Our agents will create a personalized cognitive schedule</p>
                </div>
                <div class="stColumn">
                    <h4>3. Follow & Track</h4>
                    <p>Get daily focus areas and track your progress</p>
                </div>
            </div>
        </div>
        
        <div class="card">
            <h3>🤖 AI Agents Working for You</h3>
            <div style="display:grid; grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div class="metric-card">
                    <h4>🧠 Cognitive Planner</h4>
                    <p>Creates optimized schedules using learning science</p>
                </div>
                <div class="metric-card">
                    <h4>📅 Event Interpreter</h4>
                    <p>Analyzes life events and their impact on your study</p>
                </div>
                <div class="metric-card">
                    <h4>⚖️ Fatigue Monitor</h4>
                    <p>Tracks your energy and suggests recovery actions</p>
                </div>
                <div class="metric-card">
                    <h4>🔍 Research Agent</h4>
                    <p>Provides evidence-based study advice</p>
                </div>
            </div>
        </div>
        
        <div class="card">
            <div style="text-align:center;">
                <h3>Ready to transform your study habits?</h3>
                <p>Configure your study preferences in the sidebar and generate your AI-powered plan!</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()