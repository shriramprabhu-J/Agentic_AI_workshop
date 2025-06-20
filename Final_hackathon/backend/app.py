import os
import uuid
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Any

from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from motor.motor_asyncio import AsyncIOMotorClient
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from fastapi.middleware.cors import CORSMiddleware
from tenacity import retry, stop_after_attempt, wait_exponential

# Initialize FastAPI
app = FastAPI(
    title="Agentic Code Feedback System",
    description="AI-powered debugging companion for LMS learners",
    version="1.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyATDLOm55IA60SZqd4mUmdCQzhsif5-1aM")
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DB_NAME = "lms_code_feedback"

# Initialize MongoDB client
mongo_client = AsyncIOMotorClient(MONGODB_URL)
db = mongo_client[DB_NAME]

# Collections
submissions_col = db["code_submissions"]
feedback_col = db["feedback_history"]

# Initialize Gemini LLM with retry mechanism
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def safe_llm_invoke(chain, input_data):
    return chain.invoke(input_data)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.3,
    max_retries=3
)

# Initialize RAG components
corpus = [
    {"concept": "Loop Invariants", "content": "A loop invariant is a condition that is true before and after each iteration of a loop. It helps ensure loop correctness."},
    {"concept": "Recursion Base Case", "content": "Every recursive function must have a base case to terminate the recursion, otherwise it will cause a stack overflow."},
    {"concept": "Time Complexity", "content": "Big O notation describes the upper bound of an algorithm's running time. Common complexities: O(1), O(log n), O(n), O(n log n), O(n²)."},
    {"concept": "Syntax Errors", "content": "Syntax errors occur when code violates language grammar rules. Common examples: missing colons in Python, missing semicolons in JavaScript."},
    {"concept": "Data Structures", "content": "Choosing appropriate data structures is crucial for performance. Arrays provide O(1) access, HashMaps provide O(1) lookups, Trees provide O(log n) searches."},
]

# Create vector store for RAG using FAISS
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
texts = [f"{item['concept']}: {item['content']}" for item in corpus]
metadatas = [{"concept": item["concept"]} for item in corpus]
vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

# ----- Pydantic Models -----
class SyntaxIssue(BaseModel):
    line: int
    message: str
    fix_suggestion: str

class LogicFlaw(BaseModel):
    context: str
    explanation: str

class OptimizationSuggestion(BaseModel):
    original_code: str
    optimized_code: str
    rationale: str

class Explanation(BaseModel):
    concept: str
    explanation: str
    example_code: str

    @field_validator('concept', 'explanation', 'example_code', mode='before')
    def convert_to_string(cls, v):
        if isinstance(v, list):
            return "\n".join(v)
        return str(v)

class HintStage(BaseModel):
    level: int
    hint: str
    timestamp: datetime

class CodeSubmission(BaseModel):
    learner_id: str = Field(..., example="learner123")
    code: str = Field(..., example="def sum(a,b):\n    return a + b")
    language: str = Field(..., example="python")

class FullFeedbackResponse(BaseModel):
    session_id: str
    syntax_issues: List[SyntaxIssue] = []
    logic_flaws: List[LogicFlaw] = []
    optimizations: List[OptimizationSuggestion] = []
    explanations: List[Explanation] = []
    hint_trail: List[HintStage] = []
    final_fix: Optional[str] = None

# ----- Tool Functions -----
class AgentTools:
    @staticmethod
    def syntax_validator(code: str, language: str) -> List[Dict]:
        """Tool to validate code syntax"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert {language} developer. Analyze code for syntax errors."),
            ("human", "Code:\n{code}\n\nIdentify all syntax issues. For each: line number, error description, fix suggestion. Return JSON array with keys: line, message, fix_suggestion.")
        ])
        chain = prompt | llm | JsonOutputParser()
        try:
            result = safe_llm_invoke(chain, {"code": code, "language": language})
            # Ensure we always return a list of dictionaries
            if isinstance(result, dict):
                return [result]
            return result
        except Exception as e:
            print(f"Syntax validation failed: {e}")
            return []

    @staticmethod
    def logic_consistency(code: str, language: str) -> List[Dict]:
        """Tool to check logical consistency"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Analyze {language} code for logical flaws like infinite loops, unreachable code, etc."),
            ("human", "Code:\n{code}\n\nFor each flaw: context and explanation. Return JSON array with keys: context, explanation.")
        ])
        chain = prompt | llm | JsonOutputParser()
        try:
            result = safe_llm_invoke(chain, {"code": code, "language": language})
            # Ensure we always return a list of dictionaries
            if isinstance(result, dict):
                return [result]
            return result
        except Exception as e:
            print(f"Logic check failed: {e}")
            return []

    @staticmethod
    def optimization_advisor(code: str, language: str) -> List[Dict]:
        """Tool to suggest optimizations"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Optimize {language} code for performance (time/space complexity)."),
            ("human", "Code:\n{code}\n\nFor each suggestion: original code, optimized code, Big-O rationale. Return JSON array with keys: original_code, optimized_code, rationale.")
        ])
        chain = prompt | llm | JsonOutputParser()
        try:
            result = safe_llm_invoke(chain, {"code": code, "language": language})
            # Ensure we always return a list of dictionaries
            if isinstance(result, dict):
                return [result]
            return result
        except Exception as e:
            print(f"Optimization failed: {e}")
            return []

    @staticmethod
    def feedback_explainer(syntax_issues: List[Dict], logic_flaws: List[Dict]) -> List[Dict]:
        """RAG-based explanation tool"""
        explanations = []
        issues = []
        
        # Combine all issues with safe key access
        for issue in syntax_issues:
            if not isinstance(issue, dict):
                continue
            message = issue.get('message', 'Unknown error')
            line = issue.get('line', 'Unknown line')
            issues.append({
                "type": "syntax",
                "description": f"{message} at line {line}"
            })
        
        for flaw in logic_flaws:
            if not isinstance(flaw, dict):
                continue
            context = flaw.get('context', 'Unknown context')
            explanation = flaw.get('explanation', 'Unknown explanation')
            issues.append({
                "type": "logic",
                "description": f"{context}: {explanation}"
            })
        
        # Process each issue
        for issue in issues:
            result = AgentTools._generate_explanation(issue)
            if result:
                explanations.append(result)
        return explanations
    
    @staticmethod
    def _generate_explanation(issue: Dict) -> Dict:
        """Helper function to generate explanation for a single issue"""
        try:
            # RAG retrieval
            docs = retriever.invoke(issue["description"])
            rag_context = "\n\n".join([doc.page_content for doc in docs])
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", "Explain programming concepts to beginners using course materials."),
                ("human", "Issue: {issue_description}\nContext:\n{rag_context}\nProvide concept name, simple explanation, annotated example. Return JSON with keys: concept, explanation, example_code.")
            ])
            chain = prompt | llm | JsonOutputParser()
            result = safe_llm_invoke(chain, {
                "issue_description": issue["description"],
                "rag_context": rag_context
            })
            return result
        except Exception as e:
            print(f"Explanation generation failed: {e}")
            return None

    @staticmethod
    def progressive_hinter(
        syntax_issues: List[Dict], 
        logic_flaws: List[Dict], 
        explanations: List[Dict], 
        code: str
    ) -> Dict:
        """Tool to generate progressive hints"""
        issues_summary = "\n".join([
            f"- {issue.get('message', 'Unknown error')} (Line {issue.get('line', '?')})" 
            for issue in syntax_issues
        ] + [
            f"- {flaw.get('context', 'Unknown context')}: {flaw.get('explanation', 'Unknown explanation')}" 
            for flaw in logic_flaws
        ])
        
        # Add explanations to context
        explanations_summary = "\n".join([
            f"- {exp.get('concept', 'Unknown concept')}: {exp.get('explanation', 'No explanation')}" 
            for exp in explanations
        ])
        
        full_context = f"Issues:\n{issues_summary}\n\nExplanations:\n{explanations_summary}"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a programming tutor. Provide progressive hints to help learners fix their code."),
            ("human", "Code:\n{code}\nContext:\n{context}\nGenerate 3 hints: 1. General direction, 2. Specific guidance, 3. Near-solution pointer. Then provide full fixed code. Return JSON with keys: hints (array of strings), final_fix (string).")
        ])
        chain = prompt | llm | JsonOutputParser()
        try:
            result = safe_llm_invoke(chain, {"code": code, "context": full_context})
            # Ensure we get a list of hints
            hints = result.get("hints", [])
            if isinstance(hints, str):
                hints = [hints]
                
            return {
                "hint_trail": [
                    {"level": i+1, "hint": hint, "timestamp": datetime.now().isoformat()}
                    for i, hint in enumerate(hints)
                ],
                "final_fix": result.get("final_fix", "No fix generated")
            }
        except Exception as e:
            print(f"Hint generation failed: {e}")
            return {
                "hint_trail": [],
                "final_fix": "Error generating fix"
            }

# ----- Manual Workflow Orchestration -----
def run_manual_workflow(submission: CodeSubmission) -> dict:
    """Execute the full workflow manually to ensure proper sequencing"""
    print("Starting manual workflow execution")
    
    # Step 1: Syntax Validation
    syntax_issues = AgentTools.syntax_validator(submission.code, submission.language)
    print(f"Syntax issues found: {len(syntax_issues)}")
    
    # Step 2: Logic Consistency Check
    logic_flaws = AgentTools.logic_consistency(submission.code, submission.language)
    print(f"Logic flaws found: {len(logic_flaws)}")
    
    # Step 3: Optimization Suggestions
    optimizations = AgentTools.optimization_advisor(submission.code, submission.language)
    print(f"Optimizations found: {len(optimizations)}")
    
    # Step 4: Feedback Explanations (RAG-powered)
    explanations = AgentTools.feedback_explainer(syntax_issues, logic_flaws)
    print(f"Explanations generated: {len(explanations)}")
    
    # Step 5: Progressive Hints
    hinter_result = AgentTools.progressive_hinter(
        syntax_issues, 
        logic_flaws, 
        explanations, 
        submission.code
    )
    print(f"Hints generated: {len(hinter_result.get('hint_trail', []))}")
    
    return {
        "syntax_issues": syntax_issues,
        "logic_flaws": logic_flaws,
        "optimizations": optimizations,
        "explanations": explanations,
        "hint_trail": hinter_result.get("hint_trail", []),
        "final_fix": hinter_result.get("final_fix", "")
    }

# ----- MongoDB Storage -----
async def store_submission(session_id: str, submission: CodeSubmission):
    await submissions_col.insert_one({
        "session_id": session_id,
        "learner_id": submission.learner_id,
        "code": submission.code,
        "language": submission.language,
        "timestamp": datetime.now()
    })

async def store_feedback(session_id: str, feedback: dict):
    await feedback_col.insert_one({
        "session_id": session_id,
        "syntax_issues": feedback.get("syntax_issues", []),
        "logic_flaws": feedback.get("logic_flaws", []),
        "optimizations": feedback.get("optimizations", []),
        "explanations": feedback.get("explanations", []),
        "hint_trail": feedback.get("hint_trail", []),
        "final_fix": feedback.get("final_fix", "")
    })

# ----- FastAPI Endpoints -----
@app.on_event("startup")
async def startup_db():
    try:
        await db.command("ping")
        print("✅ MongoDB connected successfully")
        
        # Create indexes for performance
        await submissions_col.create_index("learner_id")
        await feedback_col.create_index("session_id")
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        raise

@app.post("/analyze-code", response_model=FullFeedbackResponse, status_code=status.HTTP_201_CREATED)
async def analyze_code(submission: CodeSubmission, background_tasks: BackgroundTasks):
    try:
        session_id = str(uuid.uuid4())
        print(f"🚀 Starting analysis for session: {session_id}")
        
        # Run manual workflow
        loop = asyncio.get_running_loop()
        feedback_data = await loop.run_in_executor(None, run_manual_workflow, submission)
        
        # Prepare response
        response = FullFeedbackResponse(
            session_id=session_id,
            syntax_issues=[
                SyntaxIssue(**issue) for issue in feedback_data.get("syntax_issues", [])
                if isinstance(issue, dict)
            ],
            logic_flaws=[
                LogicFlaw(**flaw) for flaw in feedback_data.get("logic_flaws", [])
                if isinstance(flaw, dict)
            ],
            optimizations=[
                OptimizationSuggestion(**opt) for opt in feedback_data.get("optimizations", [])
                if isinstance(opt, dict)
            ],
            explanations=[
                Explanation(**exp) for exp in feedback_data.get("explanations", [])
                if isinstance(exp, dict)
            ],
            hint_trail=[
                HintStage(**hint) for hint in feedback_data.get("hint_trail", [])
                if isinstance(hint, dict)
            ],
            final_fix=feedback_data.get("final_fix", "")
        )
        
        # Store in MongoDB (async background)
        background_tasks.add_task(store_submission, session_id, submission)
        background_tasks.add_task(store_feedback, session_id, feedback_data)
        
        print(f"✅ Analysis complete for session: {session_id}")
        return response
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

@app.get("/history/{learner_id}", response_model=List[Dict])
async def get_history(learner_id: str):
    try:
        print(f"📖 Fetching history for learner: {learner_id}")
        submissions = await submissions_col.find(
            {"learner_id": learner_id},
            {"_id": 0, "session_id": 1, "timestamp": 1, "language": 1}
        ).to_list(100)
        
        if not submissions:
            return []
        
        session_ids = [s["session_id"] for s in submissions]
        feedback_cursor = feedback_col.find(
            {"session_id": {"$in": session_ids}},
            {"_id": 0, "session_id": 1, "syntax_issues": 1, "logic_flaws": 1}
        )
        feedback_list = await feedback_cursor.to_list(100)
        
        # Combine results
        history = []
        for sub in submissions:
            fb = next((f for f in feedback_list if f["session_id"] == sub["session_id"]), {})
            issue_count = len(fb.get("syntax_issues", [])) + len(fb.get("logic_flaws", []))
            history.append({
                "session_id": sub["session_id"],
                "timestamp": sub["timestamp"],
                "language": sub["language"],
                "issue_count": issue_count
            })
        
        print(f"📚 Found {len(history)} submissions for learner: {learner_id}")
        return history
        
    except Exception as e:
        print(f"❌ History fetch failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"History fetch failed: {str(e)}"
        )

@app.get("/hints/{session_id}", response_model=Dict)
async def get_hints(session_id: str):
    try:
        print(f"💡 Fetching hints for session: {session_id}")
        session_data = await feedback_col.find_one(
            {"session_id": session_id},
            {"_id": 0, "hint_trail": 1, "final_fix": 1}
        )
        
        if not session_data:
            print(f"❌ Session not found: {session_id}")
            raise HTTPException(
                status_code=404,
                detail="Session not found"
            )
        
        return session_data
        
    except Exception as e:
        print(f"❌ Hints fetch failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Hints fetch failed: {str(e)}"
        )

# ----- Error Handling -----
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": f"Internal server error: {str(exc)}"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")