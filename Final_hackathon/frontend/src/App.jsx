

import { useState, useEffect, useRef } from "react"
import './index.css'; // or './App.css'

// Monaco Editor Component
function MonacoEditor({ value, onChange, language, height = "400px", readOnly = false }) {
  const editorRef = useRef(null)
  const containerRef = useRef(null)

  useEffect(() => {
    // Dynamically import Monaco Editor
    const loadMonaco = async () => {
      try {
        const monaco = await import("monaco-editor")

        // Configure Monaco
        monaco.editor.defineTheme("custom-theme", {
          base: "vs",
          inherit: true,
          rules: [],
          colors: {
            "editor.background": "#f8fafc",
            "editor.lineHighlightBackground": "#e2e8f0",
          },
        })

        if (containerRef.current && !editorRef.current) {
          editorRef.current = monaco.editor.create(containerRef.current, {
            value: value || "",
            language: language,
            theme: "custom-theme",
            fontSize: 14,
            lineNumbers: "on",
            roundedSelection: false,
            scrollBeyondLastLine: false,
            automaticLayout: true,
            minimap: { enabled: false },
            wordWrap: "on",
            lineHeight: 20,
            fontFamily: 'Monaco, Menlo, "Ubuntu Mono", monospace',
            readOnly: readOnly,
          })

          // Listen for content changes
          if (!readOnly) {
            editorRef.current.onDidChangeModelContent(() => {
              const currentValue = editorRef.current.getValue()
              if (onChange) {
                onChange(currentValue)
              }
            })
          }
        }
      } catch (error) {
        console.error("Failed to load Monaco Editor:", error)
      }
    }

    loadMonaco()

    return () => {
      if (editorRef.current) {
        editorRef.current.dispose()
        editorRef.current = null
      }
    }
  }, [])

  useEffect(() => {
    if (editorRef.current && value !== editorRef.current.getValue()) {
      editorRef.current.setValue(value || "")
    }
  }, [value])

  useEffect(() => {
    if (editorRef.current) {
      const model = editorRef.current.getModel()
      if (model) {
        if (window.monaco) {
          window.monaco.editor.setModelLanguage(model, language)
        }
      }
    }
  }, [language])

  return <div ref={containerRef} style={{ height, border: "1px solid #e2e8f0", borderRadius: "6px" }} />
}

// Icons as SVG components with proper sizing
const CodeIcon = () => (
  <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
  </svg>
)

const BugIcon = () => (
  <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M12 9v3m0 0v3m0-3h3m-3 0H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z"
    />
  </svg>
)

const HistoryIcon = () => (
  <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
    />
  </svg>
)

const SendIcon = () => (
  <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
  </svg>
)

const LoaderIcon = () => (
  <svg width="16" height="16" className="animate-spin" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
    />
  </svg>
)

const AlertIcon = () => (
  <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z"
    />
  </svg>
)

const CheckIcon = () => (
  <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
    />
  </svg>
)

const LightbulbIcon = () => (
  <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
    />
  </svg>
)

const BookIcon = () => (
  <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"
    />
  </svg>
)

const ZapIcon = () => (
  <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
  </svg>
)

const UserIcon = () => (
  <svg width="16" height="16" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
    />
  </svg>
)

const FileCodeIcon = () => (
  <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
    />
  </svg>
)

const ClockIcon = () => (
  <svg width="12" height="12" fill="none" stroke="currentColor" viewBox="0 0 24 24">
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
    />
  </svg>
)

// Large icons for empty states
const LargeCodeIcon = () => (
  <svg
    width="48"
    height="48"
    className="text-gray-400 mx-auto mb-4"
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
  </svg>
)

const LargeHistoryIcon = () => (
  <svg
    width="48"
    height="48"
    className="text-gray-400 mx-auto mb-4"
    fill="none"
    stroke="currentColor"
    viewBox="0 0 24 24"
  >
    <path
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth={2}
      d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
    />
  </svg>
)

export default function CodeFeedbackSystem() {
  // State management
  const [activeTab, setActiveTab] = useState("analyze")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Form state
  const [learnerId, setLearnerId] = useState("learner123")
  const [code, setCode] = useState(`def factorial(n):
    if n = 1:
        return 1
    else:
        return n * factorial(n-1)`)
  const [language, setLanguage] = useState("python")

  // Response state
  const [feedback, setFeedback] = useState(null)
  const [history, setHistory] = useState([])
  const [hints, setHints] = useState(null)

  // Backend URL - adjust this to match your FastAPI server
  const API_BASE = "http://localhost:8000"

  // Language mapping for Monaco Editor
  const getMonacoLanguage = (lang) => {
    const languageMap = {
      python: "python",
      javascript: "javascript",
      java: "java",
      cpp: "cpp",
      c: "c",
    }
    return languageMap[lang] || "python"
  }

  // Analyze code function
  const analyzeCode = async () => {
    if (!code.trim() || !learnerId.trim()) {
      setError("Please provide both learner ID and code")
      return
    }

    setLoading(true)
    setError(null)

    try {
      const submission = {
        learner_id: learnerId,
        code: code,
        language: language,
      }

      const response = await fetch(`${API_BASE}/analyze-code`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(submission),
      })

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`)
      }

      const result = await response.json()
      setFeedback(result)
      setActiveTab("results")
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed")
    } finally {
      setLoading(false)
    }
  }

  // Fetch history function
  const fetchHistory = async () => {
    if (!learnerId.trim()) {
      setError("Please provide learner ID")
      return
    }

    setLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE}/history/${learnerId}`)

      if (!response.ok) {
        throw new Error(`History fetch failed: ${response.statusText}`)
      }

      const result = await response.json()
      setHistory(result)
      setActiveTab("history")
    } catch (err) {
      setError(err instanceof Error ? err.message : "History fetch failed")
    } finally {
      setLoading(false)
    }
  }

  // Fetch hints function
  const fetchHints = async (sessionId) => {
    setLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_BASE}/hints/${sessionId}`)

      if (!response.ok) {
        throw new Error(`Hints fetch failed: ${response.statusText}`)
      }

      const result = await response.json()
      setHints(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Hints fetch failed")
    } finally {
      setLoading(false)
    }
  }

  // Auto-fetch history when learner ID changes
  useEffect(() => {
    if (learnerId.trim() && activeTab === "history") {
      fetchHistory()
    }
  }, [learnerId, activeTab])

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 p-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-2">🤖 Agentic Code Feedback System</h1>
          <p className="text-lg text-gray-600">AI-powered debugging companion for LMS learners</p>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="mb-6 p-4 border border-red-200 bg-red-50 rounded-lg flex items-center gap-2">
            <AlertIcon />
            <span className="text-red-800">{error}</span>
          </div>
        )}

        {/* Main Tabs */}
        <div className="w-full">
          {/* Tab Navigation */}
          <div className="flex bg-white rounded-lg p-1 mb-6 shadow-sm">
            <button
              onClick={() => setActiveTab("analyze")}
              className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-md font-medium transition-colors ${
                activeTab === "analyze"
                  ? "bg-blue-500 text-white shadow-sm"
                  : "text-gray-600 hover:text-gray-900 hover:bg-gray-50"
              }`}
            >
              <CodeIcon />
              Analyze Code
            </button>
            <button
              onClick={() => setActiveTab("results")}
              className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-md font-medium transition-colors ${
                activeTab === "results"
                  ? "bg-blue-500 text-white shadow-sm"
                  : "text-gray-600 hover:text-gray-900 hover:bg-gray-50"
              }`}
            >
              <BugIcon />
              Results
            </button>
            <button
              onClick={() => setActiveTab("history")}
              className={`flex-1 flex items-center justify-center gap-2 px-4 py-2 rounded-md font-medium transition-colors ${
                activeTab === "history"
                  ? "bg-blue-500 text-white shadow-sm"
                  : "text-gray-600 hover:text-gray-900 hover:bg-gray-50"
              }`}
            >
              <HistoryIcon />
              History
            </button>
          </div>

          {/* Analyze Tab */}
          {activeTab === "analyze" && (
            <div className="space-y-6">
              <div className="bg-white rounded-lg shadow-sm border">
                <div className="p-6 border-b">
                  <h2 className="text-xl font-semibold flex items-center gap-2 mb-2">
                    <FileCodeIcon />
                    Submit Code for Analysis
                  </h2>
                  <p className="text-gray-600">Enter your code below to get comprehensive AI-powered feedback</p>
                </div>
                <div className="p-6 space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <label htmlFor="learner-id" className="block text-sm font-medium text-gray-700 mb-1">
                        Learner ID
                      </label>
                      <input
                        id="learner-id"
                        type="text"
                        value={learnerId}
                        onChange={(e) => setLearnerId(e.target.value)}
                        placeholder="Enter your learner ID"
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      />
                    </div>
                    <div>
                      <label htmlFor="language" className="block text-sm font-medium text-gray-700 mb-1">
                        Programming Language
                      </label>
                      <select
                        id="language"
                        value={language}
                        onChange={(e) => setLanguage(e.target.value)}
                        className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                      >
                        <option value="python">Python</option>
                        <option value="javascript">JavaScript</option>
                        <option value="java">Java</option>
                        <option value="cpp">C++</option>
                        <option value="c">C</option>
                      </select>
                    </div>
                  </div>

                  <div>
                    <label htmlFor="code" className="block text-sm font-medium text-gray-700 mb-2">
                      Your Code
                    </label>
                    <MonacoEditor
                      value={code}
                      onChange={setCode}
                      language={getMonacoLanguage(language)}
                      height="400px"
                    />
                  </div>

                  <button
                    onClick={analyzeCode}
                    disabled={loading}
                    className="w-full bg-blue-600 text-white py-3 px-4 rounded-md font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2 transition-colors"
                  >
                    {loading ? (
                      <>
                        <LoaderIcon />
                        Analyzing...
                      </>
                    ) : (
                      <>
                        <SendIcon />
                        Analyze Code
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Results Tab */}
          {activeTab === "results" && (
            <div className="space-y-6">
              {feedback ? (
                <>
                  {/* Session Info */}
                  <div className="bg-white rounded-lg shadow-sm border p-6">
                    <h2 className="text-xl font-semibold flex items-center gap-2 text-green-600 mb-2">
                      <CheckIcon />
                      Analysis Complete
                    </h2>
                    <p className="text-gray-600">Session ID: {feedback.session_id}</p>
                  </div>

                  {/* Syntax Issues */}
                  {feedback.syntax_issues && feedback.syntax_issues.length > 0 && (
                    <div className="bg-white rounded-lg shadow-sm border">
                      <div className="p-6 border-b">
                        <h2 className="text-xl font-semibold flex items-center gap-2 text-red-600">
                          <AlertIcon />
                          Syntax Issues ({feedback.syntax_issues.length})
                        </h2>
                      </div>
                      <div className="p-6">
                        <div className="space-y-4">
                          {feedback.syntax_issues.map((issue, index) => (
                            <div key={index} className="border-l-4 border-red-500 pl-4 py-2">
                              <div className="flex items-center gap-2 mb-2">
                                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
                                  Line {issue.line}
                                </span>
                              </div>
                              <p className="text-sm text-gray-700 mb-2">{issue.message}</p>
                              <div className="bg-green-50 p-3 rounded-md">
                                <p className="text-sm font-medium text-green-800">Fix Suggestion:</p>
                                <p className="text-sm text-green-700">{issue.fix_suggestion}</p>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Logic Flaws */}
                  {feedback.logic_flaws && feedback.logic_flaws.length > 0 && (
                    <div className="bg-white rounded-lg shadow-sm border">
                      <div className="p-6 border-b">
                        <h2 className="text-xl font-semibold flex items-center gap-2 text-orange-600">
                          <BugIcon />
                          Logic Flaws ({feedback.logic_flaws.length})
                        </h2>
                      </div>
                      <div className="p-6">
                        <div className="space-y-4">
                          {feedback.logic_flaws.map((flaw, index) => (
                            <div key={index} className="border-l-4 border-orange-500 pl-4 py-2">
                              <p className="font-medium text-gray-900 mb-2">{flaw.context}</p>
                              <p className="text-sm text-gray-700">{flaw.explanation}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Optimizations */}
                  {feedback.optimizations && feedback.optimizations.length > 0 && (
                    <div className="bg-white rounded-lg shadow-sm border">
                      <div className="p-6 border-b">
                        <h2 className="text-xl font-semibold flex items-center gap-2 text-blue-600">
                          <ZapIcon />
                          Optimization Suggestions ({feedback.optimizations.length})
                        </h2>
                      </div>
                      <div className="p-6">
                        <div className="space-y-6">
                          {feedback.optimizations.map((opt, index) => (
                            <div key={index} className="space-y-3">
                              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div>
                                  <label className="block text-sm font-medium text-red-600 mb-2">Original Code:</label>
                                  <MonacoEditor
                                    value={opt.original_code}
                                    language={getMonacoLanguage(language)}
                                    height="150px"
                                    readOnly={true}
                                  />
                                </div>
                                <div>
                                  <label className="block text-sm font-medium text-green-600 mb-2">
                                    Optimized Code:
                                  </label>
                                  <MonacoEditor
                                    value={opt.optimized_code}
                                    language={getMonacoLanguage(language)}
                                    height="150px"
                                    readOnly={true}
                                  />
                                </div>
                              </div>
                              <div className="bg-blue-50 p-3 rounded-md">
                                <p className="text-sm font-medium text-blue-800">Rationale:</p>
                                <p className="text-sm text-blue-700">{opt.rationale}</p>
                              </div>
                              {index < feedback.optimizations.length - 1 && <hr className="my-4" />}
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Explanations */}
                  {feedback.explanations && feedback.explanations.length > 0 && (
                    <div className="bg-white rounded-lg shadow-sm border">
                      <div className="p-6 border-b">
                        <h2 className="text-xl font-semibold flex items-center gap-2 text-purple-600">
                          <BookIcon />
                          Educational Explanations ({feedback.explanations.length})
                        </h2>
                      </div>
                      <div className="p-6">
                        <div className="space-y-6">
                          {feedback.explanations.map((exp, index) => (
                            <div key={index} className="space-y-3">
                              <h4 className="font-semibold text-lg text-purple-800">{exp.concept}</h4>
                              <p className="text-gray-700">{exp.explanation}</p>
                              <div>
                                <label className="block text-sm font-medium text-purple-600 mb-2">Example Code:</label>
                                <MonacoEditor
                                  value={exp.example_code}
                                  language={getMonacoLanguage(language)}
                                  height="120px"
                                  readOnly={true}
                                />
                              </div>
                              {index < feedback.explanations.length - 1 && <hr className="my-4" />}
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Progressive Hints */}
                  {feedback.hint_trail && feedback.hint_trail.length > 0 && (
                    <div className="bg-white rounded-lg shadow-sm border">
                      <div className="p-6 border-b">
                        <h2 className="text-xl font-semibold flex items-center gap-2 text-indigo-600">
                          <LightbulbIcon />
                          Progressive Hints ({feedback.hint_trail.length})
                        </h2>
                      </div>
                      <div className="p-6">
                        <div className="space-y-4">
                          {feedback.hint_trail.map((hint, index) => (
                            <div key={index} className="flex gap-4 p-4 bg-indigo-50 rounded-lg">
                              <div className="flex-shrink-0">
                                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                                  Level {hint.level}
                                </span>
                              </div>
                              <div className="flex-1">
                                <p className="text-sm text-gray-700">{hint.hint}</p>
                                <p className="text-xs text-gray-500 mt-1">
                                  {new Date(hint.timestamp).toLocaleString()}
                                </p>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Final Fix */}
                  {feedback.final_fix && (
                    <div className="bg-white rounded-lg shadow-sm border">
                      <div className="p-6 border-b">
                        <h2 className="text-xl font-semibold flex items-center gap-2 text-green-600">
                          <CheckIcon />
                          Final Fixed Code
                        </h2>
                      </div>
                      <div className="p-6">
                        <MonacoEditor
                          value={feedback.final_fix}
                          language={getMonacoLanguage(language)}
                          height="300px"
                          readOnly={true}
                        />
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="bg-white rounded-lg shadow-sm border">
                  <div className="text-center py-12">
                    <LargeCodeIcon />
                    <p className="text-gray-500">No analysis results yet. Submit your code first!</p>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* History Tab */}
          {activeTab === "history" && (
            <div className="space-y-6">
              <div className="bg-white rounded-lg shadow-sm border">
                <div className="p-6 border-b">
                  <h2 className="text-xl font-semibold flex items-center gap-2 mb-2">
                    <UserIcon />
                    Submission History for {learnerId}
                  </h2>
                  <p className="text-gray-600">View your past code submissions and analysis results</p>
                </div>
                <div className="p-6">
                  <button
                    onClick={fetchHistory}
                    disabled={loading}
                    className="mb-4 bg-blue-600 text-white py-2 px-4 rounded-md font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 transition-colors"
                  >
                    {loading ? (
                      <>
                        <LoaderIcon />
                        Loading...
                      </>
                    ) : (
                      <>
                        <HistoryIcon />
                        Refresh History
                      </>
                    )}
                  </button>

                  {history.length > 0 ? (
                    <div className="max-h-96 overflow-y-auto">
                      <div className="space-y-3">
                        {history.map((item, index) => (
                          <div
                            key={index}
                            className="flex items-center justify-between p-4 border rounded-lg hover:bg-gray-50 transition-colors"
                          >
                            <div className="flex items-center gap-4">
                              <div className="flex-shrink-0">
                                <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                                  {item.language}
                                </span>
                              </div>
                              <div>
                                <p className="font-medium text-sm">Session: {item.session_id.slice(0, 8)}...</p>
                                <p className="text-xs text-gray-500 flex items-center gap-1">
                                  <ClockIcon />
                                  {new Date(item.timestamp).toLocaleString()}
                                </p>
                              </div>
                            </div>
                            <div className="flex items-center gap-2">
                              <span
                                className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                  item.issue_count > 0 ? "bg-red-100 text-red-800" : "bg-green-100 text-green-800"
                                }`}
                              >
                                {item.issue_count} issues
                              </span>
                              <button
                                onClick={() => fetchHints(item.session_id)}
                                className="px-3 py-1 text-sm border border-gray-300 rounded-md hover:bg-gray-50 transition-colors"
                              >
                                View Hints
                              </button>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <LargeHistoryIcon />
                      <p className="text-gray-500">No submission history found</p>
                    </div>
                  )}
                </div>
              </div>

              {/* Hints Modal/Card */}
              {hints && (
                <div className="bg-white rounded-lg shadow-sm border">
                  <div className="p-6 border-b">
                    <h2 className="text-xl font-semibold flex items-center gap-2">
                      <LightbulbIcon />
                      Session Hints & Fix
                    </h2>
                  </div>
                  <div className="p-6 space-y-4">
                    {hints.hint_trail && hints.hint_trail.length > 0 && (
                      <div className="space-y-3">
                        <h4 className="font-medium">Progressive Hints:</h4>
                        {hints.hint_trail.map((hint, index) => (
                          <div key={index} className="flex gap-3 p-3 bg-yellow-50 rounded-lg">
                            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
                              Level {hint.level}
                            </span>
                            <p className="text-sm">{hint.hint}</p>
                          </div>
                        ))}
                      </div>
                    )}

                    {hints.final_fix && (
                      <div>
                        <h4 className="font-medium mb-2">Final Fix:</h4>
                        <MonacoEditor
                          value={hints.final_fix}
                          language={getMonacoLanguage(language)}
                          height="200px"
                          readOnly={true}
                        />
                      </div>
                    )}

                    <button
                      onClick={() => setHints(null)}
                      className="px-4 py-2 border border-gray-300 rounded-md hover:bg-gray-50 transition-colors"
                    >
                      Close
                    </button>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
