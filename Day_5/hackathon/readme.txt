

### ✅ **Python Version**

* **Python ≥ 3.9** (due to `typing`, `dataclasses`, and modern libraries used)

---

### 📦 **Python Package Requirements**

You can create a `requirements.txt` with the following content:

```txt
streamlit>=1.27.0
pandas
numpy
plotly
google-generativeai>=0.3.2
python-dotenv
typing-extensions
tqdm
regex
protobuf
markdown2
```

Also, install standard packages for document I/O and formatting:

```txt
PyPDF2>=3.0.0
python-docx
```

---

### 🗝️ **Secrets Configuration**

Add your **Gemini API key** to `secrets.toml`:

Create a file at:

```bash
~/.streamlit/secrets.toml
```

Add:

```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
```

---

### 📁 **Folder Structure (Recommended)**

```
project/
│
├── app.py                # Main script
├── requirements.txt
└── .streamlit/
    └── secrets.toml        # Gemini API key
```

---

### 🧠 **System Capabilities Used**

* Streamlit frontend with CSS enhancements
* AI agents:

  * `GeminiDataGenerator`
  * `CognitiveStudyPlannerAgent`
  * `LifeEventInterpreterAgent`
  * `FatigueWorkloadMonitorAgent`
  * `RAGProductivityResearchAgent`
* Advanced visualizations with Plotly
* Dynamic multi-tab layout
* Local session state management for agents and chat
* Data models with `@dataclass`

---

### 🚀 **To Run the App**

After installing dependencies:

```bash
streamlit run app.py
```


