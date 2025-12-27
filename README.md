---
title: LLM Council
emoji: 🏛️
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
license: mit
---
# 🏛️ LLM Council System

A production-ready multi-agent AI system where 3 agents independently generate answers and 2 judges evaluate them using a structured rubric. Features safety gating, persistent audit logs, and advanced reflection capabilities.

**Built for Aonxi Interview Challenge** | [Aonxi Website](https://www.aonxi.app)

---

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Project Structure](#project-structure)
- [Design Decisions](#design-decisions)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Contributing](#contributing)

---

## ✨ Features

### Core Features
- **3 Independent Agents**: Different personas generate diverse answers
- **2 Comparative Judges**: Evaluate using 4-dimension rubric (accuracy, completeness, clarity, reasoning)
- **Safety Gating**: Pre-filter dangerous/inappropriate questions
- **Persistent Audit Log**: Every decision saved to disk (JSON + JSONL)
- **Structured Output**: DecisionObject with confidence, risks, citations
- **Rich CLI**: Beautiful terminal interface with progress indicators

### Advanced Features (Optional)
- **Chain-of-Thought Reasoning**: Step-by-step thinking process
- **Self-Reflection**: Agents critique and revise their own answers
- **LangGraph Orchestration**: Graph-based multi-agent workflow
- **Iterative Improvement**: Automatic revision loops

---

## 🏗️ Architecture

```
                    ┌─────────────┐
                    │   Question  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Safety Gate │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
         ┌────▼───┐   ┌───▼────┐   ┌──▼─────┐
         │Agent 1 │   │Agent 2 │   │Agent 3 │
         │Analyst │   │Creative│   │Pragmatic│
         └────┬───┘   └───┬────┘   └──┬─────┘
              │           │            │
              └───────────┼────────────┘
                          │
                   ┌──────▼──────┐
                   │  Reflection │ (Optional)
                   └──────┬──────┘
                          │
              ┌───────────┼───────────┐
              │                       │
         ┌────▼────┐             ┌───▼─────┐
         │ Judge 1 │             │ Judge 2 │
         └────┬────┘             └───┬─────┘
              │                      │
              └──────────┬───────────┘
                         │
                  ┌──────▼──────┐
                  │  Synthesize │
                  └──────┬──────┘
                         │
                  ┌──────▼──────┐
                  │   Decision  │
                  │   + Audit   │
                  └─────────────┘
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/Hima-D/llm-council.git
cd llm-council

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your API key
export OPENAI_API_KEY=sk-proj-xxxxx
```

### Run Demo

```bash
# Basic version
python council.py


# Interactive examples
python example.py
```

---

## 💡 Usage Examples

### Example 1: Basic Usage

```python
from council import LLMCouncil

# Initialize
council = LLMCouncil(api_key="your_key")

# Ask a question
decision = council.deliberate("What is machine learning?")

# Access results
print(f"Answer: {decision.final_answer}")
print(f"Confidence: {decision.confidence:.1%}")
print(f"Winner: {decision.metadata['winner']}")
```

### Example 2: Advanced with Reflection

```python
from council_advanced import AdvancedLLMCouncil

# Initialize with reflection
council = AdvancedLLMCouncil(api_key="your_key", model="gpt-4o")

# Deliberate with up to 2 reflection rounds
decision = council.deliberate(
    question="How should startups balance growth vs profitability?",
    max_reflection_rounds=2
)

# Access thinking process
for response in decision.agent_responses:
    print(f"{response.agent_id}:")
    print(f"  Analysis: {response.thinking_process.analysis}")
    print(f"  Confidence: {response.confidence:.1%}")
```

### Example 3: Batch Processing

```python
questions = [
    "What is Python?",
    "Best practices for API design?",
    "Cloud vs on-premise?"
]

results = []
for q in questions:
    decision = council.deliberate(q)
    results.append({
        "question": q,
        "confidence": decision.confidence,
        "winner": decision.metadata["winner"]
    })
```

### Example 4: Export to JSON

```python
import json

decision = council.deliberate("Your question")

# Export full decision
with open("decision.json", "w") as f:
    json.dump(decision.model_dump(mode='json'), f, indent=2, default=str)
```

---

## 📁 Project Structure

```
llm-council/
├── council.py              # Basic OpenAI council
├── council_advanced.py     # Advanced with LangChain/LangGraph
├── example.py              # Interactive usage examples
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .env.example           # API key template
├── .gitignore             # Git ignore rules
├── audit_logs/            # Persistent decision logs
│   ├── decision_YYYYMMDD_HHMMSS.json
│   └── master_log.jsonl
└── tests/                 # Unit tests (optional)
    └── test_council.py
```

---

## 🎯 Design Decisions

### Decision 1: Manual Judge Rubric ⭐

**What I did NOT automate:** The judge evaluation rubric dimensions.

**Rubric Dimensions (Hardcoded):**
- Accuracy (factual correctness)
- Completeness (coverage)
- Clarity (understandability)
- Reasoning (logic quality)

**Why Not Automated:**

1. **Domain Specificity**: Different use cases need different criteria
   - Medical: safety, evidence quality
   - Creative: originality, engagement
   - Technical: correctness, efficiency

2. **Transparency & Auditability**: 
   - Hardcoded rubrics are debuggable
   - Auto-generated rubrics can drift
   - Regulatory compliance requires documented criteria

3. **Human Oversight**: 
   - Humans define "what good looks like"
   - System executes, humans set standards
   - Prevents optimization for wrong metrics

4. **Consistency**: 
   - Same criteria across all questions
   - Comparable results over time
   - Easier A/B testing

**Alternative Approach (Rejected):**
```python
# Could auto-generate rubric per question
rubric = llm.generate_rubric(question, domain)
# But this makes results unpredictable
```

**Principle:** *Automate execution, preserve human judgment on values.*

### Decision 2: Fixed Reflection Rounds

**What:** Maximum reflection rounds is a parameter (default: 2), not fully adaptive.

**Why:**
- Cost control (each round = 3+ API calls)
- Predictable runtime
- Diminishing returns after 2-3 rounds
- Prevents infinite loops

---

## 📊 Decision Object Structure

```typescript
{
  question: string,
  final_answer: string,
  confidence: float,           // 0.0 - 1.0
  risks: string[],
  citations: string[],
  safety_status: "safe" | "caution" | "blocked",
  
  agent_responses: [
    {
      agent_id: string,
      response: string,
      thinking_process: {      // Advanced version only
        initial_thoughts: string,
        analysis: string,
        potential_issues: string[],
        reasoning_steps: string[]
      },
      confidence: float,
      timestamp: datetime
    }
  ],
  
  judge_evaluations: [
    {
      judge_id: string,
      scores: {
        agent_id: {
          accuracy: float,     // 0-10
          completeness: float,
          clarity: float,
          reasoning: float,
          overall: float
        }
      },
      winner: string,
      reasoning: string,
      timestamp: datetime
    }
  ],
  
  reflection_rounds: int,      // Advanced version only
  metadata: {
    winner: string,
    votes: { agent_id: count },
    agent_count: int,
    judge_count: int
  },
  timestamp: datetime
}
```

---

## 🔧 API Reference

### LLMCouncil (Basic)

```python
council = LLMCouncil(api_key: str)
```

**Methods:**
- `deliberate(question: str) -> DecisionObject`
- `print_decision(decision: DecisionObject) -> None`

### AdvancedLLMCouncil

```python
council = AdvancedLLMCouncil(
    api_key: str,
    model: str = "gpt-4o"  # or "gpt-4-turbo", "gpt-3.5-turbo"
)
```

**Methods:**
- `deliberate(question: str, max_reflection_rounds: int = 2) -> DecisionObject`
- `print_decision(decision: DecisionObject) -> None`

### Safety Gate

```python
from council import SafetyGate

level, risks = SafetyGate.check(question)
# Returns: (SafetyLevel.SAFE|CAUTION|BLOCKED, List[str])
```

### Audit Log

```python
from council import AuditLog

log = AuditLog(log_dir="audit_logs")
log.log_decision(decision)
recent = log.get_recent_decisions(n=10)
```

---

## 🚢 Deployment

### Deploy to GitHub

```bash
git init
git add .
git commit -m "Initial commit: LLM Council system"
git remote add origin https://github.com/yourusername/llm-council.git
git push -u origin main
```

### Deploy to Hugging Face Spaces

1. Create a `app.py` with Gradio interface:

```python
import gradio as gr
from council import LLMCouncil
import os

council = LLMCouncil(os.getenv("OPENAI_API_KEY"))

def deliberate_wrapper(question):
    decision = council.deliberate(question)
    return {
        "answer": decision.final_answer,
        "confidence": f"{decision.confidence:.1%}",
        "winner": decision.metadata["winner"]
    }

demo = gr.Interface(
    fn=deliberate_wrapper,
    inputs=gr.Textbox(label="Question"),
    outputs=gr.JSON(label="Decision"),
    title="LLM Council"
)

demo.launch()
```

2. Push to Hugging Face:

```bash
git push https://huggingface.co/spaces/username/llm-council
```

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "council.py"]
```

---

## 📈 Performance & Costs

### Basic Version
- **Time per question**: 30-45 seconds
- **Cost per question**: ~$0.05-0.10 (gpt-4o)
- **API calls**: 5 (3 agents + 2 judges)

### Advanced Version (with reflection)
- **Time per question**: 60-120 seconds
- **Cost per question**: ~$0.15-0.40 (gpt-4o)
- **API calls**: 5-15 (depends on reflection rounds)

### Cost Optimization

```python
# Use cheaper model
council = AdvancedLLMCouncil(api_key=key, model="gpt-3.5-turbo")

# Reduce reflection rounds
decision = council.deliberate(question, max_reflection_rounds=1)

# Reduce max_tokens
# Edit in council.py: max_tokens=500 (instead of 1000)
```

---

## 🧪 Testing

```bash
# Run basic test
python -c "
from council import LLMCouncil
import os
council = LLMCouncil(os.getenv('OPENAI_API_KEY'))
decision = council.deliberate('What is Python?')
assert decision.confidence > 0
print('✅ Test passed!')
"

# Run safety tests
python example.py
# Select option 4: Safety Gate Testing
```

---

## 🤝 Contributing

We welcome contributions! Areas for improvement:

1. **Memory/Context**: Store past decisions for context
2. **Multi-turn**: Support follow-up questions
3. **External Tools**: Add web search, calculators
4. **Adaptive Rubrics**: Domain-specific evaluation
5. **Parallel Execution**: Run agents concurrently
6. **Human-in-the-Loop**: Manual intervention option

---

## 📄 License

MIT License - see LICENSE file

---

## 👤 Author

**Himanshu**  


## 🙏 Acknowledgments

- Aonxi team for the challenge
- OpenAI for GPT-4 API
- LangChain/LangGraph for orchestration
- Anthropic for Claude (alternate version)

---

## 📚 Further Reading

- [Prompt Engineering Guide](https://www.promptingguide.ai/)
- [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)
- [ReAct: Reasoning + Acting](https://arxiv.org/abs/2210.03629)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Constitutional AI Paper](https://arxiv.org/abs/2212.08073)

---

## 📞 Support

For questions or issues:
1. Check [Issues](https://github.com/e/llm-council/issues)
2. Read the [FAQ](#faq)
3. Contact: origin@aonxi.com

---

## FAQ

**Q: Why OpenAI instead of Anthropic?**  
A: You requested it! We have an Anthropic version too (see `council_anthropic.py`).

**Q: How accurate are the decisions?**  
A: With gpt-4o and 2 judges, typically 85-95% confidence. Lower for ambiguous questions.

**Q: Can I add more agents/judges?**  
A: Yes! Edit the `agents` and `judges` lists in the `__init__` method.

**Q: What about rate limits?**  
A: OpenAI free tier: 3 RPM, 200 RPD. Paid tier: much higher. Add retry logic if needed.

**Q: Why not just use one LLM call?**  
A: Multi-agent consensus reduces bias, catches errors, provides diverse perspectives.

---

**⭐ If this helped, please star the repo!**