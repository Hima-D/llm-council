import json
import os
from datetime import datetime
from typing import List, Dict, Any, TypedDict, Annotated
from enum import Enum
from pathlib import Path
import operator

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.tree import Tree

console = Console()


# ============================================================================
# Data Models
# ============================================================================

class SafetyLevel(str, Enum):
    SAFE = "safe"
    CAUTION = "caution"
    BLOCKED = "blocked"


class ThinkingProcess(BaseModel):
    """Structured thinking with chain-of-thought"""
    initial_thoughts: str = Field(description="First impressions and key considerations")
    analysis: str = Field(description="Deep analysis of the question")
    potential_issues: List[str] = Field(description="Identified problems or gaps")
    reasoning_steps: List[str] = Field(description="Step-by-step logical reasoning")


class AgentResponse(BaseModel):
    agent_id: str
    response: str
    thinking_process: ThinkingProcess
    confidence: float = Field(ge=0, le=1)
    sources: List[str] = Field(default_factory=list)
    timestamp: datetime


class ReflectionResult(BaseModel):
    """Agent's self-critique of their answer"""
    strengths: List[str] = Field(description="What's good about this answer")
    weaknesses: List[str] = Field(description="What could be improved")
    revision_needed: bool = Field(description="Whether revision is needed")
    improvement_suggestions: List[str] = Field(description="How to improve")


class JudgeScores(BaseModel):
    """Structured scoring from judge"""
    accuracy: float = Field(ge=0, le=10)
    completeness: float = Field(ge=0, le=10)
    clarity: float = Field(ge=0, le=10)
    reasoning: float = Field(ge=0, le=10)
    overall: float = Field(ge=0, le=10)


class JudgeEvaluation(BaseModel):
    judge_id: str
    scores: Dict[str, JudgeScores]  # agent_id -> scores
    winner: str
    reasoning: str
    comparative_analysis: str = Field(description="Detailed comparison")
    timestamp: datetime


class DecisionObject(BaseModel):
    question: str
    final_answer: str
    confidence: float = Field(ge=0, le=1)
    risks: List[str]
    citations: List[str]
    safety_status: SafetyLevel
    agent_responses: List[AgentResponse]
    judge_evaluations: List[JudgeEvaluation]
    reflection_rounds: int = Field(default=0)
    metadata: Dict[str, Any]
    timestamp: datetime


# ============================================================================
# LangGraph State
# ============================================================================

class CouncilState(TypedDict):
    """State shared across all nodes in the graph"""
    question: str
    safety_status: SafetyLevel
    safety_risks: List[str]
    agent_responses: Annotated[List[AgentResponse], operator.add]
    reflections: Annotated[List[Dict], operator.add]
    judge_evaluations: Annotated[List[JudgeEvaluation], operator.add]
    final_decision: Dict[str, Any]
    iteration: int
    max_iterations: int


# ============================================================================
# Safety Gate
# ============================================================================

class SafetyGate:
    """Enhanced safety checking"""
    
    BLOCKED_PATTERNS = [
        "how to make weapons", "how to harm", "illegal activities",
        "exploit vulnerabilities", "bypass security", "create malware"
    ]
    
    CAUTION_PATTERNS = [
        "medical advice", "legal advice", "financial advice",
        "privacy concerns", "health diagnosis", "investment"
    ]
    
    @classmethod
    def check(cls, question: str) -> tuple:
        q_lower = question.lower()
        risks = []
        
        for pattern in cls.BLOCKED_PATTERNS:
            if pattern in q_lower:
                risks.append(f"Blocked: {pattern}")
                return SafetyLevel.BLOCKED, risks
        
        for pattern in cls.CAUTION_PATTERNS:
            if pattern in q_lower:
                risks.append(f"Caution: {pattern}")
        
        return (SafetyLevel.CAUTION, risks) if risks else (SafetyLevel.SAFE, [])


# ============================================================================
# Advanced Agents with Reflection
# ============================================================================

class ReflectiveAgent:
    """Agent with chain-of-thought reasoning and self-reflection"""
    
    def __init__(self, agent_id: str, persona: str, specialty: str, llm: ChatOpenAI):
        self.agent_id = agent_id
        self.persona = persona
        self.specialty = specialty
        self.llm = llm
        
        # Thinking prompt with structured output
        self.thinking_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are {persona}, specializing in {specialty}.

You use structured thinking to answer questions thoroughly:
1. Initial Thoughts: What strikes you first about this question?
2. Deep Analysis: Break down the core components
3. Potential Issues: What could go wrong or be misunderstood?
4. Reasoning Steps: Walk through your logic step-by-step

Be thorough but concise. Focus on your specialty."""),
            ("human", "Question: {question}\n\nProvide your structured thinking:")
        ])
        
        # Answer generation prompt
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are {persona}, specializing in {specialty}.

Based on your thinking process, provide a clear, well-reasoned answer.

Include:
- Direct answer to the question
- Key reasoning points
- Any relevant sources or evidence
- Caveats or limitations

Be authoritative but acknowledge uncertainty where appropriate."""),
            ("human", """Question: {question}

Your Thinking Process:
{thinking}

Now provide your final answer:""")
        ])
        
        # Reflection prompt for self-critique
        self.reflection_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a critical evaluator reviewing your own work.

Analyze this answer objectively:
- What are its strengths?
- What are its weaknesses?
- Does it need revision?
- How could it be improved?

Be honest and constructive."""),
            ("human", """Question: {question}

Your Answer:
{answer}

Provide critical reflection:""")
        ])
    
    def think(self, question: str) -> ThinkingProcess:
        """Chain-of-thought reasoning"""
        response = self.llm.invoke(
            self.thinking_prompt.format_messages(
                persona=self.persona,
                specialty=self.specialty,
                question=question
            )
        )
        
        content = response.content
        
        # Parse structured thinking (simple parsing)
        thinking = ThinkingProcess(
            initial_thoughts=self._extract_section(content, "Initial Thoughts", "Deep Analysis"),
            analysis=self._extract_section(content, "Deep Analysis", "Potential Issues"),
            potential_issues=self._extract_list(content, "Potential Issues", "Reasoning Steps"),
            reasoning_steps=self._extract_list(content, "Reasoning Steps")
        )
        
        return thinking
    
    def generate_answer(self, question: str, thinking: ThinkingProcess) -> str:
        """Generate answer based on thinking"""
        response = self.llm.invoke(
            self.answer_prompt.format_messages(
                persona=self.persona,
                specialty=self.specialty,
                question=question,
                thinking=thinking.model_dump_json(indent=2)
            )
        )
        
        return response.content
    
    def reflect(self, question: str, answer: str) -> ReflectionResult:
        """Self-critique and reflection"""
        response = self.llm.invoke(
            self.reflection_prompt.format_messages(
                question=question,
                answer=answer
            )
        )
        
        content = response.content
        
        # Parse reflection
        reflection = ReflectionResult(
            strengths=self._extract_list(content, "Strengths", "Weaknesses"),
            weaknesses=self._extract_list(content, "Weaknesses", "Revision"),
            revision_needed="yes" in content.lower() and "revision" in content.lower(),
            improvement_suggestions=self._extract_list(content, "Improvements")
        )
        
        return reflection
    
    def revise(self, question: str, original_answer: str, reflection: ReflectionResult) -> str:
        """Revise answer based on reflection"""
        revision_prompt = f"""Question: {question}

Original Answer:
{original_answer}

Issues Identified:
{chr(10).join(f"- {w}" for w in reflection.weaknesses)}

Improvement Suggestions:
{chr(10).join(f"- {s}" for s in reflection.improvement_suggestions)}

Provide a revised, improved answer:"""
        
        response = self.llm.invoke([HumanMessage(content=revision_prompt)])
        return response.content
    
    def _extract_section(self, text: str, start_marker: str, end_marker: str = None) -> str:
        """Extract text between markers"""
        try:
            start = text.lower().find(start_marker.lower())
            if start == -1:
                return ""
            start = text.find(":", start) + 1
            
            if end_marker:
                end = text.lower().find(end_marker.lower(), start)
                if end != -1:
                    return text[start:end].strip()
            
            return text[start:].strip()
        except:
            return ""
    
    def _extract_list(self, text: str, section: str, next_section: str = None) -> List[str]:
        """Extract bullet points from section"""
        section_text = self._extract_section(text, section, next_section)
        lines = section_text.split("\n")
        items = []
        for line in lines:
            line = line.strip()
            if line.startswith("-") or line.startswith("•") or line.startswith("*"):
                items.append(line[1:].strip())
        return items if items else ["No items found"]


# ============================================================================
# Advanced Judge
# ============================================================================

class AdvancedJudge:
    """Judge with structured evaluation rubric"""
    
    def __init__(self, judge_id: str, llm: ChatOpenAI):
        self.judge_id = judge_id
        self.llm = llm
        
        self.evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert evaluator comparing multiple responses.

Evaluation Rubric (score 0-10 for each):
1. Accuracy: Factual correctness and precision
2. Completeness: Coverage of all key aspects
3. Clarity: Clear, understandable explanation
4. Reasoning: Quality of logical reasoning and evidence
5. Overall: Holistic assessment

For each response, provide:
- Scores for all dimensions
- Detailed comparative analysis
- Clear winner selection with reasoning

Be objective and thorough."""),
            ("human", """Question: {question}

{responses}

Evaluate all responses and return your assessment as JSON:
{{
  "scores": {{
    "agent_1": {{"accuracy": X, "completeness": X, "clarity": X, "reasoning": X, "overall": X}},
    "agent_2": {{...}},
    "agent_3": {{...}}
  }},
  "winner": "agent_X",
  "reasoning": "detailed explanation",
  "comparative_analysis": "how responses compare"
}}""")
        ])
    
    def evaluate(self, question: str, responses: List[AgentResponse]) -> JudgeEvaluation:
        """Structured evaluation with rubric"""
        
        responses_text = "\n\n".join([
            f"=== {r.agent_id.upper()} ===\n{r.response}\n\nConfidence: {r.confidence:.1%}\nThinking: {r.thinking_process.analysis[:200]}..."
            for r in responses
        ])
        
        response = self.llm.invoke(
            self.evaluation_prompt.format_messages(
                question=question,
                responses=responses_text
            )
        )
        
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            result = json.loads(content.strip())
            
            # Convert to structured scores
            structured_scores = {}
            for agent_id, scores in result["scores"].items():
                structured_scores[agent_id] = JudgeScores(**scores)
            
            return JudgeEvaluation(
                judge_id=self.judge_id,
                scores=structured_scores,
                winner=result["winner"],
                reasoning=result["reasoning"],
                comparative_analysis=result.get("comparative_analysis", ""),
                timestamp=datetime.now()
            )
        
        except Exception as e:
            console.print(f"[red]Judge {self.judge_id} error: {e}[/red]")
            # Fallback
            return JudgeEvaluation(
                judge_id=self.judge_id,
                scores={r.agent_id: JudgeScores(accuracy=5, completeness=5, clarity=5, reasoning=5, overall=5) for r in responses},
                winner=responses[0].agent_id,
                reasoning=f"Fallback due to error: {str(e)}",
                comparative_analysis="Error in evaluation",
                timestamp=datetime.now()
            )


# ============================================================================
# LangGraph Workflow
# ============================================================================

def create_council_graph(llm: ChatOpenAI) -> StateGraph:
    """Build the council deliberation graph"""
    
    # Initialize agents with different specialties
    agents = [
        ReflectiveAgent("agent_1", "a meticulous analyst", "data analysis and logical reasoning", llm),
        ReflectiveAgent("agent_2", "an innovative strategist", "creative problem-solving and systems thinking", llm),
        ReflectiveAgent("agent_3", "a practical implementer", "pragmatic solutions and real-world applications", llm)
    ]
    
    judges = [
        AdvancedJudge("judge_1", llm),
        AdvancedJudge("judge_2", llm)
    ]
    
    # Define nodes
    def safety_check(state: CouncilState) -> CouncilState:
        """Check question safety"""
        console.print("[bold]🛡️  Safety Gate Check...[/bold]")
        safety_level, risks = SafetyGate.check(state["question"])
        state["safety_status"] = safety_level
        state["safety_risks"] = risks
        
        if safety_level == SafetyLevel.BLOCKED:
            console.print("[bold red]❌ Question blocked[/bold red]")
        elif safety_level == SafetyLevel.CAUTION:
            console.print("[yellow]⚠️  Proceeding with caution[/yellow]")
        else:
            console.print("[green]✓ Safe to proceed[/green]")
        
        return state
    
    def agents_think(state: CouncilState) -> CouncilState:
        """Agents generate answers with thinking"""
        console.print("\n[bold]🤔 Step 1: Agents Thinking...[/bold]")
        
        responses = []
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            for agent in agents:
                task = progress.add_task(f"{agent.agent_id} thinking...", total=None)
                
                # Chain-of-thought reasoning
                thinking = agent.think(state["question"])
                
                # Generate answer
                answer = agent.generate_answer(state["question"], thinking)
                
                # Calculate confidence (simple heuristic)
                confidence = 0.8 if len(thinking.potential_issues) <= 2 else 0.6
                
                response = AgentResponse(
                    agent_id=agent.agent_id,
                    response=answer,
                    thinking_process=thinking,
                    confidence=confidence,
                    sources=[],
                    timestamp=datetime.now()
                )
                
                responses.append(response)
                progress.remove_task(task)
                console.print(f"✓ {agent.agent_id} completed (confidence: {confidence:.1%})")
        
        state["agent_responses"] = responses
        return state
    
    def agents_reflect(state: CouncilState) -> CouncilState:
        """Agents reflect on their answers"""
        console.print("\n[bold]🔍 Step 2: Self-Reflection...[/bold]")
        
        reflections = []
        revised_responses = []
        
        for i, agent in enumerate(agents):
            response = state["agent_responses"][i]
            
            # Self-critique
            reflection = agent.reflect(state["question"], response.response)
            reflections.append({
                "agent_id": agent.agent_id,
                "reflection": reflection.model_dump()
            })
            
            console.print(f"✓ {agent.agent_id} reflection: {'Revision needed' if reflection.revision_needed else 'Satisfied'}")
            
            # Revise if needed
            if reflection.revision_needed and state["iteration"] < state["max_iterations"]:
                console.print(f"  → Revising {agent.agent_id}...")
                revised_answer = agent.revise(state["question"], response.response, reflection)
                response.response = revised_answer
                response.confidence = min(response.confidence + 0.1, 1.0)
            
            revised_responses.append(response)
        
        state["agent_responses"] = revised_responses
        state["reflections"] = reflections
        state["iteration"] += 1
        
        return state
    
    def judges_evaluate(state: CouncilState) -> CouncilState:
        """Judges evaluate responses"""
        console.print("\n[bold]⚖️  Step 3: Judges Evaluating...[/bold]")
        
        evaluations = []
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            for judge in judges:
                task = progress.add_task(f"{judge.judge_id} evaluating...", total=None)
                
                evaluation = judge.evaluate(state["question"], state["agent_responses"])
                evaluations.append(evaluation)
                
                progress.remove_task(task)
                console.print(f"✓ {judge.judge_id} selected {evaluation.winner}")
        
        state["judge_evaluations"] = evaluations
        return state
    
    def synthesize_decision(state: CouncilState) -> CouncilState:
        """Synthesize final decision"""
        console.print("\n[bold]📊 Step 4: Synthesizing Decision...[/bold]")
        
        # Count votes
        votes = {}
        for eval in state["judge_evaluations"]:
            votes[eval.winner] = votes.get(eval.winner, 0) + 1
        
        winner_id = max(votes, key=votes.get)
        winner_response = next(r for r in state["agent_responses"] if r.agent_id == winner_id)
        
        confidence = votes[winner_id] / len(state["judge_evaluations"])
        
        # Extract citations
        citations = []
        for r in state["agent_responses"]:
            if r.sources:
                citations.extend(r.sources)
        
        # Aggregate risks
        risks = state["safety_risks"].copy()
        if confidence < 0.7:
            risks.append(f"Low consensus (confidence: {confidence:.1%})")
        
        state["final_decision"] = {
            "winner": winner_id,
            "votes": votes,
            "confidence": confidence,
            "answer": winner_response.response,
            "citations": citations if citations else ["No explicit citations"],
            "risks": risks if risks else ["No significant risks"]
        }
        
        console.print(f"✓ Winner: {winner_id} (confidence: {confidence:.1%})")
        
        return state
    
    def should_continue_reflection(state: CouncilState) -> str:
        """Decide whether to continue reflection loop"""
        if state["iteration"] >= state["max_iterations"]:
            return "evaluate"
        
        # Check if any agent needs revision
        if state.get("reflections"):
            needs_revision = any(
                r["reflection"]["revision_needed"] 
                for r in state["reflections"]
            )
            if needs_revision:
                return "reflect"
        
        return "evaluate"
    
    # Build graph
    workflow = StateGraph(CouncilState)
    
    # Add nodes
    workflow.add_node("safety", safety_check)
    workflow.add_node("think", agents_think)
    workflow.add_node("reflect", agents_reflect)
    workflow.add_node("evaluate", judges_evaluate)
    workflow.add_node("synthesize", synthesize_decision)
    
    # Add edges
    workflow.set_entry_point("safety")
    workflow.add_edge("safety", "think")
    workflow.add_edge("think", "reflect")
    workflow.add_conditional_edges(
        "reflect",
        should_continue_reflection,
        {
            "reflect": "reflect",
            "evaluate": "evaluate"
        }
    )
    workflow.add_edge("evaluate", "synthesize")
    workflow.add_edge("synthesize", END)
    
    return workflow.compile()


# ============================================================================
# Main Council Class
# ============================================================================

class AdvancedLLMCouncil:
    """Main orchestrator using LangGraph"""
    
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=0.7
        )
        self.graph = create_council_graph(self.llm)
        self.audit_log = AuditLog()
    
    def deliberate(self, question: str, max_reflection_rounds: int = 2) -> DecisionObject:
        """Run council deliberation with reflection"""
        
        console.print(Panel.fit(
            f"[bold cyan]Question:[/bold cyan] {question}",
            title="🏛️  Advanced LLM Council"
        ))
        
        # Initialize state
        initial_state: CouncilState = {
            "question": question,
            "safety_status": SafetyLevel.SAFE,
            "safety_risks": [],
            "agent_responses": [],
            "reflections": [],
            "judge_evaluations": [],
            "final_decision": {},
            "iteration": 0,
            "max_iterations": max_reflection_rounds
        }
        
        # Run graph
        final_state = self.graph.invoke(initial_state)
        
        # Check if blocked
        if final_state["safety_status"] == SafetyLevel.BLOCKED:
            decision = DecisionObject(
                question=question,
                final_answer="Question blocked due to safety concerns.",
                confidence=0.0,
                risks=final_state["safety_risks"],
                citations=[],
                safety_status=final_state["safety_status"],
                agent_responses=[],
                judge_evaluations=[],
                reflection_rounds=0,
                metadata={"blocked": True},
                timestamp=datetime.now()
            )
            self.audit_log.log_decision(decision)
            return decision
        
        # Build decision object
        final = final_state["final_decision"]
        
        decision = DecisionObject(
            question=question,
            final_answer=final["answer"],
            confidence=final["confidence"],
            risks=final["risks"],
            citations=final["citations"],
            safety_status=final_state["safety_status"],
            agent_responses=final_state["agent_responses"],
            judge_evaluations=final_state["judge_evaluations"],
            reflection_rounds=final_state["iteration"],
            metadata={
                "winner": final["winner"],
                "votes": final["votes"],
                "reflections": len(final_state["reflections"])
            },
            timestamp=datetime.now()
        )
        
        self.audit_log.log_decision(decision)
        return decision
    
    def print_decision(self, decision: DecisionObject):
        """Pretty print with reflection details"""
        
        console.print("\n" + "="*80)
        console.print(Panel.fit(
            f"[bold green]Final Answer[/bold green]\n\n{decision.final_answer}",
            title=f"Decision (Confidence: {decision.confidence:.1%})"
        ))
        
        console.print(f"\n[bold]Safety:[/bold] {decision.safety_status.value.upper()}")
        console.print(f"[bold]Confidence:[/bold] {decision.confidence:.1%}")
        console.print(f"[bold]Reflection Rounds:[/bold] {decision.reflection_rounds}")
        
        console.print(f"\n[bold]Risks:[/bold]")
        for risk in decision.risks:
            console.print(f"  • {risk}")
        
        console.print(f"\n[bold]Agent Thinking (Sample):[/bold]")
        if decision.agent_responses:
            sample = decision.agent_responses[0]
            tree = Tree(f"[cyan]{sample.agent_id}[/cyan]")
            tree.add(f"Analysis: {sample.thinking_process.analysis[:100]}...")
            tree.add(f"Confidence: {sample.confidence:.1%}")
            console.print(tree)
        
        console.print(f"\n[bold]Metadata:[/bold]")
        console.print(f"  Winner: {decision.metadata['winner']}")
        console.print(f"  Votes: {decision.metadata['votes']}")
        
        console.print("\n" + "="*80)


# ============================================================================
# Audit Log
# ============================================================================

class AuditLog:
    def __init__(self, log_dir: str = "audit_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
    
    def log_decision(self, decision: DecisionObject):
        timestamp = decision.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = self.log_dir / f"decision_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(decision.model_dump(mode='json'), f, indent=2, default=str)
        
        master_log = self.log_dir / "master_log.jsonl"
        with open(master_log, 'a') as f:
            f.write(json.dumps(decision.model_dump(mode='json'), default=str) + '\n')


# ============================================================================
# Main
# ============================================================================

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[red]Error: Set OPENAI_API_KEY[/red]")
        return
    
    council = AdvancedLLMCouncil(api_key, model="gpt-4o")
    
    questions = [
        "What are the key differences between supervised and unsupervised learning?",
        "How should startups balance growth vs profitability in 2025?"
    ]
    
    console.print("[bold cyan]🏛️  Advanced LLM Council with Reflection[/bold cyan]\n")
    
    for i, question in enumerate(questions, 1):
        console.print(f"\n[bold yellow]═══ Question {i}/{len(questions)} ═══[/bold yellow]")
        decision = council.deliberate(question, max_reflection_rounds=1)
        council.print_decision(decision)
        
        if i < len(questions):
            input("\n⏎ Press Enter to continue...")


if __name__ == "__main__":
    main()