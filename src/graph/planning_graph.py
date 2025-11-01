"""LangGraph state machine for orchestrating planning workflows."""

from typing import Annotated, Dict, List, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from config.llm_config import get_llm
from src.agents.goal_interpreter import GoalInterpreter
from src.profiles.profile_manager import ProfileManager, UserProfile
from src.schemas.user_goal import UserGoal


class PlanningState(TypedDict):
    """
    State schema for the planning graph.

    This defines all the data that flows through the graph nodes.
    """

    user_input: str
    user_id: str
    user_profile: UserProfile | None
    extracted_goal: UserGoal | None
    plan: Dict | None
    plan_valid: bool
    error: str | None
    messages: List[str]  # Conversation messages


def extract_goal_node(state: PlanningState) -> PlanningState:
    """
    Extract structured goal from user input.

    Args:
        state: Current planning state

    Returns:
        Updated state with extracted_goal
    """
    try:
        interpreter = GoalInterpreter()
        goal = interpreter.interpret_goal(state["user_input"])
        return {
            **state,
            "extracted_goal": goal,
            "messages": state.get("messages", []) + [f"Extracted goal: {goal.goal_type}"],
        }
    except Exception as e:
        return {
            **state,
            "extracted_goal": None,
            "error": f"Failed to extract goal: {str(e)}",
            "messages": state.get("messages", []) + [f"Error: {str(e)}"],
        }


def load_profile_node(state: PlanningState) -> PlanningState:
    """
    Load or create user profile.

    Args:
        state: Current planning state

    Returns:
        Updated state with user_profile
    """
    try:
        manager = ProfileManager()
        user_id = state.get("user_id", "default_user")
        
        # Get or create profile
        profile = manager.get_or_create_profile(user_id)
        
        return {
            **state,
            "user_profile": profile,
            "user_id": user_id,
            "messages": state.get("messages", []) + [f"Loaded profile for user: {user_id}"],
        }
    except Exception as e:
        return {
            **state,
            "user_profile": None,
            "error": f"Failed to load profile: {str(e)}",
            "messages": state.get("messages", []) + [f"Error: {str(e)}"],
        }


def generate_plan_node(state: PlanningState) -> PlanningState:
    """
    Generate high-level plan based on goal and profile.

    Args:
        state: Current planning state

    Returns:
        Updated state with plan
    """
    try:
        goal = state.get("extracted_goal")
        profile = state.get("user_profile")

        if not goal:
            return {
                **state,
                "plan": None,
                "error": "Cannot generate plan without extracted goal",
                "messages": state.get("messages", []) + ["Error: No goal available"],
            }

        # Initialize LLM for plan generation
        llm = get_llm()

        # Create plan generation prompt
        goal_info = f"""
Goal Type: {goal.goal_type}
Target Metric: {goal.target_metric}
Timeframe: {goal.timeframe}
Constraints: {', '.join(goal.constraints) if goal.constraints else 'None'}
Preferences: {', '.join(goal.preferences) if goal.preferences else 'None'}
Current Status: {goal.current_status}
"""

        profile_info = ""
        if profile:
            profile_info = f"""
User Profile:
- Preferences: {', '.join(profile.preferences) if profile.preferences else 'None'}
- Constraints: {', '.join(profile.constraints) if profile.constraints else 'None'}
- Age: {profile.age if profile.age else 'Not specified'}
"""

        prompt = f"""You are an expert fitness coach. Create a high-level plan for achieving the following goal.

{goal_info}

{profile_info}

Generate a comprehensive but high-level plan that includes:
1. Overview/Summary
2. Key milestones and phases
3. Recommended approach
4. Important considerations

Format the plan as a structured response with clear sections."""

        response = llm.invoke(prompt)

        plan = {
            "goal": goal.model_dump(),
            "plan_content": response.content,
            "generated_at": __import__("datetime").datetime.now().isoformat(),
        }

        return {
            **state,
            "plan": plan,
            "messages": state.get("messages", []) + ["Generated high-level plan"],
        }
    except Exception as e:
        return {
            **state,
            "plan": None,
            "error": f"Failed to generate plan: {str(e)}",
            "messages": state.get("messages", []) + [f"Error: {str(e)}"],
        }


def validate_and_refine_node(state: PlanningState) -> PlanningState:
    """
    Validate plan completeness and refine if needed.

    Args:
        state: Current planning state

    Returns:
        Updated state with validated plan
    """
    try:
        plan = state.get("plan")
        goal = state.get("extracted_goal")

        if not plan:
            return {
                **state,
                "plan_valid": False,
                "error": "No plan to validate",
                "messages": state.get("messages", []) + ["Validation failed: No plan"],
            }

        # Basic validation checks
        plan_valid = (
            plan.get("plan_content") is not None
            and len(plan.get("plan_content", "")) > 50  # Minimum content length
            and goal is not None
        )

        # If invalid, try to refine
        if not plan_valid and goal:
            llm = get_llm()
            refine_prompt = f"""The following plan needs refinement. Please improve it to be more comprehensive.

Goal: {goal.model_dump()}
Current Plan: {plan.get('plan_content', '')}

Provide an improved, more detailed plan."""
            
            refined_response = llm.invoke(refine_prompt)
            plan["plan_content"] = refined_response.content
            plan_valid = True

        return {
            **state,
            "plan": plan,
            "plan_valid": plan_valid,
            "messages": state.get("messages", []) + [
                "Plan validated" if plan_valid else "Plan validation failed"
            ],
        }
    except Exception as e:
        return {
            **state,
            "plan_valid": False,
            "error": f"Validation error: {str(e)}",
            "messages": state.get("messages", []) + [f"Validation error: {str(e)}"],
        }


def should_refine(state: PlanningState) -> Literal["refine", "save_profile", END]:
    """
    Conditional routing: decide whether to refine plan or save profile.

    Args:
        state: Current planning state

    Returns:
        Next node to route to
    """
    if state.get("error"):
        return END
    
    if not state.get("plan_valid", False):
        return "refine"
    
    return "save_profile"


def refine_plan_node(state: PlanningState) -> PlanningState:
    """
    Refine the plan by regenerating with additional context.

    Args:
        state: Current planning state

    Returns:
        Updated state with refined plan
    """
    # Call generate_plan_node again, which will use existing goal
    return generate_plan_node(state)


def save_profile_node(state: PlanningState) -> PlanningState:
    """
    Save goal to user profile.

    Args:
        state: Current planning state

    Returns:
        Updated state
    """
    try:
        goal = state.get("extracted_goal")
        user_id = state.get("user_id", "default_user")

        if goal:
            manager = ProfileManager()
            manager.update_profile_with_goal(user_id, goal)
            return {
                **state,
                "messages": state.get("messages", []) + ["Profile updated with goal"],
            }
        
        return {
            **state,
            "messages": state.get("messages", []) + ["No goal to save"],
        }
    except Exception as e:
        return {
            **state,
            "error": f"Failed to save profile: {str(e)}",
            "messages": state.get("messages", []) + [f"Save error: {str(e)}"],
        }


class PlanningGraph:
    """
    LangGraph state machine for orchestrating planning workflows.

    This graph coordinates goal extraction, profile loading, plan generation,
    and validation in a structured workflow.
    """

    def __init__(self):
        """Initialize the planning graph."""
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build and compile the LangGraph workflow."""
        workflow = StateGraph(PlanningState)

        # Add nodes
        workflow.add_node("extract_goal", extract_goal_node)
        workflow.add_node("load_profile", load_profile_node)
        workflow.add_node("generate_plan", generate_plan_node)
        workflow.add_node("validate", validate_and_refine_node)
        workflow.add_node("refine", refine_plan_node)
        workflow.add_node("save_profile", save_profile_node)

        # Define edges
        workflow.add_edge(START, "extract_goal")
        workflow.add_edge("extract_goal", "load_profile")
        workflow.add_edge("load_profile", "generate_plan")
        workflow.add_edge("generate_plan", "validate")
        
        # Conditional routing from validate
        workflow.add_conditional_edges(
            "validate",
            should_refine,
            {
                "refine": "refine",
                "save_profile": "save_profile",
                END: END,
            },
        )
        
        # Refine loops back to validate
        workflow.add_edge("refine", "validate")
        
        # Save profile goes to end
        workflow.add_edge("save_profile", END)

        return workflow.compile()

    def run(
        self,
        user_input: str,
        user_id: str = "default_user",
        initial_state: Dict | None = None,
    ) -> PlanningState:
        """
        Run the planning workflow.

        Args:
            user_input: Natural language description of user's goal
            user_id: Unique identifier for the user
            initial_state: Optional initial state dictionary

        Returns:
            Final planning state after workflow execution
        """
        initial = {
            "user_input": user_input,
            "user_id": user_id,
            "user_profile": None,
            "extracted_goal": None,
            "plan": None,
            "plan_valid": False,
            "error": None,
            "messages": [],
            **(initial_state or {}),
        }

        result = self.graph.invoke(initial)
        return result

    def stream(
        self,
        user_input: str,
        user_id: str = "default_user",
        initial_state: Dict | None = None,
    ):
        """
        Stream the planning workflow execution.

        Args:
            user_input: Natural language description of user's goal
            user_id: Unique identifier for the user
            initial_state: Optional initial state dictionary

        Yields:
            State updates as the workflow progresses
        """
        initial = {
            "user_input": user_input,
            "user_id": user_id,
            "user_profile": None,
            "extracted_goal": None,
            "plan": None,
            "plan_valid": False,
            "error": None,
            "messages": [],
            **(initial_state or {}),
        }

        for state in self.graph.stream(initial):
            yield state

