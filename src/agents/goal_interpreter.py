"""LangChain application for goal interpretation and extraction."""

from typing import List, Optional, Tuple

from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config.llm_config import get_llm
from src.schemas.user_goal import UserGoal
from src.tools.goal_description_tool import GoalDescriptionTool


class GoalInterpreter:
    """
    LangChain-based application for interpreting user goals from natural language.

    This agent provides a conversational interface that helps users articulate
    their fitness goals and extracts structured goal information.
    """

    def __init__(self, memory: Optional[ConversationBufferMemory] = None):
        """
        Initialize the Goal Interpreter.

        Args:
            memory: Optional conversation memory for maintaining context.
                    If None, a new ConversationBufferMemory will be created.
        """
        self.llm = get_llm()
        self.goal_tool = GoalDescriptionTool()
        self.memory = memory or ConversationBufferMemory(
            return_messages=True, memory_key="chat_history"
        )

        # System prompt for goal interpretation
        self.system_prompt = """You are a helpful fitness advisor assistant. Your role is to:

1. Understand the user's fitness goals by asking clarifying questions when needed
2. Extract structured information about their objectives
3. Help them articulate their goals more clearly

Be conversational, friendly, and ask follow-up questions if the user's goal description
is incomplete or unclear. Once you have enough information, extract the structured goal.

Focus on:
- What type of fitness goal (weight loss, muscle gain, endurance, general wellness)
- Specific targets or metrics
- Timeframe for achievement
- Any constraints (equipment, time, health issues)
- Preferences (workout styles, activities)
- Current fitness status

Keep the conversation natural and helpful."""

        # Create the prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=self.system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{user_input}"),
            ]
        )

    def interpret_goal(
        self, user_input: str, conversation_history: Optional[List] = None
    ) -> UserGoal:
        """
        Interpret user input and extract a structured goal.

        Args:
            user_input: Natural language description of the user's goal
            conversation_history: Optional list of previous messages for context

        Returns:
            Structured UserGoal object

        Raises:
            ValueError: If goal extraction fails after multiple attempts
        """
        # Prepare messages
        messages = [HumanMessage(content=user_input)]

        # Add conversation history if provided
        if conversation_history:
            messages = conversation_history + messages

        # Get LLM response for conversational interaction
        formatted_messages = self.prompt.format_messages(
            user_input=user_input,
            chat_history=conversation_history or [],
        )

        llm_response = self.llm.invoke(formatted_messages)

        # Use the goal tool to extract structured information
        # Combine user input with LLM's understanding if it asked clarifying questions
        extraction_input = user_input
        if llm_response.content and "?" in llm_response.content:
            # If LLM asked questions, use original input but tool should handle it
            # The tool will extract what it can from available information
            pass

        # Extract structured goal
        goal = self.goal_tool.extract_goal(extraction_input)

        # Save to memory
        self.memory.save_context(
            {"input": user_input}, {"output": llm_response.content}
        )

        return goal

    def interpret_goal_with_clarification(
        self, initial_input: str, max_iterations: int = 3
    ) -> Tuple[UserGoal, List[str]]:
        """
        Interpret goal with interactive clarification if needed.

        Args:
            initial_input: Initial user input about their goal
            max_iterations: Maximum number of clarification rounds

        Returns:
            Tuple of (UserGoal, list of clarification questions/answers)
        """
        conversation_log = []
        current_input = initial_input
        extracted_goal = None

        for iteration in range(max_iterations):
            # Get conversational response
            messages = self.prompt.format_messages(
                user_input=current_input,
                chat_history=conversation_log,
            )

            llm_response = self.llm.invoke(messages)
            conversation_log.append(HumanMessage(content=current_input))
            conversation_log.append(llm_response)

            # Try to extract goal
            extracted_goal = self.goal_tool.extract_goal_safe(current_input)

            if extracted_goal:
                break

            # If extraction failed, ask for clarification
            if iteration < max_iterations - 1:
                clarification = llm_response.content
                conversation_log.append(HumanMessage(content=clarification))
                current_input = clarification

        if not extracted_goal:
            # Final attempt with full conversation context
            full_context = " ".join(
                [
                    msg.content
                    for msg in conversation_log
                    if isinstance(msg, HumanMessage)
                ]
            )
            extracted_goal = self.goal_tool.extract_goal(full_context)

        return extracted_goal, [
            msg.content for msg in conversation_log if hasattr(msg, "content")
        ]

    def get_conversation_history(self) -> List:
        """Get the current conversation history."""
        return self.memory.chat_memory.messages if hasattr(self.memory, "chat_memory") else []

