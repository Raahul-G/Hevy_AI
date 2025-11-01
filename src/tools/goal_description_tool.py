"""Goal Description Tool for extracting structured goals from natural language."""

from typing import Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from config.llm_config import get_llm
from src.schemas.user_goal import UserGoal


@tool
def goal_description_tool(user_input: str) -> dict:
    """
    Extract structured goal information from natural language user input.

    This tool uses an LLM to parse unstructured user text and extract
    structured goal information conforming to the UserGoal schema.

    Args:
        user_input: Natural language description of the user's fitness goal

    Returns:
        Dictionary representation of the extracted UserGoal

    Example:
        >>> result = goal_description_tool("I want to lose 10kg in 3 months")
        >>> print(result["goal_type"])
        'weight_loss'
    """
    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=UserGoal)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an expert fitness advisor that helps users articulate their fitness goals.
Your task is to extract structured information from user statements about their fitness objectives.

Analyze the user's input and extract:
1. The primary goal type (weight_loss, muscle_gain, endurance, general_wellness)
2. Specific target metrics or objectives
3. The timeframe for achieving the goal
4. Any constraints or limitations mentioned
5. Preferences about workout styles or activities
6. Current fitness status or activity level

If information is not explicitly stated, make reasonable inferences based on context.
For missing information, use sensible defaults:
- If no constraints mentioned, use empty list
- If no preferences mentioned, use empty list
- If status unclear, default to 'beginner'

{format_instructions}
""",
            ),
            ("human", "{user_input}"),
        ]
    )

    chain = prompt | llm.with_structured_output(UserGoal)

    try:
        result = chain.invoke({"user_input": user_input})
        return result.model_dump()
    except Exception as e:
        # Fallback: Try with parser if structured output fails
        try:
            formatted_prompt = prompt.format_messages(
                user_input=user_input,
                format_instructions=parser.get_format_instructions(),
            )
            response = llm.invoke(formatted_prompt)
            parsed = parser.parse(response.content)
            return parsed.model_dump()
        except Exception as parse_error:
            raise ValueError(
                f"Failed to extract goal from input: {str(e)}. Parse error: {str(parse_error)}"
            ) from parse_error


class GoalDescriptionTool:
    """
    Goal Description Tool class wrapper for easier integration.

    This class provides a convenient interface to the goal extraction tool
    with additional error handling and validation.
    """

    def __init__(self):
        """Initialize the Goal Description Tool."""
        self._tool = goal_description_tool
        self._parser = PydanticOutputParser(pydantic_object=UserGoal)

    def extract_goal(self, user_input: str) -> UserGoal:
        """
        Extract a structured UserGoal from natural language input.

        Args:
            user_input: Natural language description of the user's goal

        Returns:
            Validated UserGoal object

        Raises:
            ValueError: If goal extraction fails
        """
        result_dict = self._tool.invoke({"user_input": user_input})
        
        # Validate the result is a valid UserGoal
        try:
            return UserGoal(**result_dict)
        except Exception as e:
            raise ValueError(
                f"Failed to validate extracted goal: {str(e)}. "
                f"Extracted data: {result_dict}"
            ) from e

    def extract_goal_safe(self, user_input: str) -> Optional[UserGoal]:
        """
        Extract goal with error handling, returning None on failure.

        Args:
            user_input: Natural language description of the user's goal

        Returns:
            UserGoal object if extraction succeeds, None otherwise
        """
        try:
            return self.extract_goal(user_input)
        except Exception:
            return None

