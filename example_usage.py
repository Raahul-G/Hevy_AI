"""
Example usage of the Hevy AI planning system.

This demonstrates the end-to-end workflow of goal interpretation and planning.
"""

import os
from pathlib import Path

# Add project root to Python path for imports
import sys

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.graph.planning_graph import PlanningGraph


def main():
    """Example usage of the planning system."""
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Please create a .env file with your API key.")
        print("See .env.example for reference.")
        return

    # Initialize the planning graph
    graph = PlanningGraph()

    # Example user input
    user_input = "I want to lose 10kg in 3 months. I have limited equipment at home and can only workout 30 minutes a day. I enjoy strength training but don't like running."

    print("=" * 60)
    print("Hevy AI - Planning System Example")
    print("=" * 60)
    print(f"\nUser Input: {user_input}\n")
    print("Processing...\n")

    try:
        # Run the planning workflow
        result = graph.run(user_input, user_id="example_user")

        # Display results
        print("=" * 60)
        print("Results")
        print("=" * 60)

        if result.get("extracted_goal"):
            goal = result["extracted_goal"]
            print(f"\n✓ Extracted Goal:")
            print(f"  - Type: {goal.goal_type}")
            print(f"  - Target: {goal.target_metric}")
            print(f"  - Timeframe: {goal.timeframe}")
            print(f"  - Status: {goal.current_status}")
            print(f"  - Constraints: {', '.join(goal.constraints) if goal.constraints else 'None'}")
            print(f"  - Preferences: {', '.join(goal.preferences) if goal.preferences else 'None'}")

        if result.get("user_profile"):
            profile = result["user_profile"]
            print(f"\n✓ User Profile:")
            print(f"  - User ID: {profile.user_id}")
            print(f"  - Profile created: {profile.created_at}")

        if result.get("plan"):
            plan = result["plan"]
            print(f"\n✓ Generated Plan:")
            print(f"  - Valid: {result.get('plan_valid', False)}")
            print(f"  - Content Preview:")
            content = plan.get("plan_content", "")
            preview = content[:200] + "..." if len(content) > 200 else content
            print(f"    {preview}")

        if result.get("error"):
            print(f"\n⚠ Error: {result['error']}")

        print("\n" + "=" * 60)
        print("Workflow completed successfully!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

