"""CLI interface for the travel booking agent."""
import os
import subprocess
import sys
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from agent import run_agent_streaming, create_travel_graph
import warnings

load_dotenv()

console = Console()

# Suppress warning
warnings.filterwarnings(
    "ignore",
    message="LangSmith now uses UUID v7 for run and trace identifiers"
)

def check_setup():
    """Check if environment is properly set up."""
    model = os.getenv("MODEL", "llama3.2")

    console.print(f"Using model: {model}")
    
    if not os.path.exists("./chroma_db"):
        console.print("Warning: Knowledge base not initialized")
        console.print("Running setup_kb.py to initialize knowledge base...")

        try:
            subprocess.run([sys.executable, "-m", "knowledge_base.setup_kb"], check=True)
            console.print("✓ Knowledge base initialized")
        except subprocess.CalledProcessError:
            console.print("Error: Failed to initialize knowledge base")
            return False

    return True


def main():
    """Main CLI loop."""
    console.print(Panel.fit(
        "Travel Booking Assistant\n"
        "I can help you search for flights, hotels, and answer travel questions.\n"
        "Type 'exit' or 'quit' to leave.",
    ))

    if not check_setup():
        sys.exit(1)

    if os.getenv("LANGCHAIN_TRACING_V2") == "true":
        console.print("LangSmith tracing enabled\n")

    graph = create_travel_graph()
    state = {"messages": []}

    while True:
        try:
            user_input = console.input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                console.print("\nThank you for using Travel Booking Assistant! Safe travels!")
                break

            response_parts = []
            tool_calls = []

            try:
                for event in run_agent_streaming(graph, user_input, state):
                    if event["type"] == "tool_call":
                        tool_calls.append(event)
                        console.print(f"(Using tool: {event['tool']})")
                    elif event["type"] == "response":
                        response_parts.append(event["content"])
                        console.print(f"Assistant: {event['content']}", end="")

                if response_parts:
                    console.print()
                elif not tool_calls:
                    console.print("No response generated. Please try again.")

            except Exception as e:
                console.print(f"\nError: {str(e)}")
                console.print_exception()

        except KeyboardInterrupt:
            console.print("\n\nInterrupted. Type 'exit' to quit.")
        except EOFError:
            console.print("\nGoodbye!")
            break
        except Exception as e:
            console.print(f"Unexpected error: {str(e)}")
            console.print_exception()


if __name__ == "__main__":
    main()

