import argparse

from loguru import logger

from realtime.llm.client import RealtimeClient


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an interactive voice conversation with custom instructions")
    parser.add_argument(
        "--instructions",
        type=str,
        default="You are a helpful assistant. Keep your responses concise and to the point.",
        help="Custom instructions for the AI assistant"
    )
    parser.add_argument(
        "--personality",
        type=str,
        choices=["helpful", "concise", "expert", "friendly", "creative"],
        help="Predefined personality to use (overrides custom instructions)"
    )

    return parser

def main():
    """Main function"""
    logger.info("Starting voice-activated realtime conversation")

    args = create_parser().parse_args()

    # Define personality-based instructions
    personalities = {
        "helpful": "You are a helpful assistant. Try to provide detailed answers to user questions.",
        "concise": "You are a concise assistant. Keep your responses brief and to the point.",
        "expert": "You are an expert assistant. Use technical terminology and provide in-depth analysis.",
        "friendly": "You are a friendly assistant. Use a warm, conversational tone and be empathetic.",
        "creative": "You are a creative assistant. Think outside the box and provide unique perspectives."
    }

    if args.personality:
        instructions = personalities[args.personality]
        print(f"Using {args.personality} personality: {instructions}")
    else:
        instructions = args.instructions
        print(f"Using custom instructions: {instructions}")

    # Create the client with the selected instructions
    client = RealtimeClient(default_instructions=instructions)

    try:
        client.connect()

        print("\n" + "=" * 60)
        print("Voice-Activated Realtime Conversation")
        print("=" * 60)
        print("\nJust start speaking to interact with the assistant.")
        print("The system will automatically detect when you start and stop talking.")
        print("\nPress Ctrl+C to exit.")
        print("=" * 60 + "\n")

        client.run_conversation()

    except Exception as e:
        logger.exception(f"Error in main: {e}")
    finally:
        print("\nExiting voice-activated conversation. Goodbye!")