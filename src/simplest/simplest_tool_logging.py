import dspy
from typing import Any, Dict, List
from dspy.utils.callback import BaseCallback
from common.utils import get_lm_for_model_name, dspy_configure
from common.constants import MODEL_NAME_GEMINI_2_5_FLASH


# Tool usage tracker for capturing tool calls
class ToolUsageTracker:
    """Tracks tool calls with inputs and outputs."""
    
    def __init__(self):
        self.tool_logs: List[Dict[str, Any]] = []
    
    def log_tool_call(self, tool_name: str, tool_args: Dict[str, Any], tool_output: Any) -> None:
        """Log a tool call with its inputs and output."""
        log_entry = {
            "tool_name": tool_name,
            "inputs": tool_args,
            "output": tool_output,
            "status": "completed"
        }
        self.tool_logs.append(log_entry)
        print(f"🔧 Tool '{tool_name}' called with inputs: {tool_args} -> output: {tool_output}")
    
    def get_tool_logs(self) -> List[Dict[str, Any]]:
        """Get all tracked tool logs."""
        return self.tool_logs
    
    def print_summary(self) -> None:
        """Print a summary of all tool calls."""
        if not self.tool_logs:
            print("\n📊 No tool calls were tracked.")
            return
        
        print("\n" + "="*60)
        print("📊 Tool Usage Summary")
        print("="*60)
        for i, log in enumerate(self.tool_logs, 1):
            print(f"\n{i}. Tool: {log['tool_name']}")
            print(f"   Inputs: {log['inputs']}")
            print(f"   Output: {log['output']}")
            print(f"   Status: {log['status']}")
        print("="*60 + "\n")


# DSPy-native callback handler for tool tracking
class ToolCallCallback(BaseCallback):
    """
    DSPy-native callback that intercepts and logs tool calls.
    
    This callback is registered with DSPy's callback system to track tool executions
    without modifying tool source code. Works with any tool, even if we don't have
    access to the tool's source code.
    """
    
    def __init__(self, tracker: ToolUsageTracker):
        """
        Initialize the tool call callback.
        
        Args:
            tracker: The ToolUsageTracker instance to log calls to
        """
        super().__init__()
        self.tracker = tracker
        # Store pending tool calls to match start/end
        self._pending_calls: Dict[str, Dict[str, Any]] = {}
    
    def on_tool_start(self, call_id: str, instance: dspy.Tool, inputs: Dict[str, Any]) -> None:
        """
        Called when a tool execution starts.
        
        Args:
            call_id: Unique identifier for this tool call
            instance: The dspy.Tool instance being called
            inputs: Dictionary of tool input arguments (may be wrapped in 'kwargs' by ReAct)
        """
        # Extract tool name
        tool_name = instance.name if hasattr(instance, 'name') and instance.name else (
            instance.func.__name__ if hasattr(instance, 'func') else "unknown_tool"
        )
        
        # Unwrap kwargs if present (ReAct wraps arguments in 'kwargs' dict)
        actual_inputs = inputs
        if isinstance(inputs, dict) and 'kwargs' in inputs:
            kw = inputs.get('kwargs')
            if isinstance(kw, dict):
                actual_inputs = kw
            else:
                actual_inputs = inputs
        
        # Store pending call info
        self._pending_calls[call_id] = {
            "tool_name": tool_name,
            "inputs": actual_inputs
        }
    
    def on_tool_end(self, call_id: str, outputs: Any, exception: Exception | None = None) -> None:
        """
        Called when a tool execution ends.
        
        Args:
            call_id: Unique identifier for this tool call
            outputs: The tool's output/result
            exception: Exception if tool failed, None if successful
        """
        if call_id not in self._pending_calls:
            return  # Should not happen, but handle gracefully
        
        call_info = self._pending_calls.pop(call_id)
        tool_name = call_info["tool_name"]
        inputs = call_info["inputs"]
        
        # Log the tool call
        if exception is None:
            self.tracker.log_tool_call(tool_name, inputs, outputs)
        else:
            # Log error case
            self.tracker.log_tool_call(tool_name, inputs, f"ERROR: {exception}")


# Define the tools (without any tracking code - clean functions)
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        The sum of a and b
    """
    return a + b


def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        The product of a and b
    """
    return a * b


def main():
    """Main function demonstrating DSPy ReAct agent with native tool logging."""
    # Configure DSPy (already sets track_usage=True)
    dspy_configure(get_lm_for_model_name(MODEL_NAME_GEMINI_2_5_FLASH, "disable"))
    
    # Initialize tool tracker
    tool_tracker = ToolUsageTracker()
    
    # Create DSPy-native callback handler
    tool_callback = ToolCallCallback(tool_tracker)
    
    # Save existing callbacks and register our callback
    existing_callbacks = dspy.settings.get("callbacks", []) or []
    dspy.settings.configure(callbacks=existing_callbacks + [tool_callback])
    
    try:
        # Create tool instances (clean - no wrapping needed)
        # DSPy's callback system will intercept calls automatically
        add_tool = dspy.Tool(add_numbers)
        multiply_tool = dspy.Tool(multiply_numbers)
        
        # Create ReAct agent with tools
        # Use string signature as DSPy ReAct supports it
        react_agent = dspy.ReAct(
            signature="question -> answer",  # type: ignore[arg-type]
            tools=[add_tool, multiply_tool],
            max_iters=10
        )
        
        # Example queries
        questions = [
            "What is 3 + 5?",
            "What is 4 * 7?",
            "Calculate 10 + 15 and then multiply the result by 2"
        ]
        
        print("\n" + "="*60)
        print("🤖 ReAct Agent with Tool Logging")
        print("="*60 + "\n")
        
        prediction = None
        for question in questions:
            print(f"❓ Question: {question}")
            prediction = react_agent(question=question)
            print(f"💡 Answer: {prediction.answer}\n")
        
        # Display tool usage summary
        tool_tracker.print_summary()
        
        # Also show LM usage if available
        if prediction:
            try:
                # Try to get usage from the last prediction
                if hasattr(prediction, 'get_lm_usage'):
                    usage = prediction.get_lm_usage()
                    if usage:
                        print("📈 LM Usage Statistics:")
                        print(f"   {usage}\n")
            except Exception:
                pass  # Usage stats might not be available in all cases
    
    finally:
        # Restore original callbacks to clean up global state
        dspy.settings.configure(callbacks=existing_callbacks)


if __name__ == "__main__":
    main()

