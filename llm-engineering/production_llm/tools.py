import ast
import operator as op
import hashlib
import json
import re
from typing import Callable, Any, Dict, List

# Supported operators for safe math evaluation
operators = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.USub: op.neg,
    ast.UAdd: op.pos
}

def eval_node(node):
    if hasattr(ast, "Constant") and isinstance(node, ast.Constant):  # Python 3.8+
        if isinstance(node.value, (int, float)):
            return node.value
        raise TypeError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.Num):  # Python < 3.8 fallback
        return node.n
    elif isinstance(node, ast.BinOp):
        left = eval_node(node.left)
        right = eval_node(node.right)
        op_type = type(node.op)
        if op_type in operators:
            return operators[op_type](left, right)
        raise TypeError(f"Unsupported binary operator: {op_type}")
    elif isinstance(node, ast.UnaryOp):
        operand = eval_node(node.operand)
        op_type = type(node.op)
        if op_type in operators:
            return operators[op_type](operand)
        raise TypeError(f"Unsupported unary operator: {op_type}")
    else:
        raise TypeError(f"Unsupported AST node: {type(node)}")

def safe_eval(expr_str: str) -> str:
    try:
        # Strip all whitespace
        cleaned = re.sub(r"\s+", "", expr_str)
        # Parse expression in 'eval' mode
        tree = ast.parse(cleaned, mode="eval")
        result = eval_node(tree.body)
        return str(result)
    except Exception as e:
        return f"Error: Failed to evaluate expression. {str(e)}"


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Dict[str, Any]] = {}

    def register(self, name: str, description: str, parameters: Dict[str, Any]):
        def decorator(func: Callable):
            self._tools[name] = {
                "func": func,
                "description": description,
                "parameters": parameters
            }
            return func
        return decorator

    def get_tool(self, name: str) -> Dict[str, Any] | None:
        return self._tools.get(name)

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": name,
                "description": info["description"],
                "parameters": info["parameters"]
            }
            for name, info in self._tools.items()
        ]

    def execute(self, name: str, **kwargs) -> Any:
        tool = self.get_tool(name)
        if not tool:
            raise ValueError(f"Tool {name} not found.")
        # Ensure we only pass arguments expected by the tool function
        sig = inspect_signature = getattr(tool["func"], "__code__", None)
        # Simple invocation
        return tool["func"](**kwargs)


tool_registry = ToolRegistry()


@tool_registry.register(
    name="weather",
    description="Get the current weather for a specific location.",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "The city and state/country, e.g. San Francisco, CA"
            }
        },
        "required": ["location"]
    }
)
def get_weather(location: str) -> str:
    loc = location.lower()
    if "london" in loc:
        return "60°F and light rain."
    elif "paris" in loc:
        return "68°F and partly cloudy."
    elif "tokyo" in loc:
        return "75°F and sunny."
    elif "san francisco" in loc:
        return "57°F and foggy."
    elif "seattle" in loc:
        return "55°F and drizzling."
    else:
        # Deterministic temperature and conditions based on location hash
        h = int(hashlib.md5(location.encode()).hexdigest(), 16)
        temp = 50 + (h % 40)
        conditions = ["sunny", "cloudy", "rainy", "windy"][h % 4]
        return f"{temp}°F and {conditions}."


@tool_registry.register(
    name="calculate",
    description="Evaluate mathematical expressions. Supports basic arithmetic (+, -, *, /, parentheses).",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "The mathematical expression to evaluate, e.g. 2 + 2 * 3"
            }
        },
        "required": ["expression"]
    }
)
def calculate_tool(expression: str) -> str:
    return safe_eval(expression)


@tool_registry.register(
    name="search",
    description="Search external databases and search engines for current information on a topic.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The query to search for."
            }
        },
        "required": ["query"]
    }
)
def search_tool(query: str) -> str:
    q = query.lower()
    if "ceo of acme" in q or "acme corp ceo" in q:
        return "Search result: Jane Doe is the CEO of Acme Corp, appointed in 2024."
    elif "discount" in q or "billing" in q or "subscription" in q:
        return "Search result: Acme Corp offers a 20% discount on annual subscriptions paid upfront."
    elif "photosynthesis" in q:
        return "Search result: Photosynthesis is the process used by plants and other organisms to convert light energy into chemical energy."
    elif "france" in q or "capital of france" in q:
        return "Search result: Paris is the capital and most populous city of France."
    else:
        return f"Search result: No direct matches found for '{query}'. Please refine your query."


def detect_tool_mock(query: str) -> Dict[str, Any]:
    q = query.lower()
    if "weather" in q:
        match = re.search(r"weather\s+(?:in|for)\s+([a-zA-Z\s,]+)", q)
        location = match.group(1).strip() if match else "San Francisco"
        # Clean any trailing punctuation
        location = re.sub(r"[.?]+$", "", location).strip()
        return {"tool": "weather", "arguments": {"location": location}}
    elif any(keyword in q for keyword in ["calculate", "evaluate", "what is"]) and any(char in q for char in ["+", "-", "*", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]):
        # Extract expression
        match = re.search(r"(?:calculate|evaluate|what is)\s+([0-9+\-*/.()\s]+)", q)
        if match:
            expr = match.group(1).strip()
        else:
            # Just extract all math chars
            expr = "".join([c for c in query if c in "0123456789+-*/.() "]).strip()
        # Clean trailing punctuation
        expr = re.sub(r"[.?]+$", "", expr).strip()
        return {"tool": "calculate", "arguments": {"expression": expr}}
    elif any(keyword in q for keyword in ["search", "who is", "ceo", "discount", "photosynthesis"]):
        match = re.search(r"(?:search for|search|who is)\s+(.+)", q)
        search_q = match.group(1).strip() if match else query
        search_q = re.sub(r"[.?]+$", "", search_q).strip()
        return {"tool": "search", "arguments": {"query": search_q}}
    return {"tool": None, "arguments": {}}
