#!/usr/bin/env python3
"""
Qwen2.5-72B-Instruct Tool Calling Test

This example demonstrates how to use tool calling functionality
with the Qwen/Qwen2.5-72B-Instruct model in TensorRT-LLM.
"""

import asyncio
import json
import sys
import os
import argparse
from typing import List, Dict, Any

# Add TensorRT-LLM to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.serve.tool_call_manager import ToolCallManager
    from tensorrt_llm.serve.openai_protocol import (
        ChatCompletionRequest,
        ChatCompletionToolsParam,
        FunctionDefinition
    )
    from transformers import AutoTokenizer
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure TensorRT-LLM is properly installed and in your Python path")
    sys.exit(1)


def create_qwen_tools():
    """Create tool definitions suitable for Qwen models"""
    
    tools = [
        ChatCompletionToolsParam(
            type="function",
            function=FunctionDefinition(
                name="get_current_weather",
                description="Get current weather information for a specified city",
                parameters={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name, e.g., New York, London, Tokyo"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit, default is celsius",
                            "default": "celsius"
                        }
                    },
                    "required": ["location"]
                }
            )
        ),
        ChatCompletionToolsParam(
            type="function",
            function=FunctionDefinition(
                name="calculate_math",
                description="Perform mathematical calculations",
                parameters={
                    "type": "object", 
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Mathematical expression to calculate, e.g., 2+3*4"
                        }
                    },
                    "required": ["expression"]
                }
            )
        ),
        ChatCompletionToolsParam(
            type="function",
            function=FunctionDefinition(
                name="search_information",
                description="Search for relevant information",
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search keywords"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 10
                        }
                    },
                    "required": ["query"]
                }
            )
        )
    ]
    
    return tools


def create_qwen_chat_template_with_tools(tools: List[ChatCompletionToolsParam], user_message: str) -> str:
    """Create a chat template with tool definitions for Qwen models"""
    
    # Build tool descriptions
    tools_description = "# Available Tools\n\n"
    tools_description += "You can use the following tools to help answer user questions:\n\n"
    
    for i, tool in enumerate(tools, 1):
        func = tool.function
        tools_description += f"## Tool {i}: {func.name}\n"
        tools_description += f"**Description**: {func.description}\n"
        
        if func.parameters:
            tools_description += f"**Parameters**:\n```json\n{json.dumps(func.parameters, indent=2, ensure_ascii=False)}\n```\n\n"
    
    tools_description += "## Tool Call Format\n\n"
    tools_description += "When you need to call a tool, please output in the following JSON format:\n"
    tools_description += "```json\n"
    tools_description += '{"name": "tool_name", "arguments": {"param_name": "param_value"}}\n'
    tools_description += "```\n\n"
    tools_description += "Or, if you need to think, you can use:\n"
    tools_description += "<|thinking|>\nYour thinking process...\n</|thinking|>\n\n"
    tools_description += '{"name": "tool_name", "arguments": {"param_name": "param_value"}}\n\n'
    
    # Build complete chat template
    chat_template = f"""<|im_start|>system
You are a helpful assistant that can call tools to help users.

{tools_description}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
"""
    
    return chat_template


def simulate_tool_execution(tool_call_result: Dict[str, Any]) -> str:
    """Simulate tool execution and return results"""
    
    tool_calls = tool_call_result.get('tool_calls', [])
    if not tool_calls:
        return "No tool calls detected"
    
    results = []
    
    for tool_call in tool_calls:
        func_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        
        # Simulate different tool executions
        if func_name == "get_current_weather":
            location = arguments.get("location", "Unknown location")
            unit = arguments.get("unit", "celsius")
            temp_unit = "°C" if unit == "celsius" else "°F"
            temp = "22" if unit == "celsius" else "72"
            
            result = f"Current weather in {location}: Sunny, temperature {temp}{temp_unit}, humidity 65%, light breeze"
            
        elif func_name == "calculate_math":
            expression = arguments.get("expression", "")
            try:
                # Safe mathematical calculation
                import ast
                import operator
                
                # Supported operations
                ops = {
                    ast.Add: operator.add,
                    ast.Sub: operator.sub,
                    ast.Mult: operator.mul,
                    ast.Div: operator.truediv,
                    ast.Pow: operator.pow,
                    ast.USub: operator.neg,
                }
                
                def eval_expr(node):
                    if isinstance(node, ast.Constant):
                        return node.value
                    elif isinstance(node, ast.BinOp):
                        return ops[type(node.op)](eval_expr(node.left), eval_expr(node.right))
                    elif isinstance(node, ast.UnaryOp):
                        return ops[type(node.op)](eval_expr(node.operand))
                    else:
                        raise TypeError(f"Unsupported operation: {node}")
                
                tree = ast.parse(expression, mode='eval')
                calc_result = eval_expr(tree.body)
                result = f"Calculation result: {expression} = {calc_result}"
                
            except Exception as e:
                result = f"Calculation error: {e}"
                
        elif func_name == "search_information":
            query = arguments.get("query", "")
            max_results = arguments.get("max_results", 5)
            result = f"Search results for '{query}' (simulated): Found {max_results} relevant items..."
            
        else:
            result = f"Unknown tool: {func_name}"
            
        results.append(f"{func_name}: {result}")
    
    return "\n".join(results)


def test_qwen_tool_calling(model_path: str, max_tokens: int = 512, tensor_parallel_size: int = 8, gpu_memory_utilization: float = 0.9):
    """Test Qwen model tool calling functionality"""
    
    print("=== Initializing Qwen2.5-72B-Instruct Model ===")
    
    try:
        # Load tokenizer
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Initialize TensorRT-LLM with tensor parallelism
        print(f"Loading TensorRT-LLM model with {tensor_parallel_size}-way tensor parallelism...")
        llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,  # Multi-GPU tensor parallelism
            gpu_memory_utilization=gpu_memory_utilization,  # GPU memory utilization
            distributed_executor_backend="mp",  # Use multiprocessing backend for multi-GPU
            trust_remote_code=True
        )
        
        # Set sampling parameters
        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.9,
            max_tokens=max_tokens,
            stop_token_ids=[tokenizer.eos_token_id],
        )
        
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Model loading failed: {e}")
        print("\nPlease ensure:")
        print("1. Model path is correct")
        print("2. TensorRT-LLM is properly installed and compiled")
        print("3. Sufficient GPU memory is available")
        return
    
    # Create tool definitions
    tools = create_qwen_tools()
    
    # Initialize tool call manager
    tool_manager = ToolCallManager(tokenizer, "qwen")
    
    print("\n=== Starting Tool Calling Tests ===\n")
    
    # Test cases
    test_cases = [
        {
            "name": "Weather Query",
            "message": "Please help me check the weather in Beijing"
        },
        {
            "name": "Math Calculation", 
            "message": "Please calculate 15 * 8 + 32 / 4"
        },
        {
            "name": "Information Search",
            "message": "Search for the latest developments in artificial intelligence"
        },
        {
            "name": "Composite Task",
            "message": "Please check the weather in Shanghai, then calculate 100 * 0.8"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"Test {i}: {test_case['name']}")
        print(f"User Input: {test_case['message']}")
        print("-" * 60)
        
        # Create chat template
        prompt = create_qwen_chat_template_with_tools(tools, test_case['message'])
        
        # Generate response
        print("Model generating...")
        try:
            outputs = llm.generate([prompt], sampling_params)
            generated_text = outputs[0].outputs[0].text
            
            print(f"Raw Output:\n{generated_text}\n")
            
            # Create request object for parsing
            request = ChatCompletionRequest(
                messages=[{"role": "user", "content": test_case['message']}],
                model="qwen2.5-72b-instruct",
                tools=tools
            )
            
            # Parse tool calls
            result = tool_manager.extract_tool_calls(generated_text, request)
            
            print(f"Parsing Results:")
            print(f"Tool calls detected: {result['tools_called']}")
            
            if result['tools_called']:
                print(f"Parser used: {result['parser_used']}")
                
                if result['content']:
                    print(f"Text content: {result['content']}")
                
                print("Tool call details:")
                for j, tool_call in enumerate(result['tool_calls'], 1):
                    print(f"  {j}. Function: {tool_call.function.name}")
                    args = json.loads(tool_call.function.arguments)
                    print(f"     Arguments: {json.dumps(args, indent=6, ensure_ascii=False)}")
                
                # Simulate tool execution
                tool_results = simulate_tool_execution(result)
                print(f"\nTool Execution Results:\n{tool_results}")
                
            else:
                print("No tool calls detected, contains only plain text response")
                if result['content']:
                    print(f"Content: {result['content']}")
            
        except Exception as e:
            print(f"Generation error: {e}")
        
        print("\n" + "=" * 80 + "\n")


def test_streaming_tool_calling(model_path: str):
    """Test streaming tool calling"""
    
    print("=== Streaming Tool Calling Test ===")
    
    # Streaming implementation can be added here
    # TensorRT-LLM streaming API may require different implementation approach
    print("Streaming test needs to be implemented based on specific TensorRT-LLM streaming API")


def main():
    parser = argparse.ArgumentParser(description="Qwen2.5-72B-Instruct Tool Calling Test")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="Qwen/Qwen2.5-72B-Instruct",
        help="Qwen model path or HuggingFace model ID"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=512,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=8,
        help="Number of GPUs for tensor parallelism (default: 8)"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization ratio (default: 0.9)"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Test streaming tool calling"
    )
    
    args = parser.parse_args()
    
    print("🔧 Qwen2.5-72B-Instruct Tool Calling Test")
    print(f"Model path: {args.model_path}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Tensor parallel size: {args.tensor_parallel_size}")
    print(f"GPU memory utilization: {args.gpu_memory_utilization}")
    print()
    
    try:
        if args.streaming:
            test_streaming_tool_calling(args.model_path)
        else:
            test_qwen_tool_calling(
                args.model_path, 
                args.max_tokens,
                args.tensor_parallel_size,
                args.gpu_memory_utilization
            )
            
        print("Test completed!")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 