#!/usr/bin/env python3

import json
import requests
from typing import List, Dict, Any, Optional

server_url: str = "http://localhost:8000"
# model = "meta-llama/Llama-3.3-70B-Instruct"
model = "Qwen/Qwen2.5-72B-Instruct"

def get_weather(location: str, unit: str):
    return f"Getting the weather for {location} in {unit}..."
def calculate(expression: str):
    return f"Getting the result of {expression} ..."
def search_web(query: str):
    return f"Searching the web for {query}..."

function_map = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_web": search_web
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather information for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "default": "celsius"}
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Mathematical expression"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    }
]

test_cases = [
    "What's the weather like in San Francisco, CA? Use fahrenheit.",
    "Calculate 25 * 4 + 15",
    "Search for information about TensorRT-LLM optimization"
]


for i, user_message in enumerate(test_cases, 1):
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant with access to tools."
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            "tools": tools,
            "max_tokens": 1024,
            "temperature": 0.1,
            "stream": False
        }
        
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        result = response.json()
        choice = result["choices"][0]
        # tool_call = choice["message"]['tool_calls']
        # print(choice) 
        tool_call = choice["message"]['tool_calls'][0]['function']

        function_name = tool_call['name']
        arguments = json.loads(tool_call['arguments'])
        actual_function = function_map[function_name]
        result = actual_function(**arguments)
        
        print(f"Function called: {tool_call['name']}")
        print(f"Arguments: {tool_call['arguments']}")
        print(f"Result: {result}")
        print("-" * 50)
        # break
