#!/usr/bin/env python3
"""
Kimik2 Multi-Tool Calling Test Script
Based on the discovered correct format, test multiple tool calling capabilities
"""

import requests
import json
import time

def test_tools(name, data, base_url="http://0.0.0.0:8000"):
    """Test tool calling"""
    print(f"\nTesting: {name}")
    print("-" * 60)
    
    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=data,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("SUCCESS")
            print(f"Assistant Response: {result['choices'][0]['message']['content']}")
            
            tool_calls = result['choices'][0]['message'].get('tool_calls', [])
            if tool_calls:
                print(f"Tool Calls Count: {len(tool_calls)}")
                for i, call in enumerate(tool_calls, 1):
                    print(f"  {i}. {call['function']['name']}({call['function']['arguments']})")
            else:
                print("No tool calls made")
            return True
        else:
            print("FAILED")
            error_info = response.json() if response.headers.get('content-type') == 'application/json' else response.text
            print(f"Error: {error_info}")
            return False
            
    except Exception as e:
        print(f"EXCEPTION: {str(e)}")
        return False

def main():
    print("Kimik2 Multi-Tool Calling Test")
    print("=" * 70)
    
    # Define multiple tools for testing
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather information for a specific city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string", "description": "City name"},
                        "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature units"}
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function", 
            "function": {
                "name": "search_web",
                "description": "Search for information on the internet",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search keywords"},
                        "limit": {"type": "integer", "description": "Number of results limit", "default": 5}
                    },
                    "required": ["query"]
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
                        "expression": {"type": "string", "description": "Mathematical expression"},
                        "precision": {"type": "integer", "description": "Decimal places", "default": 2}
                    },
                    "required": ["expression"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "translate_text",
                "description": "Translate text from one language to another",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to translate"},
                        "from_lang": {"type": "string", "description": "Source language"},
                        "to_lang": {"type": "string", "description": "Target language"}
                    },
                    "required": ["text", "to_lang"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_time",
                "description": "Get current time for a specific timezone",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timezone": {"type": "string", "description": "Timezone like Asia/Shanghai"},
                        "format": {"type": "string", "description": "Time format", "default": "%Y-%m-%d %H:%M:%S"}
                    }
                }
            }
        }
    ]
    
    # Test cases
    test_cases = [
        {
            "name": "Single Tool Call - Weather Query",
            "data": {
                "model": "kimik2",
                "messages": [
                    {"role": "user", "content": "What's the weather like in Shanghai now?"}
                ],
                "tools": [tools[0]],  # Only weather tool
                "max_tokens": 500
            }
        },
        {
            "name": "Dual Tool Call - Weather and Search",
            "data": {
                "model": "kimik2",
                "messages": [
                    {"role": "user", "content": "Check Beijing weather and search for today's important news"}
                ],
                "tools": tools[:2],  # Weather and search tools
                "max_tokens": 800
            }
        },
        {
            "name": "Multi-Tool Call - Calculation Related",
            "data": {
                "model": "kimik2", 
                "messages": [
                    {"role": "user", "content": "Calculate (45.6 * 3.2) + 78.9, then search for any special meaning of this number"}
                ],
                "tools": [tools[1], tools[2]],  # Search and calculate tools
                "max_tokens": 600
            }
        },
        {
            "name": "All Tools Available - Complex Request",
            "data": {
                "model": "kimik2",
                "messages": [
                    {"role": "user", "content": "I need: 1) Check Shenzhen weather 2) Calculate 100*0.85 3) Get Beijing time 4) Search for 'latest AI developments'"}
                ],
                "tools": tools,  # All tools
                "max_tokens": 1000
            }
        },
        {
            "name": "Tool Selection Test - Translation",
            "data": {
                "model": "kimik2",
                "messages": [
                    {"role": "user", "content": "Please translate 'Hello, how are you today?' to Chinese"}
                ],
                "tools": tools,
                "max_tokens": 300
            }
        },
        {
            "name": "No Relevant Tools - Regular Conversation",
            "data": {
                "model": "kimik2", 
                "messages": [
                    {"role": "user", "content": "Hello, please introduce yourself"}
                ],
                "tools": tools,
                "max_tokens": 400
            }
        },
        {
            "name": "Sequential Tool Calls Test",
            "data": {
                "model": "kimik2",
                "messages": [
                    {"role": "user", "content": "First check Shanghai weather, if it's sunny then calculate 25*4, if it's rainy then search for 'rainy day travel tips'"}
                ],
                "tools": tools,
                "max_tokens": 800
            }
        },
        {
            "name": "Math Word Problem",
            "data": {
                "model": "kimik2",
                "messages": [
                    {"role": "user", "content": "A store has 150 items. If 30% are sold in the morning and 45% of the remaining are sold in the afternoon, how many items are left?"}
                ],
                "tools": [tools[2]],  # Only calculate tool
                "max_tokens": 500
            }
        },
        {
            "name": "Multi-Language Translation Chain",
            "data": {
                "model": "kimik2",
                "messages": [
                    {"role": "user", "content": "Translate 'Good morning' to Spanish, then translate the result to French"}
                ],
                "tools": [tools[3]],  # Only translation tool
                "max_tokens": 400
            }
        },
        {
            "name": "Time Zone Comparison",
            "data": {
                "model": "kimik2",
                "messages": [
                    {"role": "user", "content": "What time is it now in New York, London, and Tokyo?"}
                ],
                "tools": [tools[4]],  # Only time tool
                "max_tokens": 600
            }
        }
    ]
    
    successful_tests = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*20} Test {i}/{len(test_cases)} {'='*20}")
        success = test_tools(test_case["name"], test_case["data"])
        if success:
            successful_tests.append(test_case["name"])
        time.sleep(2)  # Avoid too frequent requests
    
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)
    
    if successful_tests:
        print("Successful Tests:")
        for test in successful_tests:
            print(f"  - {test}")
    else:
        print("No successful tests")
        
    print(f"\nTotal Tests: {len(test_cases)} scenarios")
    print(f"Successful: {len(successful_tests)}")
    print(f"Failed: {len(test_cases) - len(successful_tests)}")
    print(f"Success Rate: {len(successful_tests)/len(test_cases)*100:.1f}%")

if __name__ == "__main__":
    main()