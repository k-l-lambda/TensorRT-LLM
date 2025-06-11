# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: TensorRT-LLM Tool Parsers

import json
import re
from typing import Any, Dict, List, Optional

from .abstract_tool_parser import (
    ExtractedToolCallInformation,
    ToolParser,
    ToolParserManager
)
from ..openai_protocol import ChatCompletionRequest


@ToolParserManager.register_module(["mistral", "mixtral"])
class MistralToolParser(ToolParser):
    """
    Tool parser for Mistral format.
    
    Expected format:
    [TOOL_CALLS] [{"name": "function_name", "arguments": {"param1": "value1"}}]
    """
    
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        
        # Mistral specific tokens
        self.tool_calls_start_token = "[TOOL_CALLS]"
        self.tool_calls_regex = re.compile(r'\[TOOL_CALLS\]\s*(\[.*?\])', re.DOTALL)
        
    def can_parse(self, text: str) -> bool:
        """Check if text contains Mistral-style tool calls."""
        return self.tool_calls_start_token in text
    
    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from complete Mistral format output."""
        
        if not self.can_parse(model_output):
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output
            )
        
        try:
            # Find tool calls section
            matches = self.tool_calls_regex.findall(model_output)
            
            if not matches:
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output
                )
            
            # Parse the JSON array
            tool_calls_json = matches[0]
            parsed_calls = self._safe_json_parse(tool_calls_json)
            
            if not parsed_calls or not isinstance(parsed_calls, list):
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output
                )
            
            # Convert to tool calls
            tool_calls = []
            for call_data in parsed_calls:
                if isinstance(call_data, dict) and "name" in call_data:
                    tool_call = self._create_tool_call(
                        name=call_data["name"],
                        arguments=call_data.get("arguments", {})
                    )
                    tool_calls.append(tool_call)
            
            if not tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output
                )
            
            # Extract content before tool calls
            tool_calls_start = model_output.find(self.tool_calls_start_token)
            content = model_output[:tool_calls_start].strip() if tool_calls_start > 0 else None
            
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content
            )
            
        except Exception as e:
            print(f"Error parsing Mistral tool calls: {e}")
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output
            )
    
    def extract_tool_calls_streaming(
        self,
        delta_text: str,
        full_text: str,
        request: ChatCompletionRequest
    ) -> Optional[Dict[str, Any]]:
        """Extract tool calls from streaming Mistral output."""
        
        # Update buffer
        self.tool_call_buffer += delta_text
        
        # Check if we're starting tool calls
        if self.tool_calls_start_token in delta_text:
            return {"type": "tool_calls_start"}
        
        # Check if we have complete tool calls
        if self.tool_calls_start_token in self.tool_call_buffer and ']' in self.tool_call_buffer:
            matches = self.tool_calls_regex.findall(self.tool_call_buffer)
            
            if matches:
                tool_calls_json = matches[0]
                parsed_calls = self._safe_json_parse(tool_calls_json)
                
                if parsed_calls and isinstance(parsed_calls, list):
                    tool_calls = []
                    for call_data in parsed_calls:
                        if isinstance(call_data, dict) and "name" in call_data:
                            tool_call = self._create_tool_call(
                                name=call_data["name"],
                                arguments=call_data.get("arguments", {})
                            )
                            tool_calls.append(tool_call)
                    
                    if tool_calls:
                        # Reset buffer
                        self.tool_call_buffer = ""
                        
                        return {
                            "type": "tool_calls_complete",
                            "tool_calls": tool_calls
                        }
        
        # Return content if not in tool calls mode
        if self.tool_calls_start_token not in self.tool_call_buffer:
            return {"type": "content", "content": delta_text}
        
        return None
    
    def format_tools_for_prompt(self, tools: List[Any]) -> str:
        """Format tools for Mistral prompt template."""
        if not tools:
            return ""
        
        formatted_tools = super().format_tools_for_prompt(tools)
        
        tools_text = "Available tools:\n"
        
        for tool in formatted_tools:
            tools_text += f"- {tool['name']}: {tool['description']}\n"
            if tool.get('parameters'):
                tools_text += f"  Parameters: {json.dumps(tool['parameters'])}\n"
        
        tools_text += ("\nTo use tools, respond with:\n"
                      "[TOOL_CALLS] ["
                      '{"name": "function_name", "arguments": {"param1": "value1"}}'
                      "]\n\n")
        
        return tools_text 