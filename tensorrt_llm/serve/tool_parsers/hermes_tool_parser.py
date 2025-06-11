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


@ToolParserManager.register_module(["hermes", "hermes2", "nous"])
class HermesToolParser(ToolParser):
    """
    Tool parser for Hermes/Nous format.
    
    Expected format:
    <tool_call>
    {"name": "function_name", "arguments": {"param1": "value1", "param2": "value2"}}
    </tool_call>
    """
    
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        
        # Tool call patterns for Hermes format
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"
        
        # Regex pattern to match tool calls
        self.tool_call_regex = re.compile(
            rf"{re.escape(self.tool_call_start_token)}(.*?){re.escape(self.tool_call_end_token)}|"
            rf"{re.escape(self.tool_call_start_token)}(.*?)$",
            re.DOTALL
        )
    
    def can_parse(self, text: str) -> bool:
        """Check if text contains Hermes-style tool calls."""
        return self.tool_call_start_token in text
    
    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from complete Hermes format output."""
        
        # Check if there are any tool call patterns
        if not self.can_parse(model_output):
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output
            )
        
        try:
            # Find all tool call matches
            function_call_tuples = self.tool_call_regex.findall(model_output)
            
            if not function_call_tuples:
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output
                )
            
            # Parse JSON from tool calls
            tool_calls = []
            for match in function_call_tuples:
                json_str = match[0] if match[0] else match[1]
                if json_str.strip():
                    parsed_call = self._safe_json_parse(json_str)
                    if parsed_call and "name" in parsed_call:
                        tool_call = self._create_tool_call(
                            name=parsed_call["name"],
                            arguments=parsed_call.get("arguments", {})
                        )
                        tool_calls.append(tool_call)
            
            if not tool_calls:
                return ExtractedToolCallInformation(
                    tools_called=False,
                    tool_calls=[],
                    content=model_output
                )
            
            # Extract content before first tool call
            content_end = model_output.find(self.tool_call_start_token)
            content = model_output[:content_end].strip() if content_end > 0 else None
            
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content
            )
            
        except Exception as e:
            # Log error and return as regular content
            print(f"Error parsing Hermes tool calls: {e}")
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
        """Extract tool calls from streaming Hermes output."""
        
        # Update buffer with new text
        self.tool_call_buffer += delta_text
        
        # Check if we're starting a tool call
        if self.tool_call_start_token in delta_text and self.current_tool_id == -1:
            self.current_tool_id += 1
            self.current_tool_name_sent = False
            return {"type": "tool_call_start"}
        
        # Check if we're ending a tool call
        if self.tool_call_end_token in self.tool_call_buffer:
            # Extract complete tool call
            tool_calls = self._extract_json_from_tags(
                self.tool_call_buffer,
                self.tool_call_start_token,
                self.tool_call_end_token
            )
            
            if tool_calls:
                parsed_call = self._safe_json_parse(tool_calls[0])
                if parsed_call and "name" in parsed_call:
                    # Reset state for next tool call
                    self.tool_call_buffer = ""
                    self.current_tool_id = -1
                    self.current_tool_name_sent = False
                    
                    return {
                        "type": "tool_call_complete",
                        "tool_call": self._create_tool_call(
                            name=parsed_call["name"],
                            arguments=parsed_call.get("arguments", {})
                        )
                    }
        
        # Return normal content if not in tool call
        if self.current_tool_id == -1:
            return {"type": "content", "content": delta_text}
        
        return None
    
    def format_tools_for_prompt(
        self,
        tools: List[Any]
    ) -> str:
        """Format tools for Hermes prompt template."""
        if not tools:
            return ""
        
        formatted_tools = super().format_tools_for_prompt(tools)
        
        tools_text = "You have access to the following functions:\n\n"
        for tool in formatted_tools:
            tools_text += f"Function: {tool['name']}\n"
            tools_text += f"Description: {tool['description']}\n"
            if tool.get('parameters'):
                tools_text += f"Parameters: {json.dumps(tool['parameters'], indent=2)}\n"
            tools_text += "\n"
        
        tools_text += ("When you need to call a function, use this exact format:\n"
                      "<tool_call>\n"
                      '{"name": "function_name", "arguments": {"param1": "value1"}}\n'
                      "</tool_call>\n\n")
        
        return tools_text 