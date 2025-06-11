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


@ToolParserManager.register_module(["qwen", "qwen2", "qwen25", "qwq"])
class QwenToolParser(ToolParser):
    """
    Tool parser for Qwen format.
    
    Expected formats:
    1. Standard JSON:
       {"name": "function_name", "arguments": {"param1": "value1"}}
    
    2. QwQ reasoning format:
       <|thinking|>
       Some reasoning content...
       </|thinking|>
       
       {"name": "function_name", "arguments": {"param1": "value1"}}
    """
    
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        
        # Qwen specific tokens
        self.thinking_start_token = "<|thinking|>"
        self.thinking_end_token = "</|thinking|>"
        
        # Regex patterns
        self.thinking_regex = re.compile(
            rf"{re.escape(self.thinking_start_token)}(.*?){re.escape(self.thinking_end_token)}",
            re.DOTALL
        )
        
        # JSON pattern for tool calls
        self.json_pattern = re.compile(r'\{[^{}]*"name"[^{}]*\}', re.DOTALL)
    
    def can_parse(self, text: str) -> bool:
        """Check if text contains Qwen-style tool calls."""
        # Check for thinking tokens
        if self.thinking_start_token in text:
            return True
        
        # Check for JSON tool calls
        if self.json_pattern.search(text):
            return True
            
        # Check for standard JSON format
        stripped = text.strip()
        if (stripped.startswith('{') and '"name"' in stripped and 
            '"arguments"' in stripped):
            return True
            
        return False
    
    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from complete Qwen format output."""
        
        if not self.can_parse(model_output):
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output
            )
        
        try:
            # Handle thinking format first
            if self.thinking_start_token in model_output:
                return self._parse_thinking_format(model_output)
            
            # Handle standard JSON format
            return self._parse_json_format(model_output)
            
        except Exception as e:
            print(f"Error parsing Qwen tool calls: {e}")
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output
            )
    
    def _parse_thinking_format(self, text: str) -> ExtractedToolCallInformation:
        """Parse QwQ thinking format with reasoning."""
        
        # Extract thinking content
        thinking_matches = self.thinking_regex.findall(text)
        thinking_content = thinking_matches[0] if thinking_matches else ""
        
        # Remove thinking sections to find tool calls
        text_without_thinking = self.thinking_regex.sub("", text).strip()
        
        # Look for JSON tool calls after thinking
        tool_calls = []
        json_matches = self.json_pattern.findall(text_without_thinking)
        
        for json_str in json_matches:
            parsed_call = self._safe_json_parse(json_str)
            if parsed_call and "name" in parsed_call:
                tool_call = self._create_tool_call(
                    name=parsed_call["name"],
                    arguments=parsed_call.get("arguments", {})
                )
                tool_calls.append(tool_call)
        
        if tool_calls:
            # Extract content before first tool call (excluding thinking)
            first_json_pos = text_without_thinking.find('{')
            content = text_without_thinking[:first_json_pos].strip() if first_json_pos > 0 else None
            
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content
            )
        
        # If no tool calls found, return thinking content as regular content
        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=text_without_thinking or text
        )
    
    def _parse_json_format(self, text: str) -> ExtractedToolCallInformation:
        """Parse standard JSON format tool calls."""
        
        tool_calls = []
        
        # Try parsing entire text as JSON first
        stripped_text = text.strip()
        if stripped_text.startswith('{'):
            parsed_call = self._safe_json_parse(stripped_text)
            if parsed_call and "name" in parsed_call:
                tool_call = self._create_tool_call(
                    name=parsed_call["name"],
                    arguments=parsed_call.get("arguments", {})
                )
                tool_calls.append(tool_call)
        
        # If that didn't work, look for JSON patterns in the text
        if not tool_calls:
            json_matches = self.json_pattern.findall(text)
            for json_str in json_matches:
                parsed_call = self._safe_json_parse(json_str)
                if parsed_call and "name" in parsed_call:
                    tool_call = self._create_tool_call(
                        name=parsed_call["name"],
                        arguments=parsed_call.get("arguments", {})
                    )
                    tool_calls.append(tool_call)
        
        if tool_calls:
            # Extract content before first tool call
            first_json_pos = text.find('{')
            content = text[:first_json_pos].strip() if first_json_pos > 0 else None
            
            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content
            )
        
        return ExtractedToolCallInformation(
            tools_called=False,
            tool_calls=[],
            content=text
        )
    
    def extract_tool_calls_streaming(
        self,
        delta_text: str,
        full_text: str,
        request: ChatCompletionRequest
    ) -> Optional[Dict[str, Any]]:
        """Extract tool calls from streaming Qwen output."""
        
        # Update buffer
        self.tool_call_buffer += delta_text
        
        # Check for thinking start
        if self.thinking_start_token in delta_text:
            return {"type": "thinking_start"}
        
        # Check for thinking end
        if self.thinking_end_token in delta_text:
            return {"type": "thinking_end"}
        
        # Check for JSON tool calls
        if '{' in self.tool_call_buffer and '}' in self.tool_call_buffer:
            # Try to extract complete JSON
            json_matches = self.json_pattern.findall(self.tool_call_buffer)
            
            for json_str in json_matches:
                parsed_call = self._safe_json_parse(json_str)
                if parsed_call and "name" in parsed_call:
                    # Reset buffer
                    self.tool_call_buffer = ""
                    
                    return {
                        "type": "tool_call_complete",
                        "tool_call": self._create_tool_call(
                            name=parsed_call["name"],
                            arguments=parsed_call.get("arguments", {})
                        )
                    }
        
        # Return content if not in special mode
        return {"type": "content", "content": delta_text}
    
    def format_tools_for_prompt(self, tools: List[Any]) -> str:
        """Format tools for Qwen prompt template."""
        if not tools:
            return ""
        
        formatted_tools = super().format_tools_for_prompt(tools)
        
        tools_text = "# Tools\n\n"
        tools_text += "You have access to the following tools:\n\n"
        
        for i, tool in enumerate(formatted_tools, 1):
            tools_text += f"## Tool {i}: {tool['name']}\n"
            tools_text += f"**Description**: {tool['description']}\n"
            if tool.get('parameters'):
                tools_text += f"**Parameters**:\n```json\n{json.dumps(tool['parameters'], indent=2)}\n```\n"
            tools_text += "\n"
        
        tools_text += ("When you need to call a function, output JSON in this format:\n"
                      "```json\n"
                      '{"name": "function_name", "arguments": {"param1": "value1"}}\n'
                      "```\n\n"
                      "For QwQ models, you can use reasoning:\n"
                      f"{self.thinking_start_token}\n"
                      "Your reasoning here...\n"
                      f"{self.thinking_end_token}\n\n"
                      '{"name": "function_name", "arguments": {"param1": "value1"}}\n\n')
        
        return tools_text 