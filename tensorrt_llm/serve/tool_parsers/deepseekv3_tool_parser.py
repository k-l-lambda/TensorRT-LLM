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


@ToolParserManager.register_module(["deepseek", "deepseek_v3", "deepseekv3"])
class DeepSeekV3ToolParser(ToolParser):
    """
    Tool parser for DeepSeek V3 format.
    
    DeepSeek V3 uses a specific format with special tokens:
    <｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function.function_name
    ```json
    {"param": "value"}
    ```<｜tool▁call▁end｜><｜tool▁calls▁end｜>
    """
    
    def __init__(self, tokenizer):
        super().__init__(tokenizer)
        
        # DeepSeek V3 special tokens
        self.tool_calls_start_token = "<｜tool▁calls▁begin｜>"
        self.tool_calls_end_token = "<｜tool▁calls▁end｜>"
        self.tool_call_start_token = "<｜tool▁call▁begin｜>"
        self.tool_call_end_token = "<｜tool▁call▁end｜>"
        self.tool_sep_token = "<｜tool▁sep｜>"
        
        # Regex patterns for parsing
        self.tool_call_regex = re.compile(
            r"<｜tool▁call▁begin｜>(?P<type>.*?)<｜tool▁sep｜>(?P<function_name>.*?)\n```json\n(?P<function_arguments>.*?)\n```<｜tool▁call▁end｜>",
            re.DOTALL
        )
        
        # Pattern for incomplete tool calls (streaming)
        self.partial_tool_call_regex = re.compile(
            r"<｜tool▁call▁begin｜>(?P<type>.*?) (?P<function_name>.*?)(?:\n```json\n(?P<function_arguments>.*?))?",
            re.DOTALL
        )
    
    def can_parse(self, text: str) -> bool:
        """Check if text contains DeepSeek V3 tool call format."""
        return self.tool_calls_start_token in text or self.tool_call_start_token in text
    
    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        """Extract tool calls from DeepSeek V3 format output."""
        
        if not self.can_parse(model_output):
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output
            )
        
        try:
            # Find all complete tool calls
            tool_call_matches = self.tool_call_regex.findall(model_output)
            
            tool_calls = []
            for match in tool_call_matches:
                tool_type, function_name, function_args = match
                
                try:
                    # Parse JSON arguments
                    if function_args.strip():
                        arguments = json.loads(function_args.strip())
                    else:
                        arguments = {}
                    
                    tool_call = self._create_tool_call(
                        name=function_name.strip(),
                        arguments=arguments
                    )
                    tool_calls.append(tool_call)
                    
                except json.JSONDecodeError:
                    # If JSON parsing fails, treat as string
                    tool_call = self._create_tool_call(
                        name=function_name.strip(),
                        arguments={"raw_args": function_args.strip()}
                    )
                    tool_calls.append(tool_call)
            
            # Extract content before tool calls
            tool_start_idx = model_output.find(self.tool_calls_start_token)
            if tool_start_idx > 0:
                content = model_output[:tool_start_idx].strip()
            else:
                content = ""
            
            # Clean up any remaining tokens from content
            content = self._clean_content(content)
            
            return ExtractedToolCallInformation(
                tools_called=len(tool_calls) > 0,
                tool_calls=tool_calls,
                content=content if content else None
            )
            
        except Exception as e:
            print(f"Error parsing DeepSeek V3 tool calls: {e}")
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output
            )
    
    def _clean_content(self, content: str) -> str:
        """Clean DeepSeek V3 special tokens from content."""
        # Remove any DeepSeek special tokens that might appear in content
        tokens_to_remove = [
            self.tool_calls_start_token,
            self.tool_calls_end_token,
            self.tool_call_start_token,
            self.tool_call_end_token,
            self.tool_sep_token
        ]
        
        for token in tokens_to_remove:
            content = content.replace(token, "")
        
        # Clean up extra whitespace
        content = re.sub(r'\n\s*\n', '\n', content).strip()
        
        return content
    
    def format_tools_for_prompt(
        self,
        tools: List[Any]
    ) -> str:
        """Format tools for DeepSeek V3 prompt template."""
        if not tools:
            return ""
        
        formatted_tools = super().format_tools_for_prompt(tools)
        
        tools_text = "You have access to the following functions. "
        tools_text += "Use them if required:\n\n"
        
        for tool in formatted_tools:
            tools_text += f"Function: {tool['name']}\n"
            tools_text += f"Description: {tool['description']}\n"
            if tool.get('parameters'):
                tools_text += f"Parameters: {json.dumps(tool['parameters'], indent=2)}\n"
            tools_text += "\n"
        
        tools_text += ("To call a function, use this format:\n"
                      f"{self.tool_calls_start_token}{self.tool_call_start_token}function{self.tool_sep_token}function_name\n"
                      "```json\n"
                      '{"param1": "value1", "param2": "value2"}\n'
                      f"```{self.tool_call_end_token}{self.tool_calls_end_token}\n\n")
        
        return tools_text
    
    def extract_tool_calls_streaming(
        self,
        delta_text: str,
        full_text: str,
        request: ChatCompletionRequest
    ) -> Optional[Dict[str, Any]]:
        """Extract tool calls from streaming DeepSeek V3 output."""
        
        # For now, use the non-streaming version on full text
        # TODO: Implement proper streaming support
        if self.can_parse(full_text):
            result = self.extract_tool_calls(full_text, request)
            if result.tools_called:
                return {
                    "type": "tool_calls",
                    "tool_calls": result.tool_calls,
                    "content": result.content
                }
        
        return {"type": "content", "content": delta_text}
    
    def reset_state(self):
        """Reset parser state for new conversations."""
        # DeepSeek V3 parser doesn't maintain state between calls
        pass 