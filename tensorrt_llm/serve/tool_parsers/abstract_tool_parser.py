# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: TensorRT-LLM Tool Parsers

import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from ..openai_protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionCall,
    ToolCall
)


class ExtractedToolCallInformation:
    """Information extracted from tool calls in model output."""
    
    def __init__(
        self,
        tools_called: bool,
        tool_calls: List[ToolCall],
        content: Optional[str] = None
    ):
        self.tools_called = tools_called
        self.tool_calls = tool_calls
        self.content = content


class ToolParserManager:
    """Manager for registering and retrieving tool parsers."""
    
    _parsers: Dict[str, type] = {}
    
    @classmethod
    def register_module(cls, names: Union[str, List[str]]):
        """Decorator to register a tool parser with one or more names."""
        if isinstance(names, str):
            names = [names]
            
        def decorator(parser_class):
            for name in names:
                cls._parsers[name] = parser_class
            return parser_class
        return decorator
    
    @classmethod
    def get_parser(cls, name: str, tokenizer) -> 'ToolParser':
        """Get a tool parser instance by name."""
        if name not in cls._parsers:
            raise ValueError(f"Unknown tool parser: {name}. Available: {list(cls._parsers.keys())}")
        return cls._parsers[name](tokenizer)


class ToolParser(ABC):
    """
    Abstract base class for tool parsers.
    
    Each tool parser handles extracting tool calls from model outputs
    in different formats (e.g., Hermes, Llama, Qwen, etc.).
    """
    
    def __init__(self, tokenizer):
        """Initialize the tool parser with a tokenizer."""
        self.tokenizer = tokenizer
        
        # State tracking for streaming
        self.current_tool_id: int = -1
        self.current_tool_name_sent: bool = False
        self.streamed_args_for_tool: List[str] = []
        self.tool_call_buffer: str = ""
        
    def reset_state(self):
        """Reset parser state (useful for streaming)."""
        self.current_tool_id = -1
        self.current_tool_name_sent = False
        self.streamed_args_for_tool = []
        self.tool_call_buffer = ""
    
    @abstractmethod
    def can_parse(self, text: str) -> bool:
        """Check if this parser can handle the given text format."""
        pass
    
    @abstractmethod 
    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest
    ) -> ExtractedToolCallInformation:
        """
        Extract tool calls from complete model output.
        
        Args:
            model_output: The complete text output from the model
            request: The original chat completion request
            
        Returns:
            ExtractedToolCallInformation containing parsed tool calls
        """
        pass
    
    def extract_tool_calls_streaming(
        self,
        delta_text: str,
        full_text: str,
        request: ChatCompletionRequest
    ) -> Optional[Dict[str, Any]]:
        """
        Extract tool calls from streaming model output.
        
        Args:
            delta_text: New text chunk
            full_text: Complete text so far
            request: The original chat completion request
            
        Returns:
            Dictionary with streaming tool call information or None
        """
        # Default implementation - can be overridden by subclasses
        return None
    
    def format_tools_for_prompt(
        self, 
        tools: List[ChatCompletionToolsParam]
    ) -> str:
        """
        Format tool definitions for inclusion in prompts.
        
        This is a helper method that can be used by specific parsers
        to format tools in their expected format.
        """
        formatted_tools = []
        for tool in tools:
            if tool.type == "function":
                func_def = tool.function
                formatted_tools.append({
                    "name": func_def.name,
                    "description": func_def.description,
                    "parameters": func_def.parameters
                })
        
        return formatted_tools
    
    def _safe_json_parse(self, text: str) -> Optional[Dict[str, Any]]:
        """Safely parse JSON text, returning None if invalid."""
        try:
            import json
            return json.loads(text.strip())
        except (json.JSONDecodeError, ValueError):
            return None
    
    def _extract_json_from_tags(
        self, 
        text: str, 
        start_tag: str, 
        end_tag: str
    ) -> List[str]:
        """Extract JSON content between specified tags."""
        pattern = rf"{re.escape(start_tag)}(.*?){re.escape(end_tag)}"
        matches = re.findall(pattern, text, re.DOTALL)
        return [match.strip() for match in matches]
    
    def _create_tool_call(
        self, 
        name: str, 
        arguments: Union[Dict[str, Any], str]
    ) -> ToolCall:
        """Create a ToolCall object from name and arguments."""
        import json
        
        if isinstance(arguments, dict):
            arguments_str = json.dumps(arguments, ensure_ascii=False)
        else:
            arguments_str = str(arguments)
            
        return ToolCall(
            type="function",
            function=FunctionCall(
                name=name,
                arguments=arguments_str
            )
        ) 