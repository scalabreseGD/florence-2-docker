from datetime import datetime
from typing import Literal, Optional, Tuple, List

from pydantic import BaseModel, Field, field_validator, conlist

from utils import parse_tags_from_content
from models.base import FLORENCE_MODELS, FLORENCE_PROMPTS


class ChatCompletionMessage(BaseModel):
    role: Literal['user', 'assistant'] = Field(..., description="The role of the messages author. Accepted only user")
    content: str = Field(f"<task>{FLORENCE_PROMPTS[2]}</task>Text<text></text><media_path>/my/asset/path.png</media_path>",
                         description="""
                         If role is user: The content of the message with the format <task></task><text></text><media_path></media_path>
                         If role is assistant: The content of the message as json string""")

    @field_validator('content', mode='before')
    @classmethod
    def check_tags(cls, content: str, info) -> str:
        role = info.data['role']
        if role == 'assistant':
            return content
        extracted_tags: Tuple[str, Optional[str], bool] = parse_tags_from_content(content)
        for tag, value, is_mandatory in extracted_tags:
            if value is None and is_mandatory:
                raise ValueError(f'Missing mandatory tag {tag}')
        return content


class ChatCompletionChoice(BaseModel):
    index: int = Field(..., description="The index of the choice")
    message: Optional[ChatCompletionMessage] = Field(..., description="The message from the assistant")
    delta: Optional[ChatCompletionMessage] = Field(..., description="The choice delta in case or stream")


class ChatCompletionModel(BaseModel):
    model: Literal[FLORENCE_MODELS] = Field(...,
                                            description="The models to use between these values\n" + '\n'.join(
                                                FLORENCE_MODELS))
    messages: conlist(ChatCompletionMessage, max_length=1, min_length=1) = Field(...,
                                                                                       description="List of messages of role-content. Max size 1")

    stream: bool = Field(..., description="Whether the messages are streaming")
    # max_completion_tokens: Optional[int] = Field(1024, description="The maximum number of tokens to complete. "
    #                                                                "Can't be more than 1024")


class ChatCompletionResponse(BaseModel):
    id: str = Field(..., description="The id of the response")
    object: Literal["chat.completion", "chat.completion.chunk"] = Field("chat.completion",
                                                                        description="The object of the response")
    created: int = Field(..., description="The date and time the object was created in timestamp")
    model: Literal[FLORENCE_MODELS] = Field(...,
                                            description="The models to use between these values\n" + '\n'.join(
                                                FLORENCE_MODELS))
    choices: List[ChatCompletionChoice] = Field(..., description="List of choices. Max size 1")
