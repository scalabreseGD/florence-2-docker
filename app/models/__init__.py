from typing import Literal, Optional, List, Union

from pydantic import BaseModel, Field

FLORENCE_MODELS = ("Florence-2-base", "Florence-2-large", "Florence-2-base-ft", "Florence-2-large-ft")
FLORENCE_PROMPTS = ("<CAPTION>",
                    "<DETAILED_CAPTION>",
                    "<MORE_DETAILED_CAPTION>",
                    "<OD>",
                    "<DENSE_REGION_CAPTION>",
                    "<REGION_PROPOSAL>",
                    "<CAPTION_TO_PHRASE_GROUNDING>",
                    "<REFERRING_EXPRESSION_SEGMENTATION>",
                    "<REGION_TO_SEGMENTATION>",
                    "<OPEN_VOCABULARY_DETECTION>",
                    "<REGION_TO_CATEGORY>",
                    "<REGION_TO_DESCRIPTION>",
                    "<OCR>",
                    "<OCR_WITH_REGION>")


class PredictArgs(BaseModel):
    model: Literal[FLORENCE_MODELS] = Field(...,
                                            description="The models to use between these values\n" + '\n'.join(
                                                FLORENCE_MODELS))
    task: Literal[FLORENCE_PROMPTS] = Field(...,
                                            description="The task to execute between these values\n" + '\n'.join(
                                                FLORENCE_PROMPTS))
    text: Optional[str] = Field('', description="An additional input to give to the model")
    images: Optional[List[str]] = Field(...,
                                        description="The images to predict in base64 or the path of the images to load")
    video: Optional[str] = Field(..., description="The path of the video to predict")
    scale_factor: Optional[float] = Field(1, description="The scale factor of the media to reduce the memory")
    batch_size: Optional[int] = Field(20, description="The batch for the frames")
    start_second: Optional[int] = Field(0, description="The starting frame for the prediction")
    end_second: Optional[int] = Field(None, description="The end frame for the prediction")


class PredictResponse(BaseModel):
    response: Optional[dict] = Field(..., description="The output from Florence depending on the task type")
