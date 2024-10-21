from typing import Literal, Optional, List, Union

from pydantic import BaseModel, Field

FLORENCE_MODELS = ("Florence-2-base", "Florence-2-large", "Florence-2-base-ft", "Florence-2-large-ft")
FLORENCE_PROMPTS = ("<CAPTION>",
                    "<DETAILED_CAPTION>",
                    "<MORE_DETAILED_CAPTION>",
                    "<OD>",
                    "<DENSE_REGION_CAPTION>",
                    "<REGION_PROPOSAL>",
                    "<CAPTION_TO_PHRASE_GROUNDING>A black and brown dog is laying on a grass field.",
                    "<REFERRING_EXPRESSION_SEGMENTATION>a black and brown dog",
                    "<REGION_TO_SEGMENTATION><loc_312><loc_168><loc_998><loc_846>",
                    "<OPEN_VOCABULARY_DETECTION>a black and brown dog",
                    "<REGION_TO_CATEGORY><loc_312><loc_168><loc_998><loc_846>",
                    "<REGION_TO_DESCRIPTION><loc_312><loc_168><loc_998><loc_846>",
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
    images: List[str] = Field(..., description="The images to predict in base64")


class PredictResponse(BaseModel):
    task: Literal[FLORENCE_PROMPTS] = Field(...,
                                            description="The task to execute between these values\n" + '\n'.join(
                                                FLORENCE_PROMPTS))
    response: Optional[Union[str, dict]] = Field(..., description="The output from Florence depending on the task type")
