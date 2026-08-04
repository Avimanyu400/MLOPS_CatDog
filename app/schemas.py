from pydantic import BaseModel, Field
from typing import List

class ImageInferenceRequest(BaseModel):
    image: List[List[List[float]]] = Field(
        ..., 
        description="Image tensor data of shape [3, height, width]"
    )

class InferenceResponse(BaseModel):
    predicted_class_index: int
    predicted_class_name: str
    probabilities: List[float]