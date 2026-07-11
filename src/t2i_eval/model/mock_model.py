import time

from PIL import Image

from ..core.model import BaseModel
from ..core.schema import GenerationConfig, GenerationResult


class MockModel(BaseModel):
    def load(self):
        pass

    def unload(self):
        pass

    def generate(self, config: GenerationConfig) -> GenerationResult:
        time.sleep(1)
        img = Image.new("RGB", (config.width, config.height), color=(73, 109, 137))
        return GenerationResult(images=[img], debug_info={"prompt": config.prompt})
