import asyncio
import sys
from dacite import from_dict

sys.path.append("./")

from common.together_web3.computer import ImageModelInferenceRequest
from common.together_web3.together import TogetherWeb3

async def test():
    together_web3 = TogetherWeb3()
    result = await together_web3.language_model_inference(
        from_dict(
            data_class=ImageModelInferenceRequest,
            data={
                "model": "StableDiffusion",
                "prompt": "test",
            }
        ),
    )
    print("result", result)

if __name__=="__main__":
    asyncio.run(test())

