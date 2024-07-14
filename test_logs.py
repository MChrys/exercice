import asyncio
import logging
from pipeline import Pipeline, Step, Parameters


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def func(val: int):
    logger.info(f"func called with value: {val}")
    result = val * 2
    logger.info(f"func returns: {result}")
    return result

async def async_func(val: int):
    logger.info(f"async_func called with value: {val}")
    await asyncio.sleep(1)
    result = val + 10
    logger.info(f"async_func returns: {result}")
    return result

async def main():
    pipe = Pipeline()
    
    step1 = Step(func)
    step2 = Step(async_func)
    step3 = Step(func)
    
    pipe >> step1 >> step2 >> step3
    
    result = await pipe.start(2)
    
    logger.info(f"pipeline result: {result}")
    print()
if __name__ == "__main__":
    asyncio.run(main())