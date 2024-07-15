import asyncio
import logging
from pipeline import Pipeline, Step, Parameters
from opentelemetry import trace

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def func(val: int):
    span = trace.get_current_span()
    span.add_event(f"func called with value: {val}")
    result = val * 2
    span.add_event(f"func returns: {result}")
    return result

async def async_func(val: int):
    span = trace.get_current_span()
    span.add_event(f"async_func called with value: {val}")
    await asyncio.sleep(1)
    result = val + 10
    span.add_event(f"async_func returns: {result}")
    return result

async def main():
    pipe = Pipeline()
    
    step1 = Step(func)
    step2 = Step(async_func)
    step3 = Step(func)
    pipe >> step1 >> step2 >> step3
    
    try:

        pipeline_future = asyncio.create_task(pipe.start(2))
        

        logger.info(f"step1 output: ======> {await step1.output}")
        logger.info(f"step2 output: ======> {await step2.output}")
        logger.info(f"step3 output: ======> {await step3.output}")
        

        result = await pipeline_future
        logger.info(f"pipeline result: {result}")
        logger.info(result)
        for step_name, value in result.items():
            logger.info(f"{step_name} final output: ======> {value}")
    except asyncio.CancelledError:
        logger.info("Pipeline execution was cancelled")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
 
        for step in [step1, step2, step3]:
            step.cancel_output()
    print()
if __name__ == "__main__":
    asyncio.run(main())