import asyncio
import logging
from workflows.pipeline import Pipeline, Step,Parameters
from workflows.nlp_steps import transcribe_empty
from opentelemetry import trace

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def func(val: int):
    span = trace.get_current_span()
    span.add_event(f"func called with value: {val}")
    result = val 
    span.add_event(f"func returns: {result}")
    return result

async def async_func(val: int):
    span = trace.get_current_span()
    span.add_event(f"async_func called with value: {val}")
    await asyncio.sleep(1)
    result = val 
    span.add_event(f"async_func returns: {result}")
    return result

async def main():
    pipe = Pipeline()
    
    step1 = Step(func)
    step2 = Step(async_func)
    step3 = Step(func)
    #step4 = Step(transcribe_empty)

    pipe >> step1 >> step2 >> step3 #>> step4
    
    try:

        pipeline_future = asyncio.create_task(pipe.start("data/transcribe_encoded.json"))
        

        logger.info(f"step1 output: ======> {await step1.output}")
        logger.info(f"step2 output: ======> {await step2.output}")
        logger.info(f"step3 output: ======> {await step3.output}")
        

        result = await pipeline_future

        for step_name, value in result.items():
            logger.info(f"{step_name} final output: ======> {value}")
    except asyncio.CancelledError as e:
        logger.info(f"Pipeline execution was cancelled: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
 
        for step in [step1, step2, step3]:
            step.cancel_output()
    print()
if __name__ == "__main__":
    asyncio.run(main())