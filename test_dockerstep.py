from workflows.pipeline import Pipeline, DockerStep, Step
from workflows.nlp_steps import transcribe_empty
import asyncio
from workflows.utils import logx
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
async def main():
    dockertest = Pipeline()
    docker_step = DockerStep(transcribe_empty)
    normal_step = Step(transcribe_empty)
    dockertest >> docker_step
    try:
        pipelinefuture = dockertest.start("data/transcribe_encoded.json")
        logger.info("docker_step_test :", len(await docker_step.output))
        #logger.info("normal_step_test :", len(await normal_step.output))
        result =await pipelinefuture


    except asyncio.CancelledError as e:
        logger.info(f"Pipeline execution was cancelled {e}")
    except Exception as e:
        logger.info(f"An error occurred: {e}")
    finally:
 
        dockertest.cancel_steps()
        pipelinefuture.cancel()
if __name__ == "__main__":
    asyncio.run(main())
