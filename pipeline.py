import asyncio
import uuid
class Pipeline:
    def __init__(self, config,input):
        self.name = None
        self.config = config
        self.input = input
        self.steps = []
        self.outputs = []

    def add_steps(self, steps):
        self.steps.extend(steps)

    async def run(self):
        run_id = uuid.uuid4()
        result = await asyncio.gather(*[step.run(step.input,self.input,run_id) for step in self.steps])
        self.outputs = await result
        return self.outputs



class Step:
    def __init__(self, config,input, function, parameters):
        self.name = None
        self.config = config
        self.steps = []
        self.function = function
        self.parameters = parameters
        self.output = None
        self.input = input


    def add_steps(self, steps):
        self.steps.extend(steps)

    async def run(self,input_name:str,input_value, run_id:uuid.uuid4()):
        self.output = await self.function(*self.parameters.args 
                      , **self.parameters.kwargs | {input_name:input_value})

        if self.steps:
            result = await asyncio.gather(*[step.run(step.input,self.output,run_id) for step in self.steps])
            return result
        return (self.name,self.output)