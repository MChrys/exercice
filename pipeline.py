import asyncio

class Pipeline:
    def __init__(self, config,input):
        self.config = config
        self.input = input
        self.steps = []
        self.outputs = []

    def add_steps(self, steps):
        self.steps.append(steps)

    async def run(self):
        result = await asyncio.gather(*[step.run(step.input,self.input) for step in self.steps])
        self.outputs = await result
        return self.outputs



class Step:
    def __init__(self, config,input, function, parameters):
        self.config = config
        self.steps = []
        self.function = function
        self.parameters = parameters
        self.output = None
        self.input = input


    def add_steps(self, steps):
        self.steps.append(steps)

    async def run(self,input_name:str,input_value):
        self.output = self.function(*self.parameters.args 
                      , **self.parameters.kwargs | {input_name:input_value})

        if self.steps:
            result = await asyncio.gather(*[step.run(step.input,self.output) for step in self.steps])
            return result
        return await self.output