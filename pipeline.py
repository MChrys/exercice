from __future__ import annotations
import asyncio
import uuid

from varname import varname

from typing import MutableSequence, TypeVar
from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Parameters:
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)


class stepList(MutableSequence):
    _ACTUAL_  = None
    def __init__(self,liste :list = list() ):
        self.bool = False
        self._l=liste
        self.last = None
        
        self.link = None

    def __getitem__(self, key):
        return self._l[key]

    def __setitem__(self, key, value):
        self._l[key] = value

    def __call__(self,liste :list):
        self._l = liste

    def __len__(self):
        return len(self._l)

    def append(self, item):
        self._l.append(item)

    def __delitem__(self, key):
        del self._l[key]

    def insert(self, key,value):
        self._l(key, value)

    def __bool__(self):
        bool(self.bool)

    def __rshift__(self, other:Step|stepList):


        if self.link is None:
            self.link = self
        other.link =self.link
        if isinstance(other, Step):
            for o in self :
                o >> other
        elif isinstance(other, stepList):
            for o in self :
                for t in other:
                    o >> t

        return other
    
    def iter(self):
        return iter(self._l)


    def flip(self):
        self.bool = not self.bool

    def __add__(self,other):
        if isinstance(other, Step):
            print('add Test  to StepList')
            self._l.append(other)
            return self
class Pipeline:
    def __init__(self,inherite =False):
        #self.name = self.set_varname()
        if not inherite:
            self.name = varname()
        #self.input = input

        self.parents =[]
        self.childs: list[Step]= []

        self.steps = []
        self.outputs = dict()
        self.curent_id = None

    def set_varname(self):
        from varname import varname
        try:
            self.varname = varname()
        except Exception:
            self.varname = "unnamed"

    def add_steps(self, steps:list['Step']):
        self.childs.extend(steps)
        map(lambda x: x.parents.append(self),steps)

    def add_step(self,step:'Step'):
        self.childs.append(step)
        step.parents.append(self)

    async def start(self,input, run_id=None):
        self.outputs = dict()
        run_id = uuid.uuid4()
        return await self.run(input, run_id)
    
    async def run(self,input, run_id=None):
        if self.curent_id != run_id:
            self.curent_id = run_id
            self.outputs = dict()
        for child in self.childs:
            result = await child.run(input, child.input_name, run_id)

            self.outputs = self.outputs | result

        return  self.outputs


    def __rshift__(self, other:Step|stepList):
        if isinstance(other, stepList):
            for step in other:
                self.childs.append(step)
                step.parents.append(self)
        elif isinstance(other, Step):
            self.childs.append(other)
            other.parents.append(self)
        else:
            raise TypeError('Set a tuple instead of {}'.format(other))
        return other
    def __repr__(self):
        return f"Pipeline({self.name})"
    def __or__(self,other:Step|stepList):
        return self


    def __ror__(self,other:Step|stepList):
        return self

    def __add__(self, other:Step|stepList):
        '''
        handle to add / merge many node
        create a Steplist objet
        '''
        if isinstance(self, stepList):
            self.append(other)
            return self
        elif isinstance(other, stepList):
            other.append(self)
            return other
        else:
            return stepList([self,other])




class Step(Pipeline):
    def __init__(self,
                  function:callable, 
                  parameters:Parameters=None,
                  input_name:str="first_args"):
        self.name = varname()
        super().__init__(inherite=True)


        self.f = function
        self.parameters = parameters or Parameters()
        self._output = None
        self.input_name = input_name
        self.is_async = asyncio.iscoroutinefunction(self.f)
        self.__future_output = asyncio.Future()

    def set_params(self, parameters: Parameters):
        self.parameters.args = parameters.args
        self.parameters.kwargs.update(parameters.kwargs)
    def add_steps(self, steps):
        self.childs.extend(steps)

    @property
    async def output(self):
        return await self.__future_output

    async def run(self,input_value,input_name:str, run_id):
        if self.input_name == "first_args":
            args = [input_value] + self.parameters.args
            kwargs = self.parameters.kwargs
            if self.is_async:
                self._output = await self.f(*args, **kwargs)
            else:
                self._output = await asyncio.to_thread(self.f, *args, **kwargs)
        else:
            kwargs = self.parameters.kwargs | {input_name: input_value}
            if self.is_async:
                self._output = await self.f(**kwargs)
            else:
                self._output = await asyncio.to_thread(self.f, **kwargs)

        self.__future_output.set_result(self.output)

        if self.childs:
            #result = await asyncio.gather(*[step.run(step.input,self.output,run_id) for step in self.steps])
            result = await super().run(self.output,run_id)
            return result
        else : 
            return {self.name:self.output}
        

    def __repr__(self):
        return f"Step({self.name},{self.f.__name__})"
