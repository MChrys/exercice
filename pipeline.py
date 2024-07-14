from __future__ import annotations
import asyncio
import uuid
import functools

from varname import varname

from typing import MutableSequence, TypeVar
from dataclasses import dataclass, field
from typing import List, Dict, Any


from opentelemetry import trace
from opentelemetry.trace import Span 
from opentelemetry import context  
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

class SilentExporter(SpanExporter):
    def export(self, spans):
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def log_span(span):
    logger.info(f"Span: {span.name}")
    logger.info(f"  Trace ID: {span.context.trace_id}")
    logger.info(f"  Span ID: {span.context.span_id}")
    for key, value in span.attributes.items():
        logger.info(f"  Attribute: {key} = {value}")

class Pipeline:
    def __init__(self,inherite =False):
        #self.name = self.set_varname()
        if not inherite:
            self.name = varname()
            self.tracer_name = f"{__name__}.{self.name}"
            self.__init_tracer__()
        #self.input = input
        self.inherite = inherite
        self.parents =[]
        self.childs: list[Step]= []

        self.steps = []
        self.outputs = dict()
        self.curent_id = None

    def __init_tracer__(self):
        resource = Resource(attributes={
            SERVICE_NAME: f"Pipeline-{self.name}"
        })
        provider = TracerProvider(resource=resource)

        exporter = ConsoleSpanExporter()

        processor = BatchSpanProcessor(SilentExporter())
        provider.add_span_processor(processor)

        trace.set_tracer_provider(provider)
        self.tracer = trace.get_tracer(f"{self.tracer_name}")

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

        with self.tracer.start_as_current_span(f"Pipeline :{self.name}") as span:
            span.set_attribute("run_id", str(run_id))
            log_span(span)

            self.tracer_context = context.get_current()
            return await self.run(input, run_id)
    
    async def run(self,input, run_id=None):
        if self.curent_id != run_id:
            self.curent_id = run_id
            self.outputs = dict()

        if not self.inherite:
            token = context.attach(self.tracer_context)
    
        for child in self.childs:
            if asyncio.iscoroutine(input):
                input = await input
            result = await child.run(input, child.input_name, run_id)

            self.outputs = self.outputs | result
        if not self.inherite:
            context.detach(token)

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
        Handle to add / merge many node
        Create a Steplist objet
        '''
        if isinstance(self, stepList):
            self.append(other)
            return self
        elif isinstance(other, stepList):
            other.append(self)
            return other
        else:
            return stepList([self,other])

def sync_step_trace(tracer_name_func):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            tracer = trace.get_tracer(tracer_name_func())
            with tracer.start_as_current_span(func.__name__) as span:
                span.set_attribute("function_name", func.__name__)
                return func(*args, **kwargs)
        return wrapper
    return decorator

def async_step_trace(tracer_name_func):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            tracer = trace.get_tracer(tracer_name_func())
            with tracer.start_as_current_span(func.__name__) as span:
                span.set_attribute("function_name", func.__name__)
                return await func(*args, **kwargs)
        return  wrapper
    return decorator


class Step(Pipeline):
    def __init__(self,
                  function:callable, 
                  parameters:Parameters=None,
                  input_name:str="first_args"):
        self.name = varname()
        super().__init__(inherite=True)



        self.parameters = parameters or Parameters()
        self._output = None
        self.input_name = input_name
        self.is_async = asyncio.iscoroutinefunction(function)

        tracer_name_func = lambda : self.tracer_name

        self.f = async_step_trace(tracer_name_func)(function) if self.is_async else sync_step_trace(tracer_name_func)(function)
        self.__future_output = asyncio.Future()

    @property
    def tracer_name(self):
        if type(self.parents[0]) is Pipeline:
            return self.parents[0].tracer_name
        else:
            return None

    def set_params(self, parameters: Parameters):
        self.parameters.args = parameters.args
        self.parameters.kwargs.update(parameters.kwargs)
    def add_steps(self, steps):
        self.childs.extend(steps)

    @property
    async def output(self):
        return await self.__future_output

    async def run(self,input_value,input_name:str, run_id):

        if not hasattr(self,'tracer'):
            self.tracer = self.parents[0].tracer
        with self.tracer.start_as_current_span(f"Step :{self.name}") as span:
            #span.set_attribute("function_name", self.f.__name__)
            if isinstance (self.parents[0],Step):
                span.set_attribute("parent", self.parents[0].name)
            span.set_attribute("run_id", str(run_id))
            span.set_attribute("input_name", input_name)
            span.set_attribute("function", self.f.__name__)
            log_span(span)
            if self.input_name == "first_args":
                args = [input_value] + self.parameters.args
                kwargs = self.parameters.kwargs
            else:
                args = []
                kwargs = self.parameters.kwargs | {input_name: input_value}

            if self.is_async:
                self._output = await self.f(*args, **kwargs)

            else:
                self._output = self.f(*args, **kwargs)

            self.__future_output.set_result(self._output)

            if self.childs:
                #result = await asyncio.gather(*[step.run(step.input,self.output,run_id) for step in self.steps])
                result = await super().run(self.output,run_id)
                return result
            else : 
                return {self.name:self.output}
        

    def __repr__(self):
        return f"Step({self.name},{self.f.__name__})"
