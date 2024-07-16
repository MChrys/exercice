from __future__ import annotations
import asyncio
import uuid
import functools
from datetime import datetime
from varname import varname
import os 

from typing import MutableSequence, TypeVar, Sequence
from dataclasses import dataclass, field
from typing import List, Dict, Any


from opentelemetry import trace
from opentelemetry.trace import Span 
from opentelemetry import context  
from opentelemetry.sdk.trace import TracerProvider, ReadableSpan
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult, SimpleSpanProcessor

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def log_span(span):
    logger.info(f"Span: {span.name}")
    logger.info(f"  Trace ID: {span.context.trace_id}")
    #logger.info(f"  Trace Name: {span.context.trace_name}")
    
    logger.info(f"  Span ID: {span.context.span_id}")
    for key, value in span.attributes.items():
        logger.info(f"  Attribute: {key} = {value}")
class SilentExporter(SpanExporter):
    def export(self, spans):
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass

def format_duration(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {seconds:.3f}s"
    elif minutes > 0:
        return f"{int(minutes)}m {seconds:.3f}s"
    else:
        return f"{seconds:.3f}s"

class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

def configure_logger(run_id):
    results_dir = os.path.join("results", str(run_id))
    os.makedirs(results_dir, exist_ok=True)
    log_file = os.path.join(results_dir, "log.txt")

    logger = logging.getLogger(f"run_{run_id}")
    logger.setLevel(logging.INFO)

    file_handler = FlushFileHandler(log_file)
    file_handler.setLevel(logging.INFO)


    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger



class DetailedConsoleSpanExporter(SpanExporter):
    def __init__(self, logger):
        self.logger = logger

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        for span in spans:
            self._export_span(span)
        return SpanExportResult.SUCCESS

    def _export_span(self, span: ReadableSpan):
        start_time = datetime.fromtimestamp(span.start_time / 1e9)
        end_time = datetime.fromtimestamp(span.end_time / 1e9)
        duration_seconds = (span.end_time - span.start_time) / 1e9
        
        self.logger.info(f"Span: {span.name}")
        self.logger.info(f"  Trace ID: {span.context.trace_id}")
        self.logger.info(f"  Span ID: {span.context.span_id}")
        self.logger.info(f"  Parent ID: {span.parent.span_id if span.parent else None}")
        self.logger.info(f"  Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        self.logger.info(f"  End time: {end_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]}")
        self.logger.info(f"  Duration: {format_duration(duration_seconds)}")
        self.logger.info("  Attributes:")
        for key, value in span.attributes.items():
            self.logger.info(f"    {key}: {value}")
        self.logger.info("  Events:")
        for event in span.events:
            event_time = datetime.fromtimestamp(event.timestamp / 1e9)
            event_offset = (event.timestamp - span.start_time) / 1e9
            self.logger.info(f"    {event.name} at {event_time.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} (+{format_duration(event_offset)}):")
            for key, value in event.attributes.items():
                self.logger.info(f"      {key}: {value}")
        self.logger.info("")

    def shutdown(self):
        pass



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

            self._l.append(other)
            return self



class Pipeline:
    def __init__(self,inherite =False):
        #self.name = self.set_varname()
        if not inherite:
            self.name = varname()
            self.tracer_name = f"{__name__}.{self.name}"

        #self.input = input
        self.inherite = inherite
        self.parents =[]
        self.childs: list[Step]= []

        self.steps = []
        self.outputs = dict()
        self.curent_id = None

    def __init_tracer__(self, logger):
        resource = Resource(attributes={
            SERVICE_NAME: f"Pipeline-{self.name}"
        })
        provider = TracerProvider(resource=resource)
        processor = SimpleSpanProcessor(DetailedConsoleSpanExporter(logger))
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
        logger = configure_logger(run_id) 
        self.__init_tracer__(logger) 
        with self.tracer.start_as_current_span(f"Pipeline :{self.name}") as span:
            span.set_attribute("run_id", str(run_id))
            #log_span(span)

            self.tracer_context = context.get_current()
            result = None
            try:
                result = await self.run(input, run_id)
                return result 
            except Exception as e:
                span.record_exception(e)
                raise
            finally:
                span.add_event(f"Pipeline : {self.name} execution completed")
                if result is not None:
                    span.add_event(f"output : {type(result)}")
                else:
                    span.add_event("No output due to exception")
            
    
    async def run(self,input, run_id=None):
        if self.curent_id != run_id:
            self.curent_id = run_id
            self.outputs = dict()

        if not self.inherite:
            token = context.attach(self.tracer_context)
    

        try:
            for child in self.childs:
                if asyncio.iscoroutine(input):
                    input = await input
                result = await child.run(input, child.input_name, run_id)

                if isinstance(result, dict):
                    for key, value in result.items():
                        if asyncio.iscoroutine(value):
                            self.outputs[key] = await value
                        else:
                            self.outputs[key] = value
                else:
                    self.outputs[child.name] = result

                input = await child.wait_for_output()

            return self.outputs
        finally:
            if not self.inherite:
                context.detach(token)


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
        
import json

def serialize_output(output, run_id, step_name):
    tracer = trace.get_tracer(__name__)
    with tracer.start_as_current_span("serialize_output") as span:
        span.add_event("Starting output serialization")


        results_dir = os.path.join("results", str(run_id))
        os.makedirs(results_dir, exist_ok=True)
        
        if isinstance(output, str):
            file_path = os.path.join(results_dir, f"{step_name}.txt")
            with open(file_path, "w") as f:
                f.write(output)
                span.add_event("Serialized string output", {
                    "file_path": file_path,
                    "file_name": f"{step_name}.txt",
                    "file_id": id(f)
                })
        elif isinstance(output, (list, dict)):
            file_path = os.path.join(results_dir, f"{step_name}.json")
            with open(file_path, "w") as f:
                json.dump(output, f, indent=4)
                span.add_event("Serialized JSON output", {
                    "file_path": file_path,
                    "file_name": f"{step_name}.json",
                    "file_id": id(f)
                })
        else:
            span.add_event("Serialization error", {"error": "Unsupported output type"})
            raise ValueError("Unsupported output type for serialization")
        
        span.add_event("Finished output serialization")

def sync_step_trace(tracer_name_func):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            span = trace.get_current_span()
            run_id = span.attributes["run_id"]  
            tracer = trace.get_tracer(tracer_name_func())
            with tracer.start_as_current_span(func.__name__) as span:
                span.set_attribute("function_name", func.__name__)
                #span.add_event(f"func called with value: { {"args": str(args), "kwargs": str(kwargs)}} ")
                result = func(*args, **kwargs)

                step_name = func.__name__
                serialize_output(result, run_id, step_name)
                return result
        return wrapper
    return decorator

def async_step_trace(tracer_name_func):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            span = trace.get_current_span()
            run_id = span.attributes["run_id"] 
            tracer = trace.get_tracer(tracer_name_func())
            with tracer.start_as_current_span(func.__name__) as span:
                span.set_attribute("function_name", func.__name__)
                #span.add_event(f"func called with value: { {"args": str(args), "kwargs": str(kwargs)}} ")
                result = await func(*args, **kwargs)
 
                step_name = func.__name__
                serialize_output(result, run_id, step_name)
                return result
        return  wrapper
    return decorator


class Step(Pipeline):
    def __init__(self,
                  function:callable, 
                  parameters:Parameters=None,
                  input_name:str="first_args"):
        self.name = varname()
        self.__future_output = asyncio.Future()
        super().__init__(inherite=True)



        self.parameters = parameters or Parameters()
        self._output = None
        self.input_name = input_name
        self.is_async = asyncio.iscoroutinefunction(function)

        tracer_name_func = lambda : self.tracer_name

        self.f = async_step_trace(tracer_name_func)(function) if self.is_async else sync_step_trace(tracer_name_func)(function)

    def is_output_ready(self):
        return self.__future_output.done()

    def cancel_output(self):
        if not self.__future_output.done():
            self.__future_output.cancel()

    @property
    def tracer_name(self):
        if type(parent :=self.parents[0]) is Pipeline:
            self._tracer_name = parent.tracer_name
            return parent.tracer_name
        elif hasattr(parent,'_tracer_name'):
            return parent._tracer_name
        else:
            return parent.tracer_name

    def set_params(self, parameters: Parameters):
        self.parameters.args = parameters.args
        self.parameters.kwargs.update(parameters.kwargs)
    def add_steps(self, steps):
        self.childs.extend(steps)

    @property
    async def output(self):
        return await self.__future_output


    async def wait_for_output(self):
        await self.__future_output
        return self.__future_output.result()

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
            #log_span(span)
            if self.input_name == "first_args":
                args = [input_value] + self.parameters.args
                kwargs = self.parameters.kwargs
            else:
                args = []
                kwargs = self.parameters.kwargs | {input_name: input_value}
            try:    
                if self.is_async:
                    self._output = await self.f(*args, **kwargs)

                else:
                    self._output = await asyncio.to_thread(self.f, *args, **kwargs)
            except Exception as e:
                self.__future_output.set_exception(e)
                raise

            self.__future_output.set_result(self._output)

            if self.childs:
                #result = await asyncio.gather(*[step.run(step.input,self.output,run_id) for step in self.steps])
                result = await super().run(self._output,run_id)
                return result
            else : 
                return {self.name:self._output}
        

    def __repr__(self):
        return f"Step({self.name},{self.f.__name__})"
