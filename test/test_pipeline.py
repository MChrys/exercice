from workflows.pipeline import Pipeline, Step
from asyncio import run
def voo(val: int):
    return "voo"
def too(val):
    return "too"
def xoo(val):
    return "xoo"  
def tal(val):
    return "tal"
pipe = Pipeline()
a = Step(voo)
b = Step(too)
c = Step(xoo)
d = Step(tal)
e = Step(voo)
f = Step(too)
pipe >> a >>  b + c  |  b >> d 
"""
    pipe
    |
    a
    |\
    b c   
    |
    d
"""
result = pipe.start(2)
print(result)
print(result[0])