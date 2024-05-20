Trainer provided requirements for associates to code on their own designed to reinforce practical application of technology.

# Langchain 004: Chains Activity

### Contributors:

William Terry

{ Reviewers }

### Prerequisites

1. Python
2. ChatGPT
3. Prompt Engineering

### Technologies Used

1. Python
2. LangChain
3. GPT-3.5-turbo


### Time
2 hours

### Applied Skills

1. learn how to create prompts with Langchain
2. using the prompt template with Langchain
3. creating function calls with prompts


### Steps

1. Combine a prompt and model to create a chain that takes user input, adds to a prompt, passes to a model, and returns the model output.

Combine PromptTemplate/ChatPromptTemplates and LLMs/ChatModels in different ways here.
```python
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

prompt = ChatPromptTemplate.from_template("tell me a joke about {foo}")
model = ChatOpenAI()
chain = prompt | model

response = chain.invoke({"foo": "bears"})

print(response)

```

2. Often times we want to attach kwargs that'll be passed to each model call. Here are a few examples of that:

```Python
chain = prompt | model.bind(stop=["\n"])

response2 = chain.invoke({"foo": "bears"})

print(response2)

```
3. Attach Function Call information
```python

functions = [
    {
        "name": "joke",
        "description": "A joke",
        "parameters": {
            "type": "object",
            "properties": {
                "setup": {"type": "string", "description": "The setup for the joke"},
                "punchline": {
                    "type": "string",
                    "description": "The punchline for the joke",
                },
            },
            "required": ["setup", "punchline"],
        },
    }
]
chain = prompt | model.bind(function_call={"name": "joke"}, functions=functions)

response3 = chain.invoke({"foo": "bears"}, config={})

print(response3)
print(type(response3))

```
4. We can also add in an output parser to easily transform the raw LLM/ChatModel output into a more workable format.
Notice that this now returns a string - a much more workable format for downstream tasks

```python
from langchain.schema.output_parser import StrOutputParser

chain = prompt | model | StrOutputParser()

response4= chain.invoke({"foo": "bears"})

print(response4)
print(type(response4))
```
5. When you specify the function to return, you may just want to parse that directly
```python
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

chain = (
    prompt
    | model.bind(function_call={"name": "joke"}, functions=functions)
    | JsonOutputFunctionsParser()
)

response5 = chain.invoke({"foo": "bears"})
print(response5)

from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser

chain = (
    prompt
    | model.bind(function_call={"name": "joke"}, functions=functions)
    | JsonKeyOutputFunctionsParser(key_name="setup")
)

response6=chain.invoke({"foo": "bears"})

print(response6)
```
6. To make invocation even simpler, we can add a RunnableMap to take care of creating the prompt input dict for us:
```python
from langchain.schema.runnable import RunnableMap, RunnablePassthrough

map_ = RunnableMap(foo=RunnablePassthrough())
chain = (
    map_
    | prompt
    | model.bind(function_call={"name": "joke"}, functions=functions)
    | JsonKeyOutputFunctionsParser(key_name="setup")
)

response7 = chain.invoke("bears")

print(response7)

chain = (
    {"foo": RunnablePassthrough()}
    | prompt
    | model.bind(function_call={"name": "joke"}, functions=functions)
    | JsonKeyOutputFunctionsParser(key_name="setup")
)

response8=chain.invoke("bears")
print(response8)
```
### Requirements
1. string together response chains above
2. redirect output with steps above
3. repeat examples 1-6 with your own prompts 

### Discussion & FAQ

[LangChain Expression Language](https://python.langchain.com/docs/expression_language)
```python
##############################################Example 3########################################
planner = (
    ChatPromptTemplate.from_template("Generate an argument about: {input}")
    | ChatOpenAI()
    | StrOutputParser()
    | {"base_response": RunnablePassthrough()}
)

arguments_for = (
    ChatPromptTemplate.from_template(
        "List the pros or positive aspects of {base_response}"
    )
    | ChatOpenAI()
    | StrOutputParser()
)
arguments_against = (
    ChatPromptTemplate.from_template(
        "List the cons or negative aspects of {base_response}"
    )
    | ChatOpenAI()
    | StrOutputParser()
)

final_responder = (
    ChatPromptTemplate.from_messages(
        [
            ("ai", "{original_response}"),
            ("human", "Pros:\n{results_1}\n\nCons:\n{results_2}"),
            ("system", "Generate a final response given the critique"),
        ]
    )
    | ChatOpenAI()
    | StrOutputParser()
)

chain = (
    planner
    | {
        "results_1": arguments_for,
        "results_2": arguments_against,
        "original_response": itemgetter("base_response"),
    }
    | final_responder
)
response3 = chain.invoke({"input": "scrum"})

print(response3)
```

### Requirements
1. Use the prompt templates above
2. Also use the function calls 
3. pay close attention to the types of output
4. repeat examples 1-8 with your own prompts

### Discussion & FAQ

[LangChain Expression Language](https://python.langchain.com/docs/expression_language)

### Solution

Solution code: below or [PromptFormatting.py](https://revature0.sharepoint.com/:u:/s/trainers/ESHWmdyksdZMmoAq4y383m8BAlZzpAUWJPkvLlT1NCIolg?e=WGeokr)
```python
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

prompt = ChatPromptTemplate.from_template("tell me a joke about {foo}")
model = ChatOpenAI()
chain = prompt | model

response = chain.invoke({"foo": "bears"})

print(response)

#############################
chain = prompt | model.bind(stop=["\n"])

response2 = chain.invoke({"foo": "bears"})

print(response2)

###############################

functions = [
    {
        "name": "joke",
        "description": "A joke",
        "parameters": {
            "type": "object",
            "properties": {
                "setup": {"type": "string", "description": "The setup for the joke"},
                "punchline": {
                    "type": "string",
                    "description": "The punchline for the joke",
                },
            },
            "required": ["setup", "punchline"],
        },
    }
]
chain = prompt | model.bind(function_call={"name": "joke"}, functions=functions)

response3 = chain.invoke({"foo": "bears"}, config={})

print(response3)
print(type(response3))

######################
from langchain.schema.output_parser import StrOutputParser

chain = prompt | model | StrOutputParser()

response4= chain.invoke({"foo": "bears"})

print(response4)
print(type(response4))

################################
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser

chain = (
    prompt
    | model.bind(function_call={"name": "joke"}, functions=functions)
    | JsonOutputFunctionsParser()
)

response5 = chain.invoke({"foo": "bears"})
print(response5)

from langchain.output_parsers.openai_functions import JsonKeyOutputFunctionsParser

chain = (
    prompt
    | model.bind(function_call={"name": "joke"}, functions=functions)
    | JsonKeyOutputFunctionsParser(key_name="setup")
)

response6=chain.invoke({"foo": "bears"})

print(response6)

#############################
from langchain.schema.runnable import RunnableMap, RunnablePassthrough

map_ = RunnableMap(foo=RunnablePassthrough())
chain = (
    map_
    | prompt
    | model.bind(function_call={"name": "joke"}, functions=functions)
    | JsonKeyOutputFunctionsParser(key_name="setup")
)

response7 = chain.invoke("bears")

print(response7)

chain = (
    {"foo": RunnablePassthrough()}
    | prompt
    | model.bind(function_call={"name": "joke"}, functions=functions)
    | JsonKeyOutputFunctionsParser(key_name="setup")
)

response8=chain.invoke("bears")
print(response8)
```