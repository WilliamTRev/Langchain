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

