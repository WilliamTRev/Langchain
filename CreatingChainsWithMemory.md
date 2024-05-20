Trainer provided requirements for associates to code on their own designed to reinforce practical application of technology.

# Langchain 005: Creating Chains With Memory Activity

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

1. learn how to set up and build a chain with memory
2. learn how to use a LangChain Agent
3. Learn how to use a LangChain Tool


### Steps

1. Input Open AI key as environmental variable 'OPENAI_API_KEY'
2. add memory to an arbitrary chain: 

```Python
from operator import itemgetter
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

#memory example
model = ChatOpenAI()
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful chatbot"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

memory = ConversationBufferMemory(return_messages=True)

memory.load_memory_variables({})

chain = (
    RunnablePassthrough.assign(
        history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
    )
    | prompt
    | model
)

inputs = {"input": "hi im bob"}
response = chain.invoke(inputs)
print(response)

memory.save_context(inputs, {"output": response.content})

memory.load_memory_variables({})


inputs = {"input": "whats my name"}
response = chain.invoke(inputs)
print(response)
```
3. pass a Runnable into an agent.

```python
###################################Example 2########
from langchain.agents import XMLAgent, tool, AgentExecutor
#from langchain.chat_models import ChatAnthropic

model = ChatOpenAI()


@tool
def search(query: str) -> str:
    """Search things about current events."""
    return "32 degrees"


tool_list = [search]

# Get prompt to use
prompt = XMLAgent.get_default_prompt()


# Logic for going from intermediate steps to a string to pass into model
# This is pretty tied to the prompt
def convert_intermediate_steps(intermediate_steps):
    log = ""
    for action, observation in intermediate_steps:
        log += (
            f"<tool>{action.tool}</tool><tool_input>{action.tool_input}"
            f"</tool_input><observation>{observation}</observation>"
        )
    return log


# Logic for converting tools to string to go in prompt
def convert_tools(tools):
    return "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
```
Building an agent from a runnable usually involves a few things:

   * Data processing for the intermediate steps. These need to represented in a way that the language model can recognize them. This should be pretty tightly coupled to the instructions in the prompt

   * The prompt itself

   * The model, complete with stop tokens if needed

   * The output parser - should be in sync with how the prompt specifies things to be formatted.
```python

agent = (
        {
            "question": lambda x: x["question"],
            "intermediate_steps": lambda x: convert_intermediate_steps(
                x["intermediate_steps"]
            ),
        }
        | prompt.partial(tools=convert_tools(tool_list))
        | model.bind(stop=["</tool_input>", "</final_answer>"])
        | XMLAgent.get_default_output_parser()
)

agent_executor = AgentExecutor(agent=agent, tools=tool_list, verbose=True)

agent_executor.invoke({"question": "whats the weather in New york?"})
```

4. Use tools with Runnables:

```python
#tools

#!pip install duckduckgo-search

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

template = """turn the following user input into a search query for a search engine:

{input}"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI()

chain = prompt | model | StrOutputParser() | search

response2 = chain.invoke({"input": "I'd like to figure out what games are tonight"})
print(response2)
```

### Requirements
1. add memory to a chain
2. use a tool
3. use an agent
4. repeat examples 2-4 with your own promps, different tools and different agents for practice

### Discussion & FAQ

[LangChain Expression Language](https://python.langchain.com/docs/expression_language)

### Solution
Solution code: below or [CreatingChainsWithMemory.py](https://revature0.sharepoint.com/:u:/s/trainers/EfUpH3C2LkxFqaINkadJcCkB_efimh2SuZ2PwL9l32wO1Q?e=5ygzbi)

```python
from operator import itemgetter
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder

#memory example
model = ChatOpenAI()
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful chatbot"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

memory = ConversationBufferMemory(return_messages=True)

memory.load_memory_variables({})

chain = (
    RunnablePassthrough.assign(
        history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
    )
    | prompt
    | model
)

inputs = {"input": "hi im bob"}
response = chain.invoke(inputs)
print(response)

memory.save_context(inputs, {"output": response.content})

memory.load_memory_variables({})


inputs = {"input": "whats my name"}
response = chain.invoke(inputs)
print(response)

#agents example

from langchain.agents import XMLAgent, tool, AgentExecutor
#from langchain.chat_models import ChatAnthropic

model = ChatOpenAI()


@tool
def search(query: str) -> str:
    """Search things about current events."""
    return "32 degrees"


tool_list = [search]

# Get prompt to use
prompt = XMLAgent.get_default_prompt()


# Logic for going from intermediate steps to a string to pass into model
# This is pretty tied to the prompt
def convert_intermediate_steps(intermediate_steps):
    log = ""
    for action, observation in intermediate_steps:
        log += (
            f"<tool>{action.tool}</tool><tool_input>{action.tool_input}"
            f"</tool_input><observation>{observation}</observation>"
        )
    return log


# Logic for converting tools to string to go in prompt
def convert_tools(tools):
    return "\n".join([f"{tool.name}: {tool.description}" for tool in tools])

agent = (
        {
            "question": lambda x: x["question"],
            "intermediate_steps": lambda x: convert_intermediate_steps(
                x["intermediate_steps"]
            ),
        }
        | prompt.partial(tools=convert_tools(tool_list))
        | model.bind(stop=["</tool_input>", "</final_answer>"])
        | XMLAgent.get_default_output_parser()
)

agent_executor = AgentExecutor(agent=agent, tools=tool_list, verbose=True)

agent_executor.invoke({"question": "whats the weather in New york?"})


#tools

#!pip install duckduckgo-search

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.tools import DuckDuckGoSearchRun

search = DuckDuckGoSearchRun()

template = """turn the following user input into a search query for a search engine:

{input}"""
prompt = ChatPromptTemplate.from_template(template)

model = ChatOpenAI()

chain = prompt | model | StrOutputParser() | search

response2 = chain.invoke({"input": "I'd like to figure out what games are tonight"})
print(response2)
```