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