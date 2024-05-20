Trainer provided requirements for associates to code on their own designed to reinforce practical application of technology.

# Langchain 003: Data Connections Excercise

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

1. learn how to set up and build a chain with a data connection
2. generate SQL syntax with the LLM prompts in English


### Steps

1. Input Open AI key as environmental variable 'OPENAI_API_KEY'
2. place Chinook.db SQLite Database in the same directory as your Python module
3. Replicate our SQLDatabaseChain with Runnables and generate an SQL query from a prompt: 

```Python
from langchain.prompts import ChatPromptTemplate
from langchain.utilities import SQLDatabase
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

template = """Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query:"""
prompt = ChatPromptTemplate.from_template(template)

db = SQLDatabase.from_uri("sqlite:///./Chinook.db")


def get_schema(_):
    return db.get_table_info()


def run_query(query):
    return db.run(query)


model = ChatOpenAI()

sql_response = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | model.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)

response = sql_response.invoke({"question": "How many employees are there?"})
print(response)
```
3. Utilize a Prompt Template for SQL query

```python

template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""
prompt_response = ChatPromptTemplate.from_template(template)
print(prompt_response)

```

4. Finally print the Response for the SQL query
```python

full_chain = (
    RunnablePassthrough.assign(query=sql_response)
    | RunnablePassthrough.assign(
        schema=get_schema,
        response=lambda x: db.run(x["query"]),
    )
    | prompt_response
    | model
)

response2 = full_chain.invoke({"question": "How many employees are there?"})

print(response2)
```

### Requirements
1. Write an SQL query from a Prompt
2. Generate the response from the query using a Template
3. Repeat steps 1-2 with your own queries and templates

### Discussion & FAQ

[LangChain Expression Language](https://python.langchain.com/docs/expression_language)

### Solution
Solution code: below or ConnectionsDB.py

```python
from langchain.prompts import ChatPromptTemplate
from langchain.utilities import SQLDatabase
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

template = """Based on the table schema below, write a SQL query that would answer the user's question:
{schema}

Question: {question}
SQL Query:"""
prompt = ChatPromptTemplate.from_template(template)

db = SQLDatabase.from_uri("sqlite:///./Chinook.db")


def get_schema(_):
    return db.get_table_info()


def run_query(query):
    return db.run(query)


model = ChatOpenAI()

sql_response = (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | model.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)

response = sql_response.invoke({"question": "How many employees are there?"})
print(response)


template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""
prompt_response = ChatPromptTemplate.from_template(template)
print(prompt_response)

full_chain = (
    RunnablePassthrough.assign(query=sql_response)
    | RunnablePassthrough.assign(
        schema=get_schema,
        response=lambda x: db.run(x["query"]),
    )
    | prompt_response
    | model
)

response2 = full_chain.invoke({"question": "How many employees are there?"})

print(response2)
```