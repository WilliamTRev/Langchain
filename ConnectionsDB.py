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