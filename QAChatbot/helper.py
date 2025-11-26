from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from urllib.parse import quote_plus
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain import hub
from langchain.chains import create_sql_query_chain
from  connections import llm
import connections

db = connections.db_connections()
sql_chain = create_sql_query_chain(llm, db)
execute_query = QuerySQLDataBaseTool(db=db)

def get_output(query):
    qu = sql_chain.invoke({"question":query})
    output = execute_query.invoke({"query":qu[10:]})
    return output
if __name__ == "__main__":
    query = get_output("How many white color Levi's t shirts we have available?")
    print(query)