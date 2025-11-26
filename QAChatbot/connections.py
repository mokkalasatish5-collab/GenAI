from langchain_google_genai import ChatGoogleGenerativeAI
from urllib.parse import quote_plus
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   # IMPORTANT: new correct name
    google_api_key="yourkey",
    temperature=0.7
)

def db_connections():
    db_user = "root"
    db_password = "password"
    encoded_password = quote_plus(db_password)
    db_host = "localhost"
    db_name = "atliq_tshirts"

    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{encoded_password}@{db_host}/{db_name}",
                              sample_rows_in_table_info=3)
    return db