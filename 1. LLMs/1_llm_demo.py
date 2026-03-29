from langchain_groq import ChatGroq   #loading groq chatmodel
from dotenv import load_dotenv  #use for loading env variable from .env to this file

load_dotenv()   #invoking (loading .env data in this file )

model = ChatGroq(model="llama-3.1-8b-instant")   #object of chatmodel class

result = model.invoke('What is Captital of Romania and Hungary.')   #take prompt to model and return the answer

print(result.content)