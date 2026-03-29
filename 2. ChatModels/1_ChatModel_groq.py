from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model =ChatGroq(model='llama-3.3-70b-versatile', temperature=0.7, max_tokens=50)

result = model.invoke('write 5 line funny poem on current indian government and pm of india about their hatred politics')

print(result.content)

# Mostly we will use chatmodel instead of llm for conversational task like chatbots, etc..
# We can also use llm for code generation, summarier and translation.
# use of both entities depends on use case.
# But here we have used the chat models of groq with different model for both llm and chatmodel, bcz we does not get llm of groq.