from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field

load_dotenv()

model = ChatGroq(model='llama-3.1-8b-instant', temperature=0)

#schema for data format of expected output
json_schema = {
  "title": "Review",
  "type": "object",
  "properties": {
    "key_themes": {
      "type": "array",
      "items": {
        "type": "string"
      },
      "description": "Write down all the key themes discussed in the review in a list"
    },
    "summary": {
      "type": "string",
      "description": "A brief summary of the review"
    },
    "sentiment": {
      "type": "string",
      "enum": ["pos", "neg", "mixed"],
      "description": "Return sentiment of the review either negative, positive or mixed"
    },
    "pros": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the pros inside a list"
    },
    "cons": {
      "type": ["array", "null"],
      "items": {
        "type": "string"
      },
      "description": "Write down all the cons inside a list"
    },
    "name": {
      "type": ["string", "null"],
      "description": "Write the name of the reviewer"
    }
  },
  "required": ["key_themes", "summary", "sentiment"]
}

structured_model = model.with_structured_output(json_schema)

result = structured_model.invoke("""
You must return ONLY, nothing else other than above things.:
- key_themes
- summary
- sentiment
- pros
- cons
Text:
I(Amit) purchased the OnePlus Nord CE4 about two weeks ago, and my experience has been a mix of positives and frustrations. 

On the positive side, the design is sleek and lightweight, making it very comfortable to hold for long periods. The display is bright and vibrant, which makes watching videos and browsing a pleasure. Battery life is also quite impressive — it easily lasts a full day with moderate to heavy usage, and the fast charging is a big advantage.

However, there are several issues that are hard to ignore. The phone tends to heat up during gaming sessions and even while using the camera for extended periods. I have also noticed occasional lag and app crashes, especially when multitasking. The camera performance is decent in daylight but struggles significantly in low-light conditions, producing grainy images.

Additionally, I contacted customer support regarding the heating issue, but the response was slow and not very helpful, which added to my frustration.

Overall, while the phone offers good value for money with its design and battery performance, the heating issues, inconsistent performance, and poor customer support experience make it difficult to fully recommend.
""")

print(result)

