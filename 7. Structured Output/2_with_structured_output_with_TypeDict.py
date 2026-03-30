from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional, Literal

load_dotenv()

model = ChatGroq(model='llama-3.1-8b-instant', temperature=0)

#schema for data format of expected output
class Review(TypedDict):
    key_themes : Annotated[list[str], 'Write down all the key themes discussed in the review.']
    summary : Annotated[str, "A brief summary of the review"]
    sentiment : Annotated[Literal['pos', 'neg', 'mixed'], 'return sentiment of review, either nagative, positive or mixed.']
    pros : Annotated[Optional[list[str]], 'write down all pros inside a list']
    cons : Annotated[Optional[list[str]], 'write down all cons inside a list']

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""
You must return ONLY, nothing else other than above things.:
- key_themes
- summary
- sentiment
- pros
- cons
Text:
I purchased the OnePlus Nord CE4 about two weeks ago, and my experience has been a mix of positives and frustrations. 

On the positive side, the design is sleek and lightweight, making it very comfortable to hold for long periods. The display is bright and vibrant, which makes watching videos and browsing a pleasure. Battery life is also quite impressive — it easily lasts a full day with moderate to heavy usage, and the fast charging is a big advantage.

However, there are several issues that are hard to ignore. The phone tends to heat up during gaming sessions and even while using the camera for extended periods. I have also noticed occasional lag and app crashes, especially when multitasking. The camera performance is decent in daylight but struggles significantly in low-light conditions, producing grainy images.

Additionally, I contacted customer support regarding the heating issue, but the response was slow and not very helpful, which added to my frustration.

Overall, while the phone offers good value for money with its design and battery performance, the heating issues, inconsistent performance, and poor customer support experience make it difficult to fully recommend.
""")

print(result)
print(result['summary'])
print(result['sentiment'])