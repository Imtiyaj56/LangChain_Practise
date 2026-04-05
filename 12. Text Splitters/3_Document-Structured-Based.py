'''It is same as RecursiveCharacterTextSplitter, but it is used in special cases,
   like for loading the codebase, etc....'''
'''We have to just specify, whether it python code, java code or other, so it can
   split the code base on python code symbols.'''

from langchain_text_splitters import RecursiveCharacterTextSplitter, Language

text = '''from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 1. Dummy Class to hold text processing logic
class TextProcessor:
    def __init__(self, chunk_size=50, chunk_overlap=10):
        # Initialize the splitter with specific separators
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

    def process_text(self, text):
        # Create dummy documents
        docs = [Document(page_content=text)]
        return self.splitter.split_documents(docs)

# 2. Function to test the splitter
def test_recursive_splitter():
    # Setup
    raw_text = """
    LangChain is a framework. It simplifies building LLM applications.
    RecursiveTextSplitter helps create chunks.
    It works by recursively splitting text.
    """
    processor = TextProcessor(chunk_size=50, chunk_overlap=10)
    
    # Execution
    result_docs = processor.process_text(raw_text)
    
    # Assertions
    print(f"Number of chunks: {len(result_docs)}")
    for i, doc in enumerate(result_docs):
        print(f"Chunk {i+1} (Len: {len(doc.page_content)}): '{doc.page_content.strip()}'")
        assert len(doc.page_content) <= 50

if __name__ == "__main__":
    test_recursive_splitter()
'''

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size = 175,
    chunk_overlap = 0,
)

chunks = splitter.split_text(text)

print(chunks[2])

