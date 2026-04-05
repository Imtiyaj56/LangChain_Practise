from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('pdfs_folder/Imtiyaj (10-14).pdf')

doc = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 0,
)

chunks = splitter.split_documents(doc)
print(chunks[3].page_content)    # 3rd chunk