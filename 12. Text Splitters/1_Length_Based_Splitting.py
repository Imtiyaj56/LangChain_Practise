from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('pdfs_folder/Imtiyaj (10-14).pdf')

docs = loader.load()

splitter = CharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap = 10,  #overlap of character btwn two chunks
    separator = ''
)

result = splitter.split_documents(docs)

print(result[0].page_content)   # 0th chunk