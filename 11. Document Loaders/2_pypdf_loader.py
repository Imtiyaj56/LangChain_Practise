'''This loader is used to load the data from pdf format to document form.'''
'''Mostly used loader.'''
'''It works page by page, which is best thing.'''
'''Limitations: Not great with scanned pdf or complex layout pdfs.'''

from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

loader = PyPDFLoader('pdfs_folder\NDA Of AKIYAM (2).pdf')

doc = loader.load()
print(len(doc))   #No of doc is equal to no of pages in pages in pdf, here it is 6
print(doc[3].page_content)


#Different Type of Pdfs Loaders

'''
--> PDFPlumberLoader : used for pdf with rows and columns.
--> AmazonTextractorPDFLoader or UnstructuredPDFLoader : used with scanned/image base pdf.
--> PyMuPDFLoader : Need data of layout and image data.
--> UnstructuredPDFLoader : Want best structure extraction.
'''
