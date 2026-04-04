"""This loader is used to load whole directory or folder."""
'''It works with all other pdf loader to load different types of files from that directory/folder'''

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path= 'pdfs_folder',
    glob= '*.pdf',
    loader_cls= PyPDFLoader
)

doc = loader.load()  # can also use lazy_load()
print(doc[0].metadata)  #metadata of 0th index page
print(doc[0].page_content)
print(len(doc))

'''lazy_load()  :-

        - Used when the file is too much big.
        - you dont want to load everything at once.
        - it loads the file on demand, not at once, reducing usage of memory.
        - it uses generator to load the needed data.
        - it works based on streaming concept.
        - when to use : with large files or large no of files
'''