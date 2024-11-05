from langchain_community.document_loaders import UnstructuredExcelLoader
import os; os.chdir(r"../excel_files")
print(os.getcwd())
print(os.listdir())
loader = UnstructuredExcelLoader("tickes_sales.xlsx", mode="elements")
docs = loader.load()

print(len(docs))

docs