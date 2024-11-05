from langchain_community.document_loaders import UnstructuredExcelLoader
import os; os.chdir(r"../excel_files")

loader = UnstructuredExcelLoader("ticket_sales_report2.xlsx", mode="elements")
docs = loader.load()

print(len(docs))

docs[1]