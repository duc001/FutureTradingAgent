from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. 准备向量库（假设已建好）

vectorstore = Chroma.from_documents(documents, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 2. Prompt 模板
template = """基于以下上下文回答问题。如果不知道，就说"我不知道"。

上下文：
{context}

问题：{question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 3. 模型
model = ChatOpenAI(model="deepseek-v4-flash", api_key="sk-205ee372c5df491f8050324eb697e504", base_url="https://api.deepseek.com")
parser = StrOutputParser()

# 4. 辅助函数：把检索结果格式化为字符串
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 5. 组装链
rag_chain = (
    {
        "context": retriever | format_docs,   # 先检索，再格式化
        "question": RunnablePassthrough()       # 直接透传用户问题
    }
    | prompt    # 填充模板
    | model     # 调用模型
    | StrOutputParser()  # 解析为纯文本
)

# 6. 执行
response = rag_chain.invoke("公司的年假政策是什么？")
print(response)