from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_xai import ChatXAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from project_package.data_ingestion import data_ingestion



from dotenv import load_dotenv
import os

load_dotenv()

os.environ["XAI_API_KEY"] = os.getenv("XAI_API_KEY")

model = ChatXAI(
    xai_api_key= os.environ["XAI_API_KEY"],
    model="grok-3",
    temperature=0.5
)


chat_history= []
store = {}
def get_session_history(session_id: str)-> BaseChatMessageHistory:
  if session_id not in store:
    store[session_id]= ChatMessageHistory()
  return store[session_id]


def generation(vstore):
    retriever = vstore.as_retriever(search_kwargs={"k": 3})

    retriever_prompt = ("Given a chat history and the latest user question which might reference context in the chat history,"
    "formulate a standalone question which can be understood without the chat history."
    "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
    ("system", retriever_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ]
)
    history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)
    PRODUCT_BOT_TEMPLATE = """
    You are an AI assistant specializing in personalized product recommendations based on user reviews and feedback.
    Your goal is to provide **accurate, relevant, and unbiased** suggestions without assuming preferences **unless explicitly stated by the user**.

    ### Memory & Retrieval Guidelines:
    - If retrieved context contains **irrelevant data (e.g., price concerns, outdated preferences)**, **IGNORE IT** unless the user asks about those specific details.
    - DO NOT assume any priorities such as budget, brand preference, or features unless the user explicitly mentions them.

    ### Response Formatting:
    - **Keep recommendations natural & conversational**, avoiding unnecessary bullet lists.
    - Provide clear comparisons **only if the user requests them**.
    - Adapt responses based on **retrieved context**, but filter out **misleading or irrelevant details**.

    ### Example Scenario:
    **User Input:** "Which are the best Bluetooth earbuds?"
    **Correct Response:**
    "If you're looking for **great sound quality and noise cancellation**, the **Sony WF-1000XM4** is a top contender.
    If durability and battery life matter more, the **JBL Tune 125TWS** is a reliable pick with long-lasting performance.
    Would you prefer **better bass** or **a longer-lasting battery**?"

    **Incorrect Response:**
    "I see that **price is a concern**, so I’d recommend budget-friendly options…" ❌ *(Avoid assumptions unless the user mentions price!)*

    CONTEXT:
    {context}

    QUESTION: {input}

    YOUR ANSWER:
    """
    qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", PRODUCT_BOT_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
    return conversational_rag_chain



if __name__ == "__main__":
   vstore = data_ingestion("done")
   conversational_rag_chain = generation(vstore)
   answer= conversational_rag_chain.invoke(
    {"input": "can you tell me the best bluetooth buds?"},
    config={
        "configurable": {"session_id": "dhruv"}
    },  # constructs a key "abc123" in `store`.
)["answer"]
   print(answer)
   answer1= conversational_rag_chain.invoke(
    {"input": "what is my previous question?"},
    config={
        "configurable": {"session_id": "dhruv"}
    },  # constructs a key "abc123" in `store`.
)["answer"]
   print(answer1)