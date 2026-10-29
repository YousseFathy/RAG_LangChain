
from langchain_core.prompts import ChatPromptTemplate


def get_prompt():
    return ChatPromptTemplate.from_template(
        """Answer using only the following context.
        If the answer is not in the context, say you don't know.

        Context:
        {context}

        Question:
        {question}
        """
    )
