import asyncio

from agents import (
    customer_query_agent,
    knowledge_base_agent,
    resolution_agent
)

from logger import log_interaction
from rag import search_knowledge_base


async def main():

    question = input("Customer Question: ")

    print("\nRetrieving knowledge...\n")

    context = search_knowledge_base(question)

    resolution_result = await resolution_agent.run(
        task=f"""
        Customer Question:

        {question}

        Retrieved Context:

        {context}

        Provide a final answer.
        """
    )

    answer = resolution_result.messages[-1].content

    print("\n========== ANSWER ==========\n")
    print(answer)

    log_interaction(
        question,
        context,
        answer
    )


if __name__ == "__main__":
    asyncio.run(main())