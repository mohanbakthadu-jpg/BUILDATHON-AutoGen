from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from rag import search_knowledge_base


model_client = OpenAIChatCompletionClient(
    model="gpt-4o-mini"
)


def retrieve_context(query: str):
    return search_knowledge_base(query)


customer_query_agent = AssistantAgent(
    name="customer_query_agent",

    model_client=model_client,

    system_message="""
    You are a Customer Query Agent.

    Understand the customer request.

    Identify:
    - intent
    - topic
    - keywords

    Rewrite the question clearly.
    """
)


knowledge_base_agent = AssistantAgent(
    name="knowledge_base_agent",

    model_client=model_client,

    tools=[retrieve_context],

    system_message="""
    You are a Knowledge Base Agent.

    Use the retrieve_context tool.

    Retrieve relevant information from the VMware vSAN FAQ.

    Do not answer the user's question.

    Only return retrieved information.
    """
)


resolution_agent = AssistantAgent(
    name="resolution_agent",

    model_client=model_client,

    system_message="""
    You are a VMware Support Engineer.

    Use the retrieved context to answer.

    If the answer is not available in the
    provided context say:

    'The knowledge base does not contain
    enough information to answer that.'
    """
)