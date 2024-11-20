from llama_cpp_agent import LlamaCppAgent
from llama_cpp_agent import MessagesFormatterType
from llama_cpp_agent.providers import LlamaCppServerProvider
import socket
# provider = LlamaCppServerProvider("http://127.0.0.1:8383")
provider = LlamaCppServerProvider(f"http://{socket.gethostname()}.local:8383")
agent = LlamaCppAgent(
    provider,
    system_prompt="You are a helpful assistant.",
    debug_output=True,
    predefined_messages_formatter_type=MessagesFormatterType.MISTRAL,
)

settings = provider.get_provider_default_settings()
settings.n_predict = 512
settings.temperature = 0.65

while True:
    user_input = input(">")
    if user_input == "exit":
        break
    agent_output = agent.get_chat_response(user_input, llm_sampling_settings=settings)
    print(f"Agent: {agent_output.strip()}")