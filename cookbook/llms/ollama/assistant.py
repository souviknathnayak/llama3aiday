import os
os.chdir("../../../")
from phi.assistant import Assistant
from phi.llm.ollama import Ollama

assistant = Assistant(
    llm=Ollama(model="llama3"),
    description="You help people with their health and fitness goals.",
)
assistant.print_response("Share a quick healthy breakfast recipe.", markdown=True)
