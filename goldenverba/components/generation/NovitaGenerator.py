import os
from dotenv import load_dotenv
from goldenverba.components.interfaces import Generator
from goldenverba.components.types import InputConfig
from goldenverba.components.util import get_environment, get_token
import json
import aiohttp

load_dotenv()


class NovitaGenerator(Generator):
    """
    Novita Generator.
    """

    def __init__(self):
        super().__init__()
        self.name = "Novita"
        self.description = "Using Novita LLM models to generate answers to queries"
        self.context_window = 10000

        models = ["meta-llama/llama-3.3-70b-instruct",
            "meta-llama/llama-3.1-8b-instruct"]

        self.config["Model"] = InputConfig(
            type="dropdown",
            value=models[0],
            description="Select an Novita Embedding Model",
            values=models,
        )

        if get_token("NOVITA_API_KEY") is None:
            self.config["API Key"] = InputConfig(
                type="password",
                value="",
                description="You can set your Novita API Key here or set it as environment variable `NOVITA_API_KEY`",
                values=[],
            )
        if os.getenv("NOVITA_BASE_URL") is None:
            self.config["URL"] = InputConfig(
                type="text",
                value="https://api.novita.ai/v3/openai",
                description="You can change the Base URL here if needed",
                values=[],
            )

    async def generate_stream(
        self,
        config: dict,
        query: str,
        context: str,
        conversation: list[dict] = [],
    ):
        system_message = config.get("System Message").value
        model = config.get("Model", {"value": "meta-llama/llama-3.3-70b-instruct"}).value
        novita_key = get_environment(
            config, "API Key", "NOVITA_API_KEY", "No Novita API Key found"
        )
        novita_url = get_environment(
            config, "URL", "NOVITA_BASE_URL", "https://api.novita.ai/v3/openai"
        )

        messages = self.prepare_messages(query, context, conversation, system_message)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {novita_key}",
        }
        data = {
            "messages": messages,
            "model": model,
            "stream": True,
        }

        async with aiohttp.ClientSession() as client:
            async with client.post(
                url=f"{novita_url}/chat/completions",
                json=data,
                headers=headers,
                timeout=None,
            ) as response:
                if response.status == 200:
                    async for line in response.content:
                        if line.strip():
                            line = line.decode("utf-8").strip()
                            json_line = json.loads(line[5:])
                            choice = json_line.get("choices")[0]
                            yield {
                                "message": choice.get("delta", {}).get("content", ""),
                                "finish_reason": (
                                    "stop" if choice.get("finish_reason", "") == "stop" else ""
                                ),
                            }
                else:
                    error_message = await response.text()
                    yield  {"message": f"HTTP Error {response.status}: {error_message}", "finish_reason": "stop"}

    def prepare_messages(
        self, query: str, context: str, conversation: list[dict], system_message: str
    ) -> list[dict]:
        messages = [
            {
                "role": "system",
                "content": system_message,
            }
        ]

        for message in conversation:
            messages.append({"role": message.type, "content": message.content})

        messages.append(
            {
                "role": "user",
                "content": f"Answer this query: '{query}' with this provided context: {context}",
            }
        )

        return messages
