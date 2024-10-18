# Revised and Expanded Swarm Cookbook

## Orchestrating Agents: A Complete Guide to Building Multi-Agent Systems with Swarm | OpenAI Cookbook

**URL Source:** https://cookbook.openai.com/examples/orchestrating_agents

**1. Introduction:**

Welcome to the world of multi-agent systems with Swarm! This guide provides a comprehensive walkthrough of building sophisticated conversational agents that can collaborate, hand off tasks, and interact with external tools. Swarm, a lightweight and ergonomic Python framework, simplifies the development and deployment of such systems.

Swarm was created to explore patterns that are lightweight, scalable, and highly customizable by design. It's particularly well-suited for situations dealing with a large number of independent capabilities and instructions that are difficult to encode into a single prompt.

It's important to note that Swarm is currently an experimental framework intended to explore ergonomic interfaces for multi-agent systems. It is not intended to be used in production and has no official support. Swarm runs almost entirely on the client and does not store state between calls, providing developers with more control over the execution environment. We'll introduce the notion of **routines** and **handoffs**, then walk through the implementation and show how they can be used to orchestrate multiple agents in a simple, powerful, and controllable way.

**2. Why Swarm?**

- **Deep Dive into Orchestration:** Swarm lets you see how agent coordination works under the hood, giving you a deeper understanding of the process.
- **Client-Side Control:** Unlike the Assistants API, Swarm runs mainly on your side, giving you more control over the execution environment.
- **Flexibility and Customization:** If you have complex, highly customized logic for your agents, Swarm provides the flexibility to implement it.
- **Educational Focus:** Swarm is primarily meant for learning about and experimenting with multi-agent systems.

**3. Swarm's Core Principles:**

Swarm is built on the following principles:

- **Lightweight:** Designed for simplicity and minimal overhead.
- **Scalable:** Can handle complex agent networks and interactions.
- **Controllable:** Provides fine-grained control over agent behavior and handoffs.
- **Testable:** Facilitates easy testing of multi-agent systems.

**4. Core Concepts:**

- **4.1 Agents:**  Agents are the heart of Swarm, representing individual conversational routines. Each agent has specific instructions, access to tools (functions), and the ability to hand off the conversation to another agent. Think of agents as specialized experts that handle particular aspects of a conversation. 

   ```python 
    class Agent(BaseModel):
        name: str = "Agent"
        model: str = "gpt-4o"
        instructions: Union[str, Callable[[dict], str]] = "You are a helpful agent."
        functions: List[Callable] = []
        tool_choice: str = None  # For advanced tool selection control
        parallel_tool_calls: bool = True # Whether to allow parallel tool calls 
   ```

   **Field Descriptions:**

   - `name`:  A string that identifies the agent.
   - `model`: The OpenAI language model the agent will use (default is "gpt-4o").
   - `instructions`:  The instructions that guide the agent's behavior. Can be a simple string or a function that returns a string. 
   - `functions`: A list of callable Python functions that the agent has access to. These functions represent the agent's "tools."
   - `tool_choice`: Advanced option for controlling which function the model should call (see OpenAI API docs).
   - `parallel_tool_calls`: A boolean that indicates whether the agent can call multiple functions in parallel (default is `True`).


- **4.2 Instructions:** Instructions define the persona and behavior of an agent. They can be simple strings or dynamic functions that adapt based on context:

    ```python
    from swarm import Agent

    # Simple string instructions:
    sales_agent = Agent(
        name="Sales Agent",
        instructions="You are a friendly sales agent.  Your goal is to help customers find the perfect product."
    )

    # Dynamic instructions based on context:
    def support_instructions(context_variables):
        user_name = context_variables.get("user_name", "user")
        return f"You are a technical support agent.  Help {user_name} troubleshoot their issue."

    support_agent = Agent(
        name="Support Agent",
        instructions=support_instructions
    )
    ```

- **4.3 Functions (Tools):** Functions serve as tools that agents can call to perform actions or retrieve information. These functions can interact with databases, APIs, or even perform simple calculations:

    ```python
    def look_up_product(product_name):
        # Simulate a database lookup
        return {"product_id": "12345", "price": 99.99}

    def send_confirmation_email(email_address):
        # Send an email (in a real application)
        print(f"Confirmation email sent to {email_address}")
        return "Email sent successfully!"

    sales_agent = Agent(functions=[look_up_product, send_confirmation_email])
    ```

- **4.4 Handoffs:** Handoffs enable smooth transitions between agents. An agent can trigger a handoff by returning the target agent from a function:

    ```python
    def transfer_to_support():
        return support_agent

    sales_agent = Agent(functions=[..., transfer_to_support]) # Add to sales_agent's functions
    ```

- **4.5 Context Variables:** Context variables are key-value pairs that provide a shared memory space for agents and functions. They allow you to maintain state and context throughout a conversation:
    ```python
    context_variables = {"user_name": "Bob", "order_id": "ORDER-789"}
    client.run(agent=sales_agent, messages=[...], context_variables=context_variables) 
    ```

- **4.6 Updating Context Variables:** 
   You can update `context_variables` within a function using the `Result` class:

   ```python
   from swarm import Result

   class Result(BaseModel): 
       """
       Encapsulates the possible return values for an agent function.
       """
       value: str = ""
       agent: Optional[Agent] = None
       context_variables: dict = {}

   def set_user_location(context_variables, location):
       context_variables["user_location"] = location 
       return Result(value="Location updated.", context_variables=context_variables) 

   # ... Agent definition with the set_user_location function ...
   ```

- **4.7 Routines:** The notion of a "routine" is not strictly defined, and instead meant to capture the idea of a set of steps. Conretely, let's define a routine to be a list of instructions in natural langauge (which we'll represent with a system prompt), along with the tools necessary to complete them.

Let's take a look at an example. Below, we've defined a routine for a customer service agent instructing it to triage the user issue, then either suggest a fix or provide a refund. We've also defined the necessary functions `execute_refund` and `look_up_item`. We can call this a customer service routine, agent, assistant, etc – however the idea itself is the same: a set of steps and the tools to execute them.

```
# Customer Service Routine

system_message = (
    "You are a customer support agent for ACME Inc."
    "Always answer in a sentence or less."
    "Follow the following routine with the user:"
    "1. First, ask probing questions and understand the user's problem deeper.\n"
    " - unless the user has already provided a reason.\n"
    "2. Propose a fix (make one up).\n"
    "3. ONLY if not satesfied, offer a refund.\n"
    "4. If accepted, search for the ID and then execute refund."
    ""
)

def look_up_item(search_query):
    """Use to find item ID.
    Search query can be a description or keywords."""

    # return hard-coded item ID - in reality would be a lookup
    return "item_132612938"


def execute_refund(item_id, reason="not provided"):

    print("Summary:", item_id, reason) # lazy summary
    return "success"
```

The main power of routines is their simplicity and robustness. Notice that these instructions contain conditionals much like a state machine or branching in code. LLMs can actually handle these cases quite robustly for small and medium sized routine, with the added benefit of having "soft" adherance – the LLM can naturally steer the conversation without getting stuck in dead-ends.

[Executing Routines](https://cookbook.openai.com/examples/orchestrating_agents#executing-routines)
--------------------------------------------------------------------------------------------------

To execute a routine, let's implement a simple loop that:

1.  Gets user input.
2.  Appends user message to `messages`.
3.  Calls the model.
4.  Appends model response to `messages`.

```
def run_full_turn(system_message, messages):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system_message}] + messages,
    )
    message = response.choices[0].message
    messages.append(message)

    if message.content: print("Assistant:", message.content)

    return message


messages = []
while True:
    user = input("User: ")
    messages.append({"role": "user", "content": user})

    run_full_turn(system_message, messages)
```

As you can see, this currently ignores function calls, so let's add that.

Models require functions to be formatted as a function schema. For convenience, we can define a helper function that turns python functions into the corresponding function schema.

```
import inspect

def function_to_schema(func) -> dict:
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": (func.__doc__ or "").strip(),
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }
```

For example:

```
def sample_function(param_1, param_2, the_third_one: int, some_optional="John Doe"):
    """
    This is my docstring. Call this function when you want.
    """
    print("Hello, world")

schema =  function_to_schema(sample_function)
print(json.dumps(schema, indent=2))
```

{
  "type": "function",
  "function": {
    "name": "sample\_function",
    "description": "This is my docstring. Call this function when you want.",
    "parameters": {
      "type": "object",
      "properties": {
        "param\_1": {
          "type": "string"
        },
        "param\_2": {
          "type": "string"
        },
        "the\_third\_one": {
          "type": "integer"
        },
        "some\_optional": {
          "type": "string"
        }
      },
      "required": \[
        "param\_1",
        "param\_2",
        "the\_third\_one"
      \]
    }
  }
}

Now, we can use this function to pass the tools to the model when we call it.

```
messages = []

tools = [execute_refund, look_up_item]
tool_schemas = [function_to_schema(tool) for tool in tools]

response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Look up the black boot."}],
            tools=tool_schemas,
        )
message = response.choices[0].message

message.tool_calls[0].function
```

Function(arguments='{"search\_query":"black boot"}', name='look\_up\_item')

Finally, when the model calls a tool we need to execute the corresponding function and provide the result back to the model.

We can do this by mapping the name of the tool to the python function in a `tool_map`, then looking it up in `execute_tool_call` and calling it. Finally we add the result to the conversation.

```
tools_map = {tool.__name__: tool for tool in tools}

def execute_tool_call(tool_call, tools_map):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    print(f"Assistant: {name}({args})")

    # call corresponding function with provided arguments
    return tools_map[name](**args)

for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call, tools_map)

            # add result back to conversation 
            result_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
            messages.append(result_message)
```

Assistant: look\_up\_item({'search\_query': 'black boot'})

In practice, we'll also want to let the model use the result to produce another response. That response might _also_ contain a tool call, so we can just run this in a loop until there are no more tool calls.

If we put everything together, it will look something like this:

```
tools = [execute_refund, look_up_item]


def run_full_turn(system_message, tools, messages):

    num_init_messages = len(messages)
    messages = messages.copy()

    while True:

        # turn python functions into tools and save a reverse map
        tool_schemas = [function_to_schema(tool) for tool in tools]
        tools_map = {tool.__name__: tool for tool in tools}

        # === 1. get openai completion ===
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system_message}] + messages,
            tools=tool_schemas or None,
        )
        message = response.choices[0].message
        messages.append(message)

        if message.content:  # print assistant response
            print("Assistant:", message.content)

        if not message.tool_calls:  # if finished handling tool calls, break
            break

        # === 2. handle tool calls ===

        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call, tools_map)

            result_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
            messages.append(result_message)

    # ==== 3. return new messages =====
    return messages[num_init_messages:]


def execute_tool_call(tool_call, tools_map):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    print(f"Assistant: {name}({args})")

    # call corresponding function with provided arguments
    return tools_map[name](**args)


messages = []
while True:
    user = input("User: ")
    messages.append({"role": "user", "content": user})

    new_messages = run_full_turn(system_message, tools, messages)
    messages.extend(new_messages)
```

Now that we have a routine, let's say we want to add more steps and more tools. We can up to a point, but eventually if we try growing the routine with too many different tasks it may start to struggle. This is where we can leverage the notion of multiple routines – given a user request, we can load the right routine with the appropriate steps and tools to address it.

Dynamically swapping system instructions and tools may seem daunting. However, if we view "routines" as "agents", then this notion of **handoffs** allow us to represent these swaps simply – as one agent handing off a conversation to another.

**5. Installation and Setup:**

**5.1 Prerequisites:**
    - Python 3.10+
    - OpenAI API key

**5.2 Installation:**

    - **Option 1: Install from PyPI:**

        ```bash
        pip install swarm
        ```

    - **Option 2: Install from Repository:**
        First, clone the Swarm repository and install the necessary dependencies:

        ```bash
        git clone https://github.com/openai/swarm.git
        cd swarm
        pip install -r requirements.txt
        ```

**5.3 OpenAI API Key Setup**
    Set your OpenAI API key as an environment variable:

    ```bash
    export OPENAI_API_KEY='your-api-key-here'
    ```

**6. Basic Usage:**

**6.1 Imports:**

First, be sure to include these minimum imports:

```python
from openai import OpenAI
from swarm import Swarm, Agent

client = OpenAI()
swarm = Swarm(client=client)
```

A more typical set of imports for a slightly larger project might look like this:

```
from openai import OpenAI
from pydantic import BaseModel
from typing import Optional
import json

client = OpenAI()
```

**6.2 Creating a Simple Agent:**

Here’s how to create a simple agent that responds to user input:

```python
from swarm import Swarm, Agent

client = Swarm()

agent = Agent(
    name="Simple Agent",
    instructions="You are a helpful agent."
)

messages = [{"role": "user", "content": "Hello!"}]
response = client.run(agent=agent, messages=messages)

print(response.messages[-1]["content"])
```

**6.3 Adding Functions to an Agent:**

Agents can call Python functions to interact with external systems:

```python
def greet(name):
    return f"Hello, {name}!"

agent = Agent(
    name="Greeting Agent",
    instructions="Greet the user by name.",
    functions=[greet]
)
```

**7. Executing Routines and Handoffs:**

Swarm provides the `client.run()` function, which is the central mechanism for executing routines and handling handoffs.

**7.1 `client.run()` Function:**

```python
class Swarm:
    # ... other code ...

    def run(
        self,
        agent: Agent,
        messages: List, 
        context_variables: dict = {},
        model_override: str = None,
        stream: bool = False,
        debug: bool = False,
        max_turns: int = float("inf"),
        execute_tools: bool = True,
    ) -> Response: 
        # ... implementation (see the Swarm repo for the full code) ... 
```

**7.2  `client.run()` Internal Loop:**

The `client.run()` function internally follows this loop:

1.  **Get Completion:** It calls the OpenAI API to get a response from the current agent, using its instructions and the chat history.
2.  **Execute Tool Calls:** If the response includes function calls (tool calls), Swarm executes the corresponding Python functions in the agent's `functions` list.
3.  **Agent Handoff:**  If a function returns a new `Agent` object, the conversation is handed off to that agent.
4.  **Update Context:** Swarm updates `context_variables` based on the results of function calls or handoffs.
5.  **Repeat or Return:** If there are no more tool calls or if `execute_tools` is False, the loop ends, and a `Response` object is returned. 

**7.3 Argument Descriptions:**

- `agent`: The (initial) agent to start the conversation.
- `messages`: A list of chat messages in the OpenAI format. This is the conversation history.
- `context_variables`:  A dictionary to store and share data between agents and functions. 
- `model_override`: A string to temporarily override the model used by the current agent.
- `stream`:  A boolean to enable streaming responses.
- `debug`: A boolean to turn on debug logging.
- `max_turns`: An integer that sets a limit on the number of conversational turns.
- `execute_tools`: A boolean that controls whether agent functions are executed (useful for testing).

**7.4 `Response` Object:**

```python
class Response(BaseModel):
    messages: List = []  
    agent: Optional[Agent] = None 
    context_variables: dict = {}
```

**7.5 Fields:**

- `messages`: The updated chat history, including messages from the agent, the user, and tool calls.
- `agent`: The last `Agent` that handled the conversation (in case of handoffs).
- `context_variables`:  The updated `context_variables` after function calls or handoffs.

**8. Agent Functions (Tools):**

-   Agent functions are regular Python functions that can be called by the model during a conversation.
-   **Explicit Function Return Type:** For clear communication with the language model, agent functions should primarily return strings. Swarm will attempt to convert other return types (e.g., dictionaries, lists) to strings, but this might not always produce the desired results. 
-   **Handoff Mechanism:** If a function returns an `Agent` object, the conversation is handed off to that new agent.
-   **Context Variables:** Functions can accept a `context_variables` argument to access and use the shared context.
-   **Tool Selection Control:** For finer control over which function the model should call, you can use the `tool_choice` parameter in the `Agent` constructor (refer to OpenAI API documentation on function calling for more details). 

**9. Advanced Features:**

**9.1 Agent Handoffs:**

Handoffs allow agents to delegate tasks to other agents. Here’s an example:

```python
if "refund" in user_input:
    refund_agent.handle_request(user_input)
else:
    support_agent.handle_request(user_input)
```

**9.2 Streaming Responses:**  Get real-time conversational output, ideal for chat-like interfaces:

    ```python
    for chunk in swarm.run(agent=triage_agent, messages=messages, stream=True):
        print(chunk, end="") # Print chunk by chunk as they arrive
    ```

**9.3 Streaming Event Types:**

   - **Standard Events:** Swarm uses the same streaming events as the Chat Completions API.
   - **Additional Events:**
     - `{"delim": "start"}` and `{"delim": "end"}` to mark the beginning and end of a single agent's response (useful for handoffs).
     - `{"response": Response}` at the end of the stream, containing the complete `Response` object for convenience. 

**9.4 Sophisticated Handoff Logic:** 

- Create complex workflows where agents collaborate and transfer control based on conversational flow. The airline customer service example (`examples/airline`) in the Swarm repo provides a great demonstration. 
- **User-Directed Handoffs:** Transfer the conversation based on explicit user requests (e.g., "Talk to the sales agent"). 
- **Agent-Determined Handoffs:** Have the agent decide to hand off based on the conversation context or specific keywords.

**Example: Agent-Determined Handoff:**

```python
def handle_complaint(context_variables, complaint_details):
    if "refund" in complaint_details.lower():
        return refunds_agent  # Implicitly handoff to the refunds agent
    # ... other logic ...
```

**9.5 Database and API Integration:** 

- Integrate your agents with external data sources and services to enhance their capabilities. 
- The personal shopper example (`examples/personal_shopper`) demonstrates SQLite database interaction.

**9.6 Parallel Tool Execution:**

- By default, Swarm allows agents to call multiple functions (tools) simultaneously if the model requests them.
- You can disable this by setting `parallel_tool_calls=False` in the `Agent` constructor. This forces the agent to execute functions sequentially.

**10.  Testing and Evaluation:**

Swarm encourages you to develop your own evaluation methods to assess your multi-agent systems. The framework provides some basic utilities to get you started.

**10.1 `run_demo_loop` Function:**

The `run_demo_loop` function offers an interactive command-line interface to test your agents:

```python
from swarm.repl import run_demo_loop

run_demo_loop(starting_agent)
```

**10.2 Evaluation Framework Example:**

The Airline and Weather Agent examples in the Swarm repo showcase a simple evaluation framework:

- You can define test cases with expected outcomes and compare them to the actual results.
- See the `evals.py` files in these examples for guidance. 

**Example: Unit Test for Tool Call:**

```python
from swarm import Swarm
from your_agents import my_agent
import pytest

client = Swarm()

def run_and_get_tool_calls(agent, query):
    message = {"role": "user", "content": query}
    response = client.run(agent, [message], execute_tools=False) # Note: execute_tools=False to prevent tool execution
    return response.messages[-1].get("tool_calls")

def test_my_agent_calls_correct_function():
    tool_calls = run_and_get_tool_calls(my_agent, "Perform the task")
    assert len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "expected_function_name"
``` 

**11. Troubleshooting:**

### 11.1 Common Issues and Solutions

1. **API Key Issues**: Ensure your OpenAI API key is correctly set as an environment variable.

2. **Rate Limiting**: Implement exponential backoff for API calls to handle rate limits gracefully.

3. **Unexpected Agent Behavior**: Review agent instructions and function implementations. Use debug mode to trace agent decision-making.

4. **Performance Issues**: Optimize prompt engineering, reduce unnecessary API calls, and consider caching frequently used information.

5. **Integration Problems**: When integrating external APIs or databases, ensure proper error handling and data validation.

- **Issue: Agent Not Responding**: Ensure the OpenAI API key is correctly set and that the agent has proper instructions and functions.
- **Issue: Handoffs Not Working**: Check that the conditions for handoff are well-defined and test using simple inputs first.

**12. API Reference:**

### 12.1 Swarm Class

```python
class Swarm:
    def __init__(self, client=None):
        """
        Initializes a Swarm client. 

        Args:
            client (Optional[OpenAI]): An optional OpenAI client instance. 
                                        If not provided, a new instance is created. 
        """
        if not client:
            client = OpenAI()
        self.client = client

    def run(self, agent, messages, context_variables={}, model_override=None, stream=False, debug=False, max_turns=float("inf"), execute_tools=True):
        """
        Runs the Swarm system. 

        Args:
            agent (Agent): The initial agent to start the conversation with.
            messages (list): A list of chat messages in OpenAI format.
            context_variables (dict, optional):  A dictionary to store and share context. 
                                                Defaults to an empty dictionary. 
            model_override (str, optional): A string to temporarily override the model used by 
                                            the current agent. Defaults to None.
            stream (bool, optional): A boolean to enable streaming responses. Defaults to False.
            debug (bool, optional): A boolean to turn on debug logging. Defaults to False.
            max_turns (int, optional): An integer that sets a limit on the number of conversational turns. 
                                        Defaults to infinity. 
            execute_tools (bool, optional):  A boolean that controls whether agent functions are 
                                            executed. Defaults to True.  
        
        Returns:
            Response: A `Response` object containing the results of the conversation.
        """
        # ... (see the Swarm repo for the full code) ... 
```

### 12.2 Agent Class

```python
class Agent:
    def __init__(self, name="Agent", model="gpt-4o", instructions="You are a helpful agent.", functions=[], tool_choice=None, parallel_tool_calls=True):
        """
        Initializes an agent.

        Args:
            name (str, optional):  A string that identifies the agent. Defaults to "Agent".
            model (str, optional):  The OpenAI language model the agent will use. Defaults to "gpt-4o".
            instructions (Union[str, Callable[[dict], str]], optional):  Instructions that guide the 
                                                                            agent's behavior. Can be a 
                                                                            string or a function that 
                                                                            returns a string. Defaults 
                                                                            to "You are a helpful 
                                                                            agent."
            functions (list, optional):  A list of callable functions the agent can call. Defaults to an
                                        empty list. 
            tool_choice (str, optional):  Advanced option for controlling which function the model should 
                                        call (see OpenAI API docs). Defaults to None. 
            parallel_tool_calls (bool, optional):  A boolean that indicates whether the agent can call 
                                                    multiple functions in parallel. Defaults to True. 
        """
        # ... (see the Swarm repo for the full code) ... 
```

### 12.3 Response Class

```python
class Response:
    def __init__(self, messages=[], agent=None, context_variables={}):
        """
        Represents a response from the Swarm system.

        Args:
            messages (list, optional): A list of chat messages. Defaults to an empty list.
            agent (Agent, optional):  The final agent in control of the conversation. Defaults to None.
            context_variables (dict, optional):  The updated context variables. Defaults to an empty 
                                                dictionary. 
        """
        # ... (see the Swarm repo for the full code) ... 
```

### 12.4 Result Class

```python
class Result:
    def __init__(self, value="", agent=None, context_variables={}):
        """
        Encapsulates the result of an agent function call.

        Args:
            value (str, optional):  The string value of the result. Defaults to "". 
            agent (Agent, optional):  The agent to hand off to (if applicable). Defaults to None. 
            context_variables (dict, optional):  Updated context variables. Defaults to an empty dictionary.
        """
        # ... (see the Swarm repo for the full code) ... 
```

### 12.5 Utility Functions

```python
def run_demo_loop(starting_agent, context_variables=None, stream=False, debug=False):
    """
    Runs an interactive demo loop for testing agents. 

    Args:
        starting_agent (Agent):  The agent to start the conversation with. 
        context_variables (dict, optional): Initial context variables. Defaults to None. 
        stream (bool, optional):  Enable streaming responses. Defaults to False.
        debug (bool, optional):  Enable debug logging. Defaults to False. 
    """
    # ... (see swarm/repl/repl.py for the full code) ... 

def function_to_json(func):
    """
    Converts a Python function to a JSON representation for OpenAI's function calling API. 

    Args:
        func (Callable):  The function to convert.

    Returns:
        dict: A dictionary representing the function in JSON format. 
    """
    # ... (see swarm/util.py for the full code) ... 

def debug_print(debug, *args):
    """
    Prints debug information if debug mode is enabled.

    Args:
        debug (bool):  A boolean indicating whether debug mode is active.
        *args:  Variable number of arguments to print.
    """
    # ... (see swarm/util.py for the full code) ... 
```

**13. Function Schemas:**

Swarm automatically handles the creation of JSON schemas for your agent functions, so you don't need to manually define them in the OpenAI format.

**13.1 Automatic Schema Generation:**

- **Docstrings to Descriptions:** Function docstrings become the `description` field in the schema.
- **Type Hints:** Swarm uses type hints to infer the `type` of each parameter.
- **Required Parameters:** Parameters without a default value are marked as `required` in the schema.

**13.2 Example:**

```python
def greet(name: str, age: int, location: str = "New York"):
   """Greets the user. 

   Args:
      name: Name of the user.
      age: Age of the user.
      location: The user's location.
   """
   return f"Hello {name}, glad you are {age} in {location}!"
```

**13.3 Swarm-Generated Schema:**

```json
{
   "type": "function",
   "function": {
      "name": "greet",
      "description": "Greets the user. ",
      "parameters": {
         "type": "object",
         "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "location": {"type": "string"}
         },
         "required": ["name", "age"] 
      }
   }
}
```

**14. Additional Resources:**

- **Swarm GitHub Repo:**  https://github.com/openai/swarm
- **OpenAI Chat Completions API Docs:** https://platform.openai.com/docs/api-reference/chat/create
- **OpenAI Function Calling Docs:** https://platform.openai.com/docs/guides/function-calling

**15. Conclusion:**

The Swarm framework provides a lightweight, flexible, and testable approach to orchestrating multi-agent systems with OpenAI's language models. While it's not yet production-ready, Swarm is a valuable educational tool for exploring the possibilities of agent coordination and building custom solutions for complex conversational AI tasks. 
