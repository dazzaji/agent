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

- **4.4 Handoffs:** Handoffs enable smooth transitions between agents. Let's define a **handoff** as an agent (or routine) handing off an active conversation to another agent, much like when you get transfered to someone else on a phone call. Except in this case, the agents have complete knowledge of your prior conversation!

To see handoffs in action, let's start by defining a basic class for an Agent.

```
class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o-mini"
    instructions: str = "You are a helpful Agent"
    tools: list = []
```

Now to make our code support it, we can change `run_full_turn` to take an `Agent` instead of separate `system_message` and `tools`:

```
def run_full_turn(agent, messages):

    num_init_messages = len(messages)
    messages = messages.copy()

    while True:

        # turn python functions into tools and save a reverse map
        tool_schemas = [function_to_schema(tool) for tool in agent.tools]
        tools_map = {tool.__name__: tool for tool in agent.tools}

        # === 1. get openai completion ===
        response = client.chat.completions.create(
            model=agent.model,
            messages=[{"role": "system", "content": agent.instructions}] + messages,
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
```

We can now run multiple agents easily:

```
def execute_refund(item_name):
    return "success"

refund_agent = Agent(
    name="Refund Agent",
    instructions="You are a refund agent. Help the user with refunds.",
    tools=[execute_refund],
)

def place_order(item_name):
    return "success"

sales_assistant = Agent(
    name="Sales Assistant",
    instructions="You are a sales assistant. Sell the user a product.",
    tools=[place_order],
)


messages = []
user_query = "Place an order for a black boot."
print("User:", user_query)
messages.append({"role": "user", "content": user_query})

response = run_full_turn(sales_assistant, messages) # sales assistant
messages.extend(response)


user_query = "Actually, I want a refund." # implitly refers to the last item
print("User:", user_query)
messages.append({"role": "user", "content": user_query})
response = run_full_turn(refund_agent, messages) # refund agent
```

User: Place an order for a black boot.
Assistant: place\_order({'item\_name': 'black boot'})
Assistant: Your order for a black boot has been successfully placed! If you need anything else, feel free to ask!
User: Actually, I want a refund.
Assistant: execute\_refund({'item\_name': 'black boot'})
Assistant: Your refund for the black boot has been successfully processed. If you need further assistance, just let me know!

Great! But we did the handoff manually here – we want the agents themselves to decide when to perform a handoff. A simple, but surprisingly effective way to do this is by giving them a `transfer_to_XXX` function, where `XXX` is some agent. The model is smart enough to know to call this function when it makes sense to make a handoff!

[Handoff Functions](https://cookbook.openai.com/examples/orchestrating_agents#handoff-functions)

Now that agent can express the _intent_ to make a handoff, we must make it actually happen. There's many ways to do this, but there's one particularly clean way.

For the agent functions we've defined so far, like `execute_refund` or `place_order` they return a string, which will be provided to the model. What if instead, we return an `Agent` object to indate which agent we want to transfer to? Like so:

```
refund_agent = Agent(
    name="Refund Agent",
    instructions="You are a refund agent. Help the user with refunds.",
    tools=[execute_refund],
)

def transfer_to_refunds():
    return refund_agent

sales_assistant = Agent(
    name="Sales Assistant",
    instructions="You are a sales assistant. Sell the user a product.",
    tools=[place_order],
)
```

We can then update our code to check the return type of a function response, and if it's an `Agent`, update the agent in use! Additionally, now `run_full_turn` will need to return the latest agent in use in case there are handoffs. (We can do this in a `Response` class to keep things neat.)

```
class Response(BaseModel):
    agent: Optional[Agent]
    messages: list
```

Now for the updated `run_full_turn`:

```
def run_full_turn(agent, messages):

    current_agent = agent
    num_init_messages = len(messages)
    messages = messages.copy()

    while True:

        # turn python functions into tools and save a reverse map
        tool_schemas = [function_to_schema(tool) for tool in current_agent.tools]
        tools = {tool.__name__: tool for tool in current_agent.tools}

        # === 1. get openai completion ===
        response = client.chat.completions.create(
            model=agent.model,
            messages=[{"role": "system", "content": current_agent.instructions}]
            + messages,
            tools=tool_schemas or None,
        )
        message = response.choices[0].message
        messages.append(message)

        if message.content:  # print agent response
            print(f"{current_agent.name}:", message.content)

        if not message.tool_calls:  # if finished handling tool calls, break
            break

        # === 2. handle tool calls ===

        for tool_call in message.tool_calls:
            result = execute_tool_call(tool_call, tools, current_agent.name)

            if type(result) is Agent:  # if agent transfer, update current agent
                current_agent = result
                result = (
                    f"Transfered to {current_agent.name}. Adopt persona immediately."
                )

            result_message = {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            }
            messages.append(result_message)

    # ==== 3. return last agent used and new messages =====
    return Response(agent=current_agent, messages=messages[num_init_messages:])


def execute_tool_call(tool_call, tools, agent_name):
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    print(f"{agent_name}:", f"{name}({args})")

    return tools[name](**args)  # call corresponding function with provided arguments
```

Let's look at an example with more Agents.

```
def escalate_to_human(summary):
    """Only call this if explicitly asked to."""
    print("Escalating to human agent...")
    print("\n=== Escalation Report ===")
    print(f"Summary: {summary}")
    print("=========================\n")
    exit()


def transfer_to_sales_agent():
    """User for anything sales or buying related."""
    return sales_agent


def transfer_to_issues_and_repairs():
    """User for issues, repairs, or refunds."""
    return issues_and_repairs_agent


def transfer_back_to_triage():
    """Call this if the user brings up a topic outside of your purview,
    including escalating to human."""
    return triage_agent


triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        "You are a customer service bot for ACME Inc. "
        "Introduce yourself. Always be very brief. "
        "Gather information to direct the customer to the right department. "
        "But make your questions subtle and natural."
    ),
    tools=[transfer_to_sales_agent, transfer_to_issues_and_repairs, escalate_to_human],
)


def execute_order(product, price: int):
    """Price should be in USD."""
    print("\n\n=== Order Summary ===")
    print(f"Product: {product}")
    print(f"Price: ${price}")
    print("=================\n")
    confirm = input("Confirm order? y/n: ").strip().lower()
    if confirm == "y":
        print("Order execution successful!")
        return "Success"
    else:
        print("Order cancelled!")
        return "User cancelled order."


sales_agent = Agent(
    name="Sales Agent",
    instructions=(
        "You are a sales agent for ACME Inc."
        "Always answer in a sentence or less."
        "Follow the following routine with the user:"
        "1. Ask them about any problems in their life related to catching roadrunners.\n"
        "2. Casually mention one of ACME's crazy made-up products can help.\n"
        " - Don't mention price.\n"
        "3. Once the user is bought in, drop a ridiculous price.\n"
        "4. Only after everything, and if the user says yes, "
        "tell them a crazy caveat and execute their order.\n"
        ""
    ),
    tools=[execute_order, transfer_back_to_triage],
)


def look_up_item(search_query):
    """Use to find item ID.
    Search query can be a description or keywords."""
    item_id = "item_132612938"
    print("Found item:", item_id)
    return item_id


def execute_refund(item_id, reason="not provided"):
    print("\n\n=== Refund Summary ===")
    print(f"Item ID: {item_id}")
    print(f"Reason: {reason}")
    print("=================\n")
    print("Refund execution successful!")
    return "success"


issues_and_repairs_agent = Agent(
    name="Issues and Repairs Agent",
    instructions=(
        "You are a customer support agent for ACME Inc."
        "Always answer in a sentence or less."
        "Follow the following routine with the user:"
        "1. First, ask probing questions and understand the user's problem deeper.\n"
        " - unless the user has already provided a reason.\n"
        "2. Propose a fix (make one up).\n"
        "3. ONLY if not satesfied, offer a refund.\n"
        "4. If accepted, search for the ID and then execute refund."
        ""
    ),
    tools=[execute_refund, look_up_item, transfer_back_to_triage],
)
```

Finally, we can run this in a loop (this won't run in python notebooks, so you can try this in a separate python file):

```
agent = triage_agent
messages = []

while True:
    user = input("User: ")
    messages.append({"role": "user", "content": user})

    response = run_full_turn(agent, messages)
    agent = response.agent
    messages.extend(response.messages)
```
**4.4.1: Dedicated `transfer_to_X` Functions**

A more robust and preferred method handoff demonstrated in the repository (specifically the `airline`, `support_bot`, and `triage_agent` examples) involves using dedicated `transfer_to_X` functions. This makes the handoff intent clear to the model.

```python
# Example from the triage_agent example
def transfer_to_sales():
    return sales_agent

def transfer_to_refunds():
    return refunds_agent

triage_agent.functions = [transfer_to_sales, transfer_to_refunds] 
```

The `create_triage_agent` function, simplifies the creation of triage agents with these transfer functions.

Note that functions must include a `context_variables` parameter in their definition if they are designed to utilize them.

```python
def my_function(context_variables, other_arg):
    # Access context variables here
    user_name = context_variables.get("user_name", "User")
    # ... rest of your function
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

The Swarm repository emphasizes the importance of explicit return types for agent functions. While strings are recommended for improved language model handling, other return types are possible, though potentially requiring additional handling.

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

Agent handoffs can be implicit, by returning the new agent from a tool function or by using the aforementioned transfer_to_X functions.


**9.5 Database and API Integration:** 

- Integrate your agents with external data sources and services to enhance their capabilities. 
- The personal shopper example (`examples/personal_shopper`) demonstrates SQLite database interaction.

**9.6 Parallel Tool Execution:**

- By default, Swarm allows agents to call multiple functions (tools) simultaneously if the model requests them.
- You can disable this by setting `parallel_tool_calls=False` in the `Agent` constructor. This forces the agent to execute functions sequentially.
- The repository's customer_service example provides additional practical insight into how functions are executed.


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

The Swarm repository showcases an interactive command-line interface using `run_demo_loop`. This simplifies testing and facilitates a more interactive development process.

```python
from swarm.repl import run_demo_loop
run_demo_loop(starting_agent)
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

------------------------------------

# README FROM SWARM REPO

------------------------------------

## README.md

# Swarm (experimental, educational)

An educational framework exploring ergonomic, lightweight multi-agent orchestration.

> [!WARNING]
> Swarm is currently an experimental sample framework intended to explore ergonomic interfaces for multi-agent systems. It is not intended to be used in production, and therefore has no official support. (This also means we will not be reviewing PRs or issues!)
>
> The primary goal of Swarm is to showcase the handoff & routines patterns explored in the [Orchestrating Agents: Handoffs & Routines](https://cookbook.openai.com/examples/orchestrating_agents) cookbook. It is not meant as a standalone library, and is primarily for educational purposes.

## Install

Requires Python 3.10+

```shell
pip install git+ssh://git@github.com/openai/swarm.git
```

or

```shell
pip install git+https://github.com/openai/swarm.git
```

## Usage

```python
from swarm import Swarm, Agent

client = Swarm()

def transfer_to_agent_b():
    return agent_b


agent_a = Agent(
    name="Agent A",
    instructions="You are a helpful agent.",
    functions=[transfer_to_agent_b],
)

agent_b = Agent(
    name="Agent B",
    instructions="Only speak in Haikus.",
)

response = client.run(
    agent=agent_a,
    messages=[{"role": "user", "content": "I want to talk to agent B."}],
)

print(response.messages[-1]["content"])
```

```
Hope glimmers brightly,
New paths converge gracefully,
What can I assist?
```

## Table of Contents

- [Overview](#overview)
- [Examples](#examples)
- [Documentation](#documentation)
  - [Running Swarm](#running-swarm)
  - [Agents](#agents)
  - [Functions](#functions)
  - [Streaming](#streaming)
- [Evaluations](#evaluations)
- [Utils](#utils)

# Overview

Swarm focuses on making agent **coordination** and **execution** lightweight, highly controllable, and easily testable.

It accomplishes this through two primitive abstractions: `Agent`s and **handoffs**. An `Agent` encompasses `instructions` and `tools`, and can at any point choose to hand off a conversation to another `Agent`.

These primitives are powerful enough to express rich dynamics between tools and networks of agents, allowing you to build scalable, real-world solutions while avoiding a steep learning curve.

> [!NOTE]
> Swarm Agents are not related to Assistants in the Assistants API. They are named similarly for convenience, but are otherwise completely unrelated. Swarm is entirely powered by the Chat Completions API and is hence stateless between calls.

## Why Swarm

Swarm explores patterns that are lightweight, scalable, and highly customizable by design. Approaches similar to Swarm are best suited for situations dealing with a large number of independent capabilities and instructions that are difficult to encode into a single prompt.

The Assistants API is a great option for developers looking for fully-hosted threads and built in memory management and retrieval. However, Swarm is an educational resource for developers curious to learn about multi-agent orchestration. Swarm runs (almost) entirely on the client and, much like the Chat Completions API, does not store state between calls.

# Examples

Check out `/examples` for inspiration! Learn more about each one in its README.

- [`basic`](examples/basic): Simple examples of fundamentals like setup, function calling, handoffs, and context variables
- [`triage_agent`](examples/triage_agent): Simple example of setting up a basic triage step to hand off to the right agent
- [`weather_agent`](examples/weather_agent): Simple example of function calling
- [`airline`](examples/airline): A multi-agent setup for handling different customer service requests in an airline context.
- [`support_bot`](examples/support_bot): A customer service bot which includes a user interface agent and a help center agent with several tools
- [`personal_shopper`](examples/personal_shopper): A personal shopping agent that can help with making sales and refunding orders

# Documentation

![Swarm Diagram](assets/swarm_diagram.png)

## Running Swarm

Start by instantiating a Swarm client (which internally just instantiates an `OpenAI` client).

```python
from swarm import Swarm

client = Swarm()
```

### `client.run()`

Swarm's `run()` function is analogous to the `chat.completions.create()` function in the Chat Completions API – it takes `messages` and returns `messages` and saves no state between calls. Importantly, however, it also handles Agent function execution, hand-offs, context variable references, and can take multiple turns before returning to the user.

At its core, Swarm's `client.run()` implements the following loop:

1. Get a completion from the current Agent
2. Execute tool calls and append results
3. Switch Agent if necessary
4. Update context variables, if necessary
5. If no new function calls, return

#### Arguments

| Argument              | Type    | Description                                                                                                                                            | Default        |
| --------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------- |
| **agent**             | `Agent` | The (initial) agent to be called.                                                                                                                      | (required)     |
| **messages**          | `List`  | A list of message objects, identical to [Chat Completions `messages`](https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages) | (required)     |
| **context_variables** | `dict`  | A dictionary of additional context variables, available to functions and Agent instructions                                                            | `{}`           |
| **max_turns**         | `int`   | The maximum number of conversational turns allowed                                                                                                     | `float("inf")` |
| **model_override**    | `str`   | An optional string to override the model being used by an Agent                                                                                        | `None`         |
| **execute_tools**     | `bool`  | If `False`, interrupt execution and immediately returns `tool_calls` message when an Agent tries to call a function                                    | `True`         |
| **stream**            | `bool`  | If `True`, enables streaming responses                                                                                                                 | `False`        |
| **debug**             | `bool`  | If `True`, enables debug logging                                                                                                                       | `False`        |

Once `client.run()` is finished (after potentially multiple calls to agents and tools) it will return a `Response` containing all the relevant updated state. Specifically, the new `messages`, the last `Agent` to be called, and the most up-to-date `context_variables`. You can pass these values (plus new user messages) in to your next execution of `client.run()` to continue the interaction where it left off – much like `chat.completions.create()`. (The `run_demo_loop` function implements an example of a full execution loop in `/swarm/repl/repl.py`.)

#### `Response` Fields

| Field                 | Type    | Description                                                                                                                                                                                                                                                                  |
| --------------------- | ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **messages**          | `List`  | A list of message objects generated during the conversation. Very similar to [Chat Completions `messages`](https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages), but with a `sender` field indicating which `Agent` the message originated from. |
| **agent**             | `Agent` | The last agent to handle a message.                                                                                                                                                                                                                                          |
| **context_variables** | `dict`  | The same as the input variables, plus any changes.                                                                                                                                                                                                                           |

## Agents

An `Agent` simply encapsulates a set of `instructions` with a set of `functions` (plus some additional settings below), and has the capability to hand off execution to another `Agent`.

While it's tempting to personify an `Agent` as "someone who does X", it can also be used to represent a very specific workflow or step defined by a set of `instructions` and `functions` (e.g. a set of steps, a complex retrieval, single step of data transformation, etc). This allows `Agent`s to be composed into a network of "agents", "workflows", and "tasks", all represented by the same primitive.

## `Agent` Fields

| Field            | Type                     | Description                                                                   | Default                      |
| ---------------- | ------------------------ | ----------------------------------------------------------------------------- | ---------------------------- |
| **name**         | `str`                    | The name of the agent.                                                        | `"Agent"`                    |
| **model**        | `str`                    | The model to be used by the agent.                                            | `"gpt-4o"`                   |
| **instructions** | `str` or `func() -> str` | Instructions for the agent, can be a string or a callable returning a string. | `"You are a helpful agent."` |
| **functions**    | `List`                   | A list of functions that the agent can call.                                  | `[]`                         |
| **tool_choice**  | `str`                    | The tool choice for the agent, if any.                                        | `None`                       |

### Instructions

`Agent` `instructions` are directly converted into the `system` prompt of a conversation (as the first message). Only the `instructions` of the active `Agent` will be present at any given time (e.g. if there is an `Agent` handoff, the `system` prompt will change, but the chat history will not.)

```python
agent = Agent(
   instructions="You are a helpful agent."
)
```

The `instructions` can either be a regular `str`, or a function that returns a `str`. The function can optionally receive a `context_variables` parameter, which will be populated by the `context_variables` passed into `client.run()`.

```python
def instructions(context_variables):
   user_name = context_variables["user_name"]
   return f"Help the user, {user_name}, do whatever they want."

agent = Agent(
   instructions=instructions
)
response = client.run(
   agent=agent,
   messages=[{"role":"user", "content": "Hi!"}],
   context_variables={"user_name":"John"}
)
print(response.messages[-1]["content"])
```

```
Hi John, how can I assist you today?
```

## Functions

- Swarm `Agent`s can call python functions directly.
- Function should usually return a `str` (values will be attempted to be cast as a `str`).
- If a function returns an `Agent`, execution will be transferred to that `Agent`.
- If a function defines a `context_variables` parameter, it will be populated by the `context_variables` passed into `client.run()`.

```python
def greet(context_variables, language):
   user_name = context_variables["user_name"]
   greeting = "Hola" if language.lower() == "spanish" else "Hello"
   print(f"{greeting}, {user_name}!")
   return "Done"

agent = Agent(
   functions=[greet]
)

client.run(
   agent=agent,
   messages=[{"role": "user", "content": "Usa greet() por favor."}],
   context_variables={"user_name": "John"}
)
```

```
Hola, John!
```

- If an `Agent` function call has an error (missing function, wrong argument, error) an error response will be appended to the chat so the `Agent` can recover gracefully.
- If multiple functions are called by the `Agent`, they will be executed in that order.

### Handoffs and Updating Context Variables

An `Agent` can hand off to another `Agent` by returning it in a `function`.

```python
sales_agent = Agent(name="Sales Agent")

def transfer_to_sales():
   return sales_agent

agent = Agent(functions=[transfer_to_sales])

response = client.run(agent, [{"role":"user", "content":"Transfer me to sales."}])
print(response.agent.name)
```

```
Sales Agent
```

It can also update the `context_variables` by returning a more complete `Result` object. This can also contain a `value` and an `agent`, in case you want a single function to return a value, update the agent, and update the context variables (or any subset of the three).

```python
sales_agent = Agent(name="Sales Agent")

def talk_to_sales():
   print("Hello, World!")
   return Result(
       value="Done",
       agent=sales_agent,
       context_variables={"department": "sales"}
   )

agent = Agent(functions=[talk_to_sales])

response = client.run(
   agent=agent,
   messages=[{"role": "user", "content": "Transfer me to sales"}],
   context_variables={"user_name": "John"}
)
print(response.agent.name)
print(response.context_variables)
```

```
Sales Agent
{'department': 'sales', 'user_name': 'John'}
```

> [!NOTE]
> If an `Agent` calls multiple functions to hand-off to an `Agent`, only the last handoff function will be used.

### Function Schemas

Swarm automatically converts functions into a JSON Schema that is passed into Chat Completions `tools`.

- Docstrings are turned into the function `description`.
- Parameters without default values are set to `required`.
- Type hints are mapped to the parameter's `type` (and default to `string`).
- Per-parameter descriptions are not explicitly supported, but should work similarly if just added in the docstring. (In the future docstring argument parsing may be added.)

```python
def greet(name, age: int, location: str = "New York"):
   """Greets the user. Make sure to get their name and age before calling.

   Args:
      name: Name of the user.
      age: Age of the user.
      location: Best place on earth.
   """
   print(f"Hello {name}, glad you are {age} in {location}!")
```

```javascript
{
   "type": "function",
   "function": {
      "name": "greet",
      "description": "Greets the user. Make sure to get their name and age before calling.\n\nArgs:\n   name: Name of the user.\n   age: Age of the user.\n   location: Best place on earth.",
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

## Streaming

```python
stream = client.run(agent, messages, stream=True)
for chunk in stream:
   print(chunk)
```

Uses the same events as [Chat Completions API streaming](https://platform.openai.com/docs/api-reference/streaming). See `process_and_print_streaming_response` in `/swarm/repl/repl.py` as an example.

Two new event types have been added:

- `{"delim":"start"}` and `{"delim":"end"}`, to signal each time an `Agent` handles a single message (response or function call). This helps identify switches between `Agent`s.
- `{"response": Response}` will return a `Response` object at the end of a stream with the aggregated (complete) response, for convenience.

# Evaluations

Evaluations are crucial to any project, and we encourage developers to bring their own eval suites to test the performance of their swarms. For reference, we have some examples for how to eval swarm in the `airline`, `weather_agent` and `triage_agent` quickstart examples. See the READMEs for more details.

# Utils

Use the `run_demo_loop` to test out your swarm! This will run a REPL on your command line. Supports streaming.

```python
from swarm.repl import run_demo_loop
...
run_demo_loop(agent, stream=True)
```



------------------------------------


# EXAMPLES FROM SWARM REPO

------------------------------------


# Personal Shopper

## examples/personal_shopper/README.md

# Personal shopper

This Swarm is a personal shopping agent that can help with making sales and refunding orders.
This example uses the helper function `run_demo_loop`, which allows us to create an interactive Swarm session.
In this example, we also use a Sqlite3 database with customer information and transaction data.

## Overview

The personal shopper example includes three main agents to handle various customer service requests:

1. **Triage Agent**: Determines the type of request and transfers to the appropriate agent.
2. **Refund Agent**: Manages customer refunds, requiring both user ID and item ID to initiate a refund.
3. **Sales Agent**: Handles actions related to placing orders, requiring both user ID and product ID to complete a purchase.

## Setup

Once you have installed dependencies and Swarm, run the example using:

```shell
python3 main.py
```

---

## examples/personal_shopper/main.py

import datetime
import random

import database
from swarm import Agent
from swarm.agents import create_triage_agent
from swarm.repl import run_demo_loop


def refund_item(user_id, item_id):
    """Initiate a refund based on the user ID and item ID.
    Takes as input arguments in the format '{"user_id":"1","item_id":"3"}'
    """
    conn = database.get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT amount FROM PurchaseHistory
        WHERE user_id = ? AND item_id = ?
    """,
        (user_id, item_id),
    )
    result = cursor.fetchone()
    if result:
        amount = result[0]
        print(f"Refunding ${amount} to user ID {user_id} for item ID {item_id}.")
    else:
        print(f"No purchase found for user ID {user_id} and item ID {item_id}.")
    print("Refund initiated")


def notify_customer(user_id, method):
    """Notify a customer by their preferred method of either phone or email.
    Takes as input arguments in the format '{"user_id":"1","method":"email"}'"""

    conn = database.get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT email, phone FROM Users
        WHERE user_id = ?
    """,
        (user_id,),
    )
    user = cursor.fetchone()
    if user:
        email, phone = user
        if method == "email" and email:
            print(f"Emailed customer {email} a notification.")
        elif method == "phone" and phone:
            print(f"Texted customer {phone} a notification.")
        else:
            print(f"No {method} contact available for user ID {user_id}.")
    else:
        print(f"User ID {user_id} not found.")


def order_item(user_id, product_id):
    """Place an order for a product based on the user ID and product ID.
    Takes as input arguments in the format '{"user_id":"1","product_id":"2"}'"""
    date_of_purchase = datetime.datetime.now()
    item_id = random.randint(1, 300)

    conn = database.get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT product_id, product_name, price FROM Products
        WHERE product_id = ?
    """,
        (product_id,),
    )
    result = cursor.fetchone()
    if result:
        product_id, product_name, price = result
        print(
            f"Ordering product {product_name} for user ID {user_id}. The price is {price}."
        )
        # Add the purchase to the database
        database.add_purchase(user_id, date_of_purchase, item_id, price)
    else:
        print(f"Product {product_id} not found.")


# Initialize the database
database.initialize_database()

# Preview tables
database.preview_table("Users")
database.preview_table("PurchaseHistory")
database.preview_table("Products")

# Define the agents

refunds_agent = Agent(
    name="Refunds Agent",
    description=f"""You are a refund agent that handles all actions related to refunds after a return has been processed.
    You must ask for both the user ID and item ID to initiate a refund. Ask for both user_id and item_id in one message.
    If the user asks you to notify them, you must ask them what their preferred method of notification is. For notifications, you must
    ask them for user_id and method in one message.""",
    functions=[refund_item, notify_customer],
)

sales_agent = Agent(
    name="Sales Agent",
    description=f"""You are a sales agent that handles all actions related to placing an order to purchase an item.
    Regardless of what the user wants to purchase, must ask for BOTH the user ID and product ID to place an order.
    An order cannot be placed without these two pieces of information. Ask for both user_id and product_id in one message.
    If the user asks you to notify them, you must ask them what their preferred method is. For notifications, you must
    ask them for user_id and method in one message.
    """,
    functions=[order_item, notify_customer],
)

triage_agent = create_triage_agent(
    name="Triage Agent",
    instructions=f"""You are to triage a users request, and call a tool to transfer to the right intent.
    Once you are ready to transfer to the right intent, call the tool to transfer to the right intent.
    You dont need to know specifics, just the topic of the request.
    If the user request is about making an order or purchasing an item, transfer to the Sales Agent.
    If the user request is about getting a refund on an item or returning a product, transfer to the Refunds Agent.
    When you need more information to triage the request to an agent, ask a direct question without explaining why you're asking it.
    Do not share your thought process with the user! Do not make unreasonable assumptions on behalf of user.""",
    agents=[sales_agent, refunds_agent],
    add_backlinks=True,
)

for f in triage_agent.functions:
    print(f.__name__)

if __name__ == "__main__":
    # Run the demo loop
    run_demo_loop(triage_agent, debug=False)


---

## examples/personal_shopper/database.py

import sqlite3

# global connection
conn = None


def get_connection():
    global conn
    if conn is None:
        conn = sqlite3.connect("application.db")
    return conn


def create_database():
    # Connect to a single SQLite database
    conn = get_connection()
    cursor = conn.cursor()

    # Create Users table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS Users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            first_name TEXT,
            last_name TEXT,
            email TEXT UNIQUE,
            phone TEXT
        )
    """
    )

    # Create PurchaseHistory table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS PurchaseHistory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            date_of_purchase TEXT,
            item_id INTEGER,
            amount REAL,
            FOREIGN KEY (user_id) REFERENCES Users(user_id)
        )
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS Products (
            product_id INTEGER PRIMARY KEY,
            product_name TEXT NOT NULL,
            price REAL NOT NULL
        );
        """
    )

    # Save (commit) the changes
    conn.commit()


def add_user(user_id, first_name, last_name, email, phone):
    conn = get_connection()
    cursor = conn.cursor()

    # Check if the user already exists
    cursor.execute("SELECT * FROM Users WHERE user_id = ?", (user_id,))
    if cursor.fetchone():
        return

    try:
        cursor.execute(
            """
            INSERT INTO Users (user_id, first_name, last_name, email, phone)
            VALUES (?, ?, ?, ?, ?)
        """,
            (user_id, first_name, last_name, email, phone),
        )

        conn.commit()
    except sqlite3.Error as e:
        print(f"Database Error: {e}")


def add_purchase(user_id, date_of_purchase, item_id, amount):
    conn = get_connection()
    cursor = conn.cursor()

    # Check if the purchase already exists
    cursor.execute(
        """
        SELECT * FROM PurchaseHistory
        WHERE user_id = ? AND item_id = ? AND date_of_purchase = ?
    """,
        (user_id, item_id, date_of_purchase),
    )
    if cursor.fetchone():
        # print(f"Purchase already exists for user_id {user_id} on {date_of_purchase} for item_id {item_id}.")
        return

    try:
        cursor.execute(
            """
            INSERT INTO PurchaseHistory (user_id, date_of_purchase, item_id, amount)
            VALUES (?, ?, ?, ?)
        """,
            (user_id, date_of_purchase, item_id, amount),
        )

        conn.commit()
    except sqlite3.Error as e:
        print(f"Database Error: {e}")


def add_product(product_id, product_name, price):
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
        INSERT INTO Products (product_id, product_name, price)
        VALUES (?, ?, ?);
        """,
            (product_id, product_name, price),
        )

        conn.commit()
    except sqlite3.Error as e:
        print(f"Database Error: {e}")


def close_connection():
    global conn
    if conn:
        conn.close()
        conn = None


def preview_table(table_name):
    conn = sqlite3.connect("application.db")  # Replace with your database name
    cursor = conn.cursor()

    cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")  # Limit to first 5 rows

    rows = cursor.fetchall()

    for row in rows:
        print(row)

    conn.close()


# Initialize and load database
def initialize_database():
    global conn

    # Initialize the database tables
    create_database()

    # Add some initial users
    initial_users = [
        (1, "Alice", "Smith", "alice@test.com", "123-456-7890"),
        (2, "Bob", "Johnson", "bob@test.com", "234-567-8901"),
        (3, "Sarah", "Brown", "sarah@test.com", "555-567-8901"),
        # Add more initial users here
    ]

    for user in initial_users:
        add_user(*user)

    # Add some initial purchases
    initial_purchases = [
        (1, "2024-01-01", 101, 99.99),
        (2, "2023-12-25", 100, 39.99),
        (3, "2023-11-14", 307, 49.99),
    ]

    for purchase in initial_purchases:
        add_purchase(*purchase)

    initial_products = [
        (7, "Hat", 19.99),
        (8, "Wool socks", 29.99),
        (9, "Shoes", 39.99),
    ]

    for product in initial_products:
        add_product(*product)

-----

# Basic Agent in Swarm

---

## examples/basic/README.md

# Swarm basic

This folder contains basic examples demonstrating core Swarm capabilities. These examples show the simplest implementations of Swarm, with one input message, and a corresponding output. The `simple_loop_no_helpers` has a while loop to demonstrate how to create an interactive Swarm session.

### Examples

1. **agent_handoff.py**

   - Demonstrates how to transfer a conversation from one agent to another.
   - **Usage**: Transfers Spanish-speaking users from an English agent to a Spanish agent.

2. **bare_minimum.py**

   - A bare minimum example showing the basic setup of an agent.
   - **Usage**: Sets up an agent that responds to a simple user message.

3. **context_variables.py**

   - Shows how to use context variables within an agent.
   - **Usage**: Uses context variables to greet a user by name and print account details.

4. **function_calling.py**

   - Demonstrates how to define and call functions from an agent.
   - **Usage**: Sets up an agent that can respond with weather information for a given location.

5. **simple_loop_no_helpers.py**
   - An example of a simple interaction loop without using helper functions.
   - **Usage**: Sets up a loop where the user can continuously interact with the agent, printing the conversation.

## Running the Examples

To run any of the examples, use the following command:

```shell
python3 <example_name>.py
```

---

## examples/basic/agent_handoff.py

from swarm import Swarm, Agent

client = Swarm()

english_agent = Agent(
    name="English Agent",
    instructions="You only speak English.",
)

spanish_agent = Agent(
    name="Spanish Agent",
    instructions="You only speak Spanish.",
)


def transfer_to_spanish_agent():
    """Transfer spanish speaking users immediately."""
    return spanish_agent


english_agent.functions.append(transfer_to_spanish_agent)

messages = [{"role": "user", "content": "Hola. ¿Como estás?"}]
response = client.run(agent=english_agent, messages=messages)

print(response.messages[-1]["content"])

---

## examples/basic/bare_minimum.py

from swarm import Swarm, Agent

client = Swarm()

agent = Agent(
    name="Agent",
    instructions="You are a helpful agent.",
)

messages = [{"role": "user", "content": "Hi!"}]
response = client.run(agent=agent, messages=messages)

print(response.messages[-1]["content"])

---

## examples/basic/context_variables.py

from swarm import Swarm, Agent

client = Swarm()


def instructions(context_variables):
    name = context_variables.get("name", "User")
    return f"You are a helpful agent. Greet the user by name ({name})."


def print_account_details(context_variables: dict):
    user_id = context_variables.get("user_id", None)
    name = context_variables.get("name", None)
    print(f"Account Details: {name} {user_id}")
    return "Success"


agent = Agent(
    name="Agent",
    instructions=instructions,
    functions=[print_account_details],
)

context_variables = {"name": "James", "user_id": 123}

response = client.run(
    messages=[{"role": "user", "content": "Hi!"}],
    agent=agent,
    context_variables=context_variables,
)
print(response.messages[-1]["content"])

response = client.run(
    messages=[{"role": "user", "content": "Print my account details!"}],
    agent=agent,
    context_variables=context_variables,
)
print(response.messages[-1]["content"])

---

## examples/basic/function_calling.py

from swarm import Swarm, Agent

client = Swarm()


def get_weather(location) -> str:
    return "{'temp':67, 'unit':'F'}"


agent = Agent(
    name="Agent",
    instructions="You are a helpful agent.",
    functions=[get_weather],
)

messages = [{"role": "user", "content": "What's the weather in NYC?"}]

response = client.run(agent=agent, messages=messages)
print(response.messages[-1]["content"])

---

## examples/basic/simple_loop_no_helpers.py

from swarm import Swarm, Agent

client = Swarm()

my_agent = Agent(
    name="Agent",
    instructions="You are a helpful agent.",
)


def pretty_print_messages(messages):
    for message in messages:
        if message["content"] is None:
            continue
        print(f"{message['sender']}: {message['content']}")


messages = []
agent = my_agent
while True:
    user_input = input("> ")
    messages.append({"role": "user", "content": user_input})

    response = client.run(agent=agent, messages=messages)
    messages = response.messages
    agent = response.agent
    pretty_print_messages(messages)
