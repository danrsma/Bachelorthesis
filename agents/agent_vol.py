from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END

from typing import TypedDict
import base64
from io import BytesIO
from PIL import Image
import asyncio
import tkinter as tk
from tkinter import filedialog



class InputApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # Get Basic Tkinter Input Window
        self.title("Picture or Prompt")
        self.geometry("400x250")
        self.resizable(False, False)

        container = tk.Frame(self, padx=20, pady=20)
        container.pack(fill="both", expand=True)

        input_frame = tk.Frame(container)
        input_frame.pack(fill="x", pady=(0, 15))

        tk.Label(input_frame, text="Prompt:", width=12, anchor="w").grid(row=1, column=0, sticky="w", pady=5)
        self.entry = tk.Entry(input_frame)
        self.entry.grid(row=1, column=1, sticky="ew", padx=5)

        input_frame.columnconfigure(1, weight=1)

        file_frame = tk.Frame(container)
        file_frame.pack(fill="x", pady=(0, 15))

        tk.Button(file_frame, text="Picture", command=self.choose_file).pack(side="left")
        self.file_path = tk.StringVar()
        self.file_label = tk.Label(file_frame, textvariable=self.file_path, anchor="w")
        self.file_label.pack(side="left", padx=10, fill="x", expand=True)

        submit_btn = tk.Button(container, text="Submit", command=self.submit)
        submit_btn.pack(pady=10)

        # Initialize Variables
        self.user_input = None
        self.selected_file_path = None

    def choose_file(self):

        path = filedialog.askopenfilename(title="Choose a Picture")
        if path:
            self.file_path.set(path)

    def submit(self):

        self.user_input = str(self.entry.get())

        self.selected_file_path = str(self.file_path.get())

        # Close The Window After Submit
        self.destroy()


class MyState(TypedDict):

    # Pass States Through Stategraph
    vision: str
    code: str
    filepath: str
    userinput: str
    promptvision: str
    promptcode: str
    error: str


def prompt_func(data):

    # Chain Image And Text Data
    text = data["text"]
    image = data["image"]

    image_part = {
        "type": "image_url",
        "image_url": f"data:image/png;base64,{image}",
    }

    content_parts = []

    text_part = {"type": "text", "text": text}

    content_parts.append(image_part)
    content_parts.append(text_part)

    return [HumanMessage(content=content_parts)]


def convert_to_base64(pil_image):

    # Convert PIL Images To Base64 Encoded Strings
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


async def vision_llm_func(state: MyState) -> MyState:

    # Get Image Data
    file_path = state["filepath"]
    user_input = state["userinput"]
    if file_path == "":
        # Create Vision Agent Chain
        vision_llm_chat = ChatOllama(
            model="gemma3:12b",
            temperature=0.9,
        )

        prompt = """Provide a detailed and extensive for the scene.
            List all assets like hdris, models and textures you need to create the scene.
            """
        # Get Agent Chain Result
        vision_result = chain.invoke(user_input+prompt)

        print("\n")
        print("ImageLLM Output:")
        print("\n")
        print(vision_result.content)
        print("\n")

        state["vision"] = vision_result.content

        return state
    try:

        pil_image = Image.open(file_path)

    except Exception as e:
        print(f"Error in main execution: {e}")


    image_b64= convert_to_base64(pil_image)

    # Create Vision Agent Chain
    vision_llm_chat = ChatOllama(
        model="gemma3:12b",
        temperature=0.9,
    )

    prompt_func_runnable = RunnableLambda(prompt_func)
    chain = prompt_func_runnable | vision_llm_chat


    # Get Agent Chain Result
    vision_result = chain.invoke({
        "text": state["promptvision"],
        "image": image_b64,
    })

    print("\n")
    print("ImageLLM Output:")
    print("\n")
    print(vision_result.content)
    print("\n")

    state["vision"] = vision_result.content

    return state

def code_llm_func(state):

    # Create Code Agent
    code_llm_chat = ChatOllama(
        model="hf.co/mradermacher/BlenderLLM-GGUF:Q8_0",
        temperature=0.9,
    )

    # Get Agent Result
    code_llm_chat_input = state["vision"]+"\n"+state["code"]+"\n"+state["promptcode"]
    code_result = code_llm_chat.invoke(code_llm_chat_input)

    print("\n")
    print("CodeLLM Output:")
    print("\n")
    print(code_result.content)
    print("\n")
    state["code"] = code_result.content

    return state


async def tools_llm_func(state):

    # Get MCP-Tools From Server
    '''
    client = MultiServerMCPClient(
        {
            "blender_mcp": {
                "command": "firejail",
                "args": ["uvx", "blender-mcp", "--private", "--net=none", "--caps.drop=all", "--seccomp", "--private-dev", "--hostname=sandbox"],
                "transport": "stdio",
            }
        }
    )
    '''
    client = MultiServerMCPClient(
        {
            "blender_mcp": {
                "command": "uvx",
                "args": ["blender-mcp"],
                "transport": "stdio",
            }
        }
    )
    try:

        tools = await client.get_tools()

    except Exception as e:
        print(f"Error in main execution: {e}")

    # Create Tool Agent
    tools_llm_chat = ChatOllama(
        model="qwen3:8b",
        temperature=0.0,
    )
    agent = create_react_agent(
        model = tools_llm_chat,
        tools=tools
    )

    # Get Agent Result
    try:
        tool_result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Execute the following Blender Python Code:\n"+state["code"]}]}
        )

    except Exception as e:
        print(f"Error in main execution: {e}")

    # Make Viewport Screenshot
    screenshot_code = """
        import bpy

        # Create a new camera object
        cam_data = bpy.data.cameras.new(name="MyCamera")
        cam_object = bpy.data.objects.new("MyCamera", cam_data)

        # Set camera location and rotation
        cam_object.location = (0, -10, 7)
        cam_object.rotation_euler = (1.1, 0, 0)

        # Link the camera to the current scene
        bpy.context.collection.objects.link(cam_object)

        # Set the new camera as the active camera
        bpy.context.scene.camera = cam_object

        bpy.context.scene.render.filepath = "C:\\Users\\cross\\Desktop\\Render.png"
        bpy.ops.render.render(write_still=True)

        """
    try:
        tool_result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Execute the following Blender Python Code:\n"+screenshot_code+
            "\nIf it does not work try to fix and reexecute it."}]}
        )
    except Exception as e:
        print(f"Error in main execution: {e}")


    state["error"] = tool_result

    return state


async def main():

    # Open Input Window
    app = InputApp()
    app.mainloop()

    # Output Collected Inputs In Terminal
    print("Inputs collected:")
    print("llm_prompt =", app.user_input)
    print("selected_file_path =", app.selected_file_path)
    file_path = app.selected_file_path
    user_input = app.user_input

    # Loop Until At Least One Input Collected
    while file_path == "" and user_input == "":
        app = InputApp()
        app.mainloop()
        print("Inputs collected:")
        print("llm_prompt =", app.user_input)
        print("selected_file_path =", app.selected_file_path)
        file_path = app.selected_file_path
        user_input = app.user_input


    # Create StateGraph With Nodes And Edges
    graph = StateGraph(MyState)
    graph.add_node("vision_llm", vision_llm_func)
    graph.add_node("code_llm", code_llm_func)
    graph.add_node("tools_llm", tools_llm_func)
    graph.add_edge(START,"vision_llm")
    graph.add_edge("vision_llm", "code_llm")
    graph.add_edge("code_llm", "tools_llm")
    graph.add_edge("tools_llm",END)
    graph = graph.compile()

    # Get StateGraph Output State
    prompt_vision = """Provide a detailed and extensive description of the image.
        Describe every object in the picture accurately.
        Describe the shape of the lanscape elements."""
    prompt_code = "Create Blender Code of the described Landscape. Create every Object and Shape with math."
    input_state = MyState(userinput=user_input,filepath=file_path,promptcode=prompt_code,promptvision=prompt_vision,code="")
    output_state = await graph.ainvoke(input_state)
    print(output_state)

    # Prepare Rendering Loop
    file_path_loop = "C:\\Users\\cross\\Desktop\\Render.png"
    prompt_vision_loop = "How does image compare to the the discription? What are the differences?"
    prompt_code_loop = """The new image is the result of the provided Blender Code.
        Improve the Blender Code to minimize the differences.
        Also look at the errors during the first execution and try to avoid them.
        """
    output_state["filepath"] = file_path_loop
    output_state["promptvision"] = output_state["userinput"]+output_state["vision"]+prompt_vision_loop
    output_state["promptcode"] = prompt_code_loop

    input_state = output_state
    print(input_state)

    # Start Rendering Loop
    for i in range(4):
        print("\n")
        print(f"+++++++++++++++++++++++++++++++")
        print(f"+ Rendering Loop iteration: {str(i+2)} +")
        print(f"+++++++++++++++++++++++++++++++")
        print("\n")
        output_state = await graph.ainvoke(input_state)
        print(output_state)
        input_state = output_state


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
    
