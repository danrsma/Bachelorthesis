from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.runnables import RunnableLambda
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

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
    filepath: str
    userinput: str
    promptvision: str
    promptplan: str


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
    
    
    if file_path == "":
        # Create Vision Agent Chain
        vision_llm_chat = ChatOllama(
            model="llama4:maverick",
            base_url="http://localhost:11434",
            temperature=0.5,
        )

        # Get Agent Result
        prompt = """You are an expert in image analysis, 3D modeling, and Blender scripting. 
            Provide a detailed and enhanced description of the userinput and list all assets including hdri, models and textures you will need to create it."""+state["userinput"]
        vision_result = vision_llm_chat.invoke(prompt)

        # Ouput Image LLM
        print("\n")
        print("ImageLLM Output:")
        print("\n")
        print(vision_result.content)
        print("\n")

        state["vision"] = vision_result.content
        return state
    
    
    # Open Image
    try:

        pil_image = Image.open(file_path)

    except Exception as e:
        print(f"Error in main execution: {e}")


    image_b64= convert_to_base64(pil_image)

    # Create Vision Agent Chain
    vision_llm_chat = ChatOllama(
        model="llama4:maverick",
        base_url="http://localhost:11434",
        temperature=0.5,
    )

    prompt_func_runnable = RunnableLambda(prompt_func)
    chain = prompt_func_runnable | vision_llm_chat


    # Get Agent Chain Result
    vision_result = chain.invoke({
        "text": state["promptvision"],
        "image": image_b64,
    })

    # Ouput Image LLM
    print("\n")
    print("ImageLLM Output:")
    print("\n")
    print(vision_result.content)
    print("\n")

    state["vision"] = vision_result.content

    return state

async def plan_llm_func(state):

    # Get MCP-Tools From Server
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

    # Filter The Tools
    tools = [
        t for t in tools
        if t.name not in {"get_hyper3d_status", "get_sketchfab_status", "search_sketchfab_models","download_sketchfab_models","generate_hyper3d_model_via_text","generate_hyper3d_model_via_images","poll_rodin_job_status","import_generated_asset"}
    ]

    # Create Agent
    tools_llm_chat = ChatOllama(
        model="qwen3:235b",
        base_url="http://localhost:11434", 
        temperature=0.5,
    )
    agent = create_react_agent(
        model = tools_llm_chat,
        tools=tools
    )

    plan_llm_chat_input = state["prompt_plan"]+"\n"+state["vision"]

    # Get Agent Result
    try:
        await agent.ainvoke(
            {"messages": [{"role": "user", "content": plan_llm_chat_input}]}
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
        await agent.ainvoke(
            {"messages": [{"role": "user", "content": "Execute the following Blender Python Code:\n"+screenshot_code+
            "\nIf it does not work try to fix and reexecute it."}]}
        )
        print("Screenshot taken.")
        print("\n")

    except Exception as e:
        print(f"Error in main execution: {e}")


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
    graph.add_node("plan_llm", plan_llm_func)
    graph.add_edge(START,"vision_llm")
    graph.add_edge("vision_llm", "plan_llm")
    graph.add_edge("plan_llm",END)
    graph = graph.compile()

    # Get StateGraph Output State
    prompt_vision = """You are an expert in image analysis, 3D modeling, and Blender scripting. 
            Provide a detailed and extensive description of the image and list all assets including hdri, models and textures you will need to create it."""

    prompt_plan = """You are an expert in image analysis, 3D modeling, and Blender scripting.
        Recreate the provided Scene in Blender. Use Polyhaven assets and Blender Code Execution
        """
    

    input_state = MyState(userinput=user_input,filepath=file_path,promptplan=prompt_plan,promptvision=prompt_vision)
    output_state = await graph.ainvoke(input_state)

    # Prepare Rendering Loop
    file_path_loop = "C:\\Users\\cross\\Desktop\\Render.png"
    
    prompt_vision_loop = """You are an expert in image analysis, 3D modeling, and Blender scripting. 
            Provide a detailed comparison of the image and the discription.
            Mark out all the differences. Provide a better discription and list of assets.
            """
    
    prompt_plan_loop = """You are an expert in image analysis, 3D modeling, and Blender scripting. 
            Improve the Scene in Blender to minimize the differences."""
 
    output_state["filepath"] = file_path_loop
    output_state["promptvision"] = output_state["vision"]+prompt_vision_loop
    output_state["promptplan"] = prompt_plan_loop

    input_state = output_state

    # Start Rendering Loop
    for i in range(4):
        print("\n")
        print(f"++++++++++++++++++++++++++++++++++++++")
        print(f"+ Rendering Loop iteration: {str(i+2)} +")
        print(f"++++++++++++++++++++++++++++++++++++++")
        print("\n")
        output_state = await graph.ainvoke(input_state)
        input_state = output_state


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
    
