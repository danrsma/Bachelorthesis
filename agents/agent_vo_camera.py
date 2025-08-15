from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
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
import time



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
    visionloop: str
    iter: str
    filepath: str
    userinput: str

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

def combine_images_side_by_side(img1, img2):

    if img1.height != img2.height:
        new_height = max(img1.height, img2.height)
        img1 = img1.resize((int(img1.width * new_height / img1.height), new_height))
        img2 = img2.resize((int(img2.width * new_height / img2.height), new_height))

    total_width = img1.width + img2.width
    combined = Image.new("RGBA", (total_width, img1.height))
    
    combined.paste(img1, (0, 0))
    combined.paste(img2, (img1.width, 0))
    
    return combined


async def vision_llm_func(state: MyState) -> MyState:

    # Get Image Data
    file_path = state["filepath"]
    if file_path == "":
        # Create Vision Agent Chain
        vision_llm_chat = ChatOllama(
            model="llama4:latest",
            base_url="http://localhost:11434",
            temperature=0.5,
        )

        # Create Prompt
        prompt = """You are an expert in image analysis, 3D modeling, and Blender scripting. 
            Provide a detailed and extensive description of the scene and list all assets including hdri, models and textures you will need to create it."""+state["userinput"]
       
        # Get Agent Result
        vision_result = vision_llm_chat.invoke(prompt)

        # Ouput Image LLM
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
        model="llama4:latest",
        temperature=0.5,
    )

    prompt_func_runnable = RunnableLambda(prompt_func)
    chain = prompt_func_runnable | vision_llm_chat

    # Create Prompt
    prompt_vision = """You are an expert in image analysis, 3D modeling, and Blender scripting. 
        Provide a detailed and extensive description of the image and list all assets including hdri, models and textures you will need to create it."""

    # Get Agent Chain Result
    vision_result = chain.invoke({
        "text": prompt_vision,
        "image": image_b64,
    })

    print("\n")
    print("ImageLLM Output:")
    print("\n")
    print(vision_result.content)
    print("\n")

    state["vision"] = vision_result.content

    return state

async def vision_llm_func_feedback(state: MyState) -> MyState:

    # Get Image Data
    file_path = state["filepath"]
    
    file_path_1 = file_path+"_1.png"
    file_path_2 = file_path+"_2.png"
    file_path_3 = file_path+"_3.png"
    file_path_4 = file_path+"_4.png"
    
    try:

        pil_image_1 = Image.open(file_path_1)
        pil_image_2 = Image.open(file_path_2)
        pil_image_3 = Image.open(file_path_3)
        pil_image_4 = Image.open(file_path_4)

    except Exception as e:
        print(f"Error in main execution: {e}")


    image_b64_1 = convert_to_base64(pil_image_1)
    image_b64_2 = convert_to_base64(pil_image_2)
    image_b64_3 = convert_to_base64(pil_image_3)
    image_b64_4 = convert_to_base64(pil_image_4)

    images = [image_b64_1,image_b64_2,image_b64_3,image_b64_4]

    # Create Vision Agent Chain
    vision_llm_chat = ChatOllama(
        model="llama4:latest",
        temperature=0.5,
    )

    prompt_func_runnable = RunnableLambda(prompt_func)
    chain = prompt_func_runnable | vision_llm_chat

    # Create Prompt
    prompt_vision_loop = """You are an expert in image analysis, 3D modeling, and Blender scripting. 
            Provide a detailed comparison of the image and the description.
            Mark out how the image is different from the description.
            Describe the errors you see in the image.
            """+state["vision"]

    # Get Agent Chain Result
    vision_result1 = chain.invoke({
        "text": prompt_vision_loop,
        "image": image_b64_1,
    })
    # Get Agent Chain Result
    vision_result2 = chain.invoke({
        "text": prompt_vision_loop,
        "image": image_b64_2,
    })
    # Get Agent Chain Result
    vision_result3 = chain.invoke({
        "text": prompt_vision_loop,
        "image": image_b64_3,
    })
    # Get Agent Chain Result
    vision_result4 = chain.invoke({
        "text": prompt_vision_loop,
        "image": image_b64_4,
    })
    # Summarize Results
    vision_prompt_loop = f"""List all the different errors, assets and differences from the following texts.\n
    {vision_result1.content}\n{vision_result2.content}\n{vision_result3.content}\n{vision_result4.content}
    """
    vision_result = vision_llm_chat.invoke(vision_prompt_loop)

    print("\n")
    print("ImageLLM Output:")
    print("\n")
    print(vision_result.content)
    print("\n")

    state["visionloop"] = vision_result.content

    return state


async def tools_llm_func(state):

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
    filtered_tools = [t for t in tools if t.name not in {"get_hyper3d_status", "get_sketchfab_status", "search_sketchfab_models","download_sketchfab_models","generate_hyper3d_model_via_text","generate_hyper3d_model_via_images","poll_rodin_job_status","import_generated_asset"}]
    
    # Create Llm Chat
    tools_llm_chat = ChatOllama(
        model="gpt-oss:120b",
        temperature=0.0,
    )
    
    # Create Tool Agent
    agent = create_react_agent(
        model = tools_llm_chat,
        tools=filtered_tools
    )


    # Get Agent Result
    try:

        tool_result = await agent.ainvoke(
            {"messages": [HumanMessage(content="You are an expert in image analysis, 3D modeling, and Blender scripting."+
            "Recreate the provided Scene in Blender. Use Polyhaven assets and Blender Code Execution.\n"+state["vision"])]}
        )
        

    except Exception as e:
        print(f"Error in main execution: {e}")

    # Make Viewport Screenshots
    iter = state["iter"]
    screenshot_code = f"""import bpy
    import math
    import mathutils

    # Set Eevee Next as render engine
    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'

    # Remove existing cameras (optional)
    for obj in list(bpy.data.objects):
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj)

    # Camera parameters
    radius = 30      # distance from center
    height = 15      # Z height
    center = (0, 0, 5)
    angles_deg = [0, 90, 180, 270]  # positions around the object

    for i, angle_deg in enumerate(angles_deg, start=1):
        angle_rad = math.radians(angle_deg)
        
        # Camera position
        x = center[0] + radius * math.cos(angle_rad)
        y = center[1] + radius * math.sin(angle_rad)
        z = height
        
        # Create new camera
        cam_data = bpy.data.cameras.new(name="Camera_"+str(i))
        cam_object = bpy.data.objects.new("Camera_"+str(i), cam_data)
        cam_object.location = (x, y, z)
        
        # Rotate camera to face the object
        direction = mathutils.Vector(center) - cam_object.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        cam_object.rotation_euler = rot_quat.to_euler()
        
        # Link camera to scene
        bpy.context.collection.objects.link(cam_object)
        
        # Set as active camera and render
        bpy.context.scene.camera = cam_object
        bpy.context.scene.render.filepath = "C:/Users/cross/Desktop/Feedback_{iter}_"+str(i)+".png"
        bpy.ops.render.render(write_still=True)

    """
    try:

        tool_result = await agent.ainvoke(
            {"messages": [HumanMessage(content="Execute the following Blender Python Code:\n"+screenshot_code+
            "\nIf it does not work try to fix and reexecute it.")]}
        )
        print("\n")
        print("ToolLLM Output:")
        print("\n")
        print(screenshot_code)
        print("\n")
        print("Screenshot taken.")
        print("\n")

    except Exception as e:
        print(f"Error in main execution: {e}")

    return state

async def tools_llm_func_feedback(state):

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
    filtered_tools = [t for t in tools if t.name not in {"get_hyper3d_status", "get_sketchfab_status", "search_sketchfab_models","download_sketchfab_models","generate_hyper3d_model_via_text","generate_hyper3d_model_via_images","poll_rodin_job_status","import_generated_asset"}]
    
    # Create Llm Chat
    tools_llm_chat = ChatOllama(
        model="gpt-oss:120b",
        temperature=0.0,
    )
    
    #Create Tool Agent
    agent = create_react_agent(
        model = tools_llm_chat,
        tools=filtered_tools
    )


    # Save Blend File
    blendfile_code = """import bpy

    # Save to a specific file path
    bpy.ops.wm.save_as_mainfile(filepath="C:/Users/cross/Desktop/result.blend")

    """
    try:
        tools_result = await agent.ainvoke(
            {"messages": [HumanMessage(content="Execute the following Blender Python Code:\n"+blendfile_code+
            "\nIf it does not work try to fix and reexecute it.")]}
        )
        print("\n")
        print("ToolLLM Output:")
        print("\n")
        print(blendfile_code)
        print("\n")
        print("Blendfile saved.")
        print("\n")
    except Exception as e:
        print(f"Error in main execution: {e}")


    # Get Agent Result
    try:

        tool_result = await agent.ainvoke(
            {"messages": [HumanMessage(content="You are an expert in image analysis, 3D modeling, and Blender scripting."+
            " Improve the Scene in Blender to minimize the differences and errors.\n"+state["visionloop"]+
            "\n Stick to the description of the scene.\n"+state["vision"])]}
        )

    except Exception as e:
        print(f"Error in main execution: {e}")

    # Make Viewport Screenshots
    iter = state["iter"]
    screenshot_code = f"""import bpy
    import math
    import mathutils

    # Set Eevee Next as render engine
    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'

    # Remove existing cameras (optional)
    for obj in list(bpy.data.objects):
        if obj.type == 'CAMERA':
            bpy.data.objects.remove(obj)

    # Camera parameters
    radius = 30      # distance from center
    height = 15      # Z height
    center = (0, 0, 5)
    angles_deg = [0, 90, 180, 270]  # positions around the object

    for i, angle_deg in enumerate(angles_deg, start=1):
        angle_rad = math.radians(angle_deg)
        
        # Camera position
        x = center[0] + radius * math.cos(angle_rad)
        y = center[1] + radius * math.sin(angle_rad)
        z = height
        
        # Create new camera
        cam_data = bpy.data.cameras.new(name="Camera_"+str(i))
        cam_object = bpy.data.objects.new("Camera_"+str(i), cam_data)
        cam_object.location = (x, y, z)
        
        # Rotate camera to face the object
        direction = mathutils.Vector(center) - cam_object.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        cam_object.rotation_euler = rot_quat.to_euler()
        
        # Link camera to scene
        bpy.context.collection.objects.link(cam_object)
        
        # Set as active camera and render
        bpy.context.scene.camera = cam_object
        bpy.context.scene.render.filepath = "C:/Users/cross/Desktop/Feedback_{iter}_"+str(i)+".png"
        bpy.ops.render.render(write_still=True)
    
    """
    
    try:

        tool_result = await agent.ainvoke(
            {"messages": [HumanMessage(content="Execute the following Blender Python Code:\n"+screenshot_code+
            "\nIf it does not work try to fix and reexecute it.")]}
        )
        print(screenshot_code)
        print("\n")
        print("Screenshot taken.")
        print("\n")
    except Exception as e:
        print(f"Error in main execution: {e}")

    return state

async def branching_feedback(state):
    # Get Image Data
    file_path = state["filepath"]
    
    file_path_1 = file_path+"_1.png"
    file_path_2 = file_path+"_2.png"
    file_path_3 = file_path+"_3.png"
    file_path_4 = file_path+"_4.png"
    
    iter = state["iter"]
    file_path_old = f"C:/Users/cross/Desktop/Feedback_{str(iter-1)}"

    file_path_old_1 = file_path_old+"_1.png"
    file_path_old_2 = file_path_old+"_2.png"
    file_path_old_3 = file_path_old+"_3.png"
    file_path_old_4 = file_path_old+"_4.png"

    try:

        pil_image_1 = Image.open(file_path_1)
        pil_image_2 = Image.open(file_path_2)
        pil_image_3 = Image.open(file_path_3)
        pil_image_4 = Image.open(file_path_4)
        pil_image_1 = Image.open(file_path_old_1)
        pil_image_2 = Image.open(file_path_old_2)
        pil_image_3 = Image.open(file_path_old_3)
        pil_image_4 = Image.open(file_path_old_4)

    except Exception as e:
        print(f"Error in main execution: {e}")


    image_b64_1 = convert_to_base64(pil_image_1)
    image_b64_2 = convert_to_base64(pil_image_2)
    image_b64_3 = convert_to_base64(pil_image_3)
    image_b64_4 = convert_to_base64(pil_image_4)
    image_b64_old_1 = convert_to_base64(pil_image_1)
    image_b64_old_2 = convert_to_base64(pil_image_2)
    image_b64_old_3 = convert_to_base64(pil_image_3)
    image_b64_old_4 = convert_to_base64(pil_image_4)

    images = [image_b64_1,image_b64_2,image_b64_3,image_b64_4]
    images_old = [image_b64_old_1,image_b64_old_2,image_b64_old_3,image_b64_old_4]
    images_comb = []
    for i,j in zip(images,images_old):
        images_comb.append(combine_images_side_by_side(i,j))

    # Create Vision Agent Chain
    vision_llm_chat = ChatOllama(
        model="llama4:latest",
        temperature=0.5,
    )

    prompt_func_runnable = RunnableLambda(prompt_func)
    chain = prompt_func_runnable | vision_llm_chat

    # Create Prompt
    prompt_vision_loop = """You are an expert in image analysis and 3D modeling. 
            Wich side of the image better matches with the description left or right?
            Only Output Left or Right!!!
            """+state["vision"]

    # Get Agent Chain Result
    vision_result1 = chain.invoke({
        "text": prompt_vision_loop,
        "image": images_comb[0],
    })
    # Get Agent Chain Result
    vision_result2 = chain.invoke({
        "text": prompt_vision_loop,
        "image": images_comb[1],
    })
    # Get Agent Chain Result
    vision_result3 = chain.invoke({
        "text": prompt_vision_loop,
        "image": images_comb[2],
    })
    # Get Agent Chain Result
    vision_result4 = chain.invoke({
        "text": prompt_vision_loop,
        "image": images_comb[3],
    })


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
    filtered_tools = [t for t in tools if t.name not in {"get_hyper3d_status", "get_sketchfab_status", "search_sketchfab_models","download_sketchfab_models","generate_hyper3d_model_via_text","generate_hyper3d_model_via_images","poll_rodin_job_status","import_generated_asset"}]
    
    # Create Llm Chat
    tools_llm_chat = ChatOllama(
        model="gpt-oss:120b",
        temperature=0.0,
    )
    
    # Create Tool Agent
    agent = create_react_agent(
        model = tools_llm_chat,
        tools=filtered_tools
    )
    

    # Load Blend File
    blendfile_code = """import bpy

    # Open a specific file path
    bpy.ops.wm.open_mainfile(filepath="C:/Users/cross/Desktop/result.blend")

    """
    # Branch Results Prompt
    branch_prompt = f"""Is there more Left or Right in:\n
        {vision_result1.content}\n{vision_result2.content}\n{vision_result3.content}\n{vision_result4.content}?
        If there is more Right execute the following Blender Code: {blendfile_code}
        If there is more Left do nothing and end execution.
        """

    # Branch Result
    try:

        tool_result = await agent.ainvoke(
            {"messages": [HumanMessage(content=branch_prompt)]}
        )

    except Exception as e:
        print(f"Error in main execution: {e}")
        

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
    graph.add_node("tools_llm", tools_llm_func)
    graph.add_edge(START,"vision_llm")
    graph.add_edge("vision_llm", "tools_llm")
    graph.add_edge("tools_llm",END)
    graph = graph.compile()

    # Get StateGraph Output State
    input_state = MyState(userinput=user_input,filepath=file_path,iter="1")
    output_state = await graph.ainvoke(input_state, config={"recursion_limit": 150})

    # Create StateGraph With Nodes And Edges for Feedback Loop
    graph = StateGraph(MyState)
    graph.add_node("vision_llm", vision_llm_func_feedback)
    graph.add_node("tools_llm", tools_llm_func_feedback)
    graph.add_node("branching",branching_feedback)
    graph.add_edge(START,"vision_llm")
    graph.add_edge("vision_llm", "tools_llm")
    graph.add_edge("tools_llm","branching")
    graph.add_edge("branching",END)
    graph = graph.compile()

    # Prepare Rendering Loop
    time.sleep(30)
    file_path_loop = "C:/Users/cross/Desktop/Feedback_1"
    output_state["filepath"] = file_path_loop
    input_state = output_state

    
    # Start Feedback Loop
    for i in range(9):
        print("\n")
        print(f"++++++++++++++++++++++++++++++")
        print(f"+ Feedback Loop iteration: {i+2} +")
        print(f"++++++++++++++++++++++++++++++")
        print("\n")
        input_state["iter"]=str(i+2)
        output_state = await graph.ainvoke(input_state, config={"recursion_limit": 150})
        time.sleep(30)
        file_path_loop = f"C:/Users/cross/Desktop/Feedback_{str(i+2)}"
        output_state["filepath"] = file_path_loop
        input_state = output_state


if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
    
