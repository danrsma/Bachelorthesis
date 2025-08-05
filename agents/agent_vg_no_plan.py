from langchain_core.messages import HumanMessage
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
import os
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
    filepath_1: str
    filepath_2: str
    userinput: str
    vision: str


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


async def llm_func(state):
    user_input = state["userinput"]

    if user_input == "":
        file_path_2 = state["filepath_2"]

        if file_path_2 == "":
            # Get Image Data
            file_path_1 = state["filepath_1"]

            try:

                pil_image = Image.open(file_path_1)

            except Exception as e:
                print(f"Error in main execution: {e}")


            image_b64_1= convert_to_base64(pil_image)


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


            # Create LLM Agent
            if "GOOGLE_API_KEY" not in os.environ:
                os.environ["GOOGLE_API_KEY"] = API_KEY
            
            llm_chat = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0,
                max_tokens=10000,
                timeout=None,
                max_retries=2,
                # other params...
            )


            # Prepare Image Chain
            prompt_func_runnable = RunnableLambda(prompt_func)
            chain = prompt_func_runnable | llm_chat

            # Create Full Prompt
            full_prompt = f"""
                You are an expert in image analysis, 3D modeling, and Blender scripting. 
                Provide a detailed and extensive description of the image. List all assets like hdris, models and textures you need to create it.
            """

            # Get Agent Chain Result
            vision_result = chain.invoke({
                "text": full_prompt,
                "image": image_b64_1,
            })

            # Ouput Image LLM
            print("\n")
            print("ImageLLM Output:")
            print("\n")
            print(vision_result.content)
            print("\n")
            state["vision"] = str(vision_result.content)

            # Prepare React Agent
            agent = create_react_agent(
                model = llm_chat,
                tools=filtered_tools,
            )
            
            # Create Full Prompt
            full_prompt = f"""
                You are an expert in image analysis, 3D modeling, and Blender scripting.
                Recreate the Scene with colors or textures in Blender using Polyhaven Assets and Blender Code Execution.
            """

            # Get Agent Result
            try:
                print("Agent Output:\n")
                await agent.ainvoke({
                    "messages": [
                        {"role": "user", "content": full_prompt+(vision_result.content)}
                    ]
                })

            except Exception as e:
                    print(f"Error in main execution: {e}")

            

            # Make Viewport Screenshot
            screenshot_code = """
            import bpy

            # Create a new camera object
            cam_data = bpy.data.cameras.new(name="MyCamera")
            cam_object = bpy.data.objects.new("MyCamera", cam_data)

            # Set camera location and rotation
            cam_object.location = (30, 0, 15)
            cam_object.rotation_euler = (1.3, 0, 1.57)

            # Link the camera to the current scene
            bpy.context.collection.objects.link(cam_object)

            # Set the new camera as the active camera
            bpy.context.scene.camera = cam_object

            bpy.context.scene.render.filepath = "C:\\Users\\cross\\Desktop\\Image.png"
            bpy.ops.render.render(write_still=True)

                """
            try:
                await agent.ainvoke(
                    {"messages": [{"role": "user", "content": "Execute the following Blender Python Code:\n"+screenshot_code+
                    "\nIf it does not work try to create a camera and reexecute it."}]}
                )
                print("Screenshot taken.")
            except Exception as e:
                print(f"Error in main execution: {e}")


            return state
        
        else:
            # Get Image Data
            try:
                pil_image = Image.open(file_path_2)

            except Exception as e:
                print(f"Error in main execution: {e}")


            image_b64_2= convert_to_base64(pil_image)

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

            # Create Tool Agent
            if "GOOGLE_API_KEY" not in os.environ:
                os.environ["GOOGLE_API_KEY"] = API_KEY
            
            llm_chat = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0,
                max_tokens=10000,
                timeout=None,
                max_retries=2,
                # other params...
            )

            # Prepare Image Chain
            prompt_func_runnable = RunnableLambda(prompt_func)
            chain = prompt_func_runnable | llm_chat

            # Create Full Prompt
            full_prompt = f"""
                You are an expert in image analysis, 3D modeling, and Blender scripting. 
                Provide a detailed and extensive comparison of the image and the discription.
            """
            
            # Get Agent Chain Result
            vision_result = chain.invoke({
                "text": full_prompt+state["vision"],
                "image": image_b64_2,
            })

            # Ouput Image LLM
            print("\n")
            print("ImageLLM Output:")
            print("\n")
            print(vision_result.content)
            print("\n")

            # Prepare React Agent
            agent = create_react_agent(
                model = llm_chat,
                tools=filtered_tools,
            )

            # Create Full Prompt
            full_prompt = f"""
                You are an expert in image analysis, 3D modeling, and Blender scripting. 
                Improve the Scene in Blender to match the description. Try to minimize the differences.
            """
            
            # Get Agent Result
            try:
                print("Agent Output:\n")
                await agent.ainvoke({
                    "messages": [
                        {"role": "user", "content": full_prompt+(vision_result.content)}
                    ]
                })

            except Exception as e:
                print(f"Error in main execution: {e}")
            


            # Make Viewport Screenshot
            screenshot_code = """
            import bpy

            # Create a new camera object
            cam_data = bpy.data.cameras.new(name="MyCamera")
            cam_object = bpy.data.objects.new("MyCamera", cam_data)

            # Set camera location and rotation
            cam_object.location = (30, 0, 15)
            cam_object.rotation_euler = (1.3, 0, 1.57)

            # Link the camera to the current scene
            bpy.context.collection.objects.link(cam_object)

            # Set the new camera as the active camera
            bpy.context.scene.camera = cam_object

            bpy.context.scene.render.filepath = "C:\\Users\\cross\\Desktop\\Feedback.png"
            bpy.ops.render.render(write_still=True)

                """
            try:
                await agent.ainvoke(
                    {"messages": [{"role": "user", "content": "Execute the following Blender Python Code:\n"+screenshot_code+
                    "\nIf it does not work try to create a camera and reexecute it."}]}
                )
                print("Screenshot taken.")
            except Exception as e:
                print(f"Error in main execution: {e}")


            return state
    else:
        file_path_2 = state["filepath_2"]

        if file_path_2 == "":
            
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


            # Create LLM Agent
            if "GOOGLE_API_KEY" not in os.environ:
                os.environ["GOOGLE_API_KEY"] = API_KEY
            
            llm_chat = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0,
                max_tokens=10000,
                timeout=None,
                max_retries=2,
                # other params...
            )

            # Create Full Prompt
            full_prompt = f"""
                You are an expert in image analysis, 3D modeling, and Blender scripting. 
                Provide a detailed and enhanced description of the scene. List all assets like hdris, models and textures you need to create it.
            """

            # Get Agent Chain Result
            vision_result = llm_chat.invoke(full_prompt+user_input)

            # Ouput Image LLM
            print("\n")
            print("ImageLLM Output:")
            print("\n")
            print(vision_result.content)
            print("\n")
            state["vision"] = vision_result.content

            # Prepare React Agent
            agent = create_react_agent(
                model = llm_chat,
                tools=filtered_tools,
            )
            
            # Create Full Prompt
            full_prompt = f"""
                You are an expert in image analysis, 3D modeling, and Blender scripting.
                Recreate the Scene with colors or textures in Blender using Polyhaven Assets and Blender Code Execution.
            """

            # Get Agent Result
            try:
                print("Agent Output:\n")
                await agent.ainvoke({
                    "messages": [
                        {"role": "user", "content": state["vision"]+"\n"+full_prompt}
                    ]
                })

            except Exception as e:
                    print(f"Error in main execution: {e}")

            

            # Make Viewport Screenshot
            screenshot_code = """
            import bpy

            # Create a new camera object
            cam_data = bpy.data.cameras.new(name="MyCamera")
            cam_object = bpy.data.objects.new("MyCamera", cam_data)

            # Set camera location and rotation
            cam_object.location = (30, 0, 15)
            cam_object.rotation_euler = (1.3, 0, 1.57)

            # Link the camera to the current scene
            bpy.context.collection.objects.link(cam_object)

            # Set the new camera as the active camera
            bpy.context.scene.camera = cam_object

            bpy.context.scene.render.filepath = "C:\\Users\\cross\\Desktop\\Image.png"
            bpy.ops.render.render(write_still=True)

                """
            try:
                await agent.ainvoke(
                    {"messages": [{"role": "user", "content": "Execute the following Blender Python Code:\n"+screenshot_code+
                    "\nIf it does not work try to create a camera and reexecute it."}]}
                )
                print("Screenshot taken.")
            except Exception as e:
                print(f"Error in main execution: {e}")


            return state
        
        else:
            # Get Image Data
            try:
                pil_image = Image.open(file_path_2)

            except Exception as e:
                print(f"Error in main execution: {e}")


            image_b64_2= convert_to_base64(pil_image)

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

        
            # Create Tool Agent
            if "GOOGLE_API_KEY" not in os.environ:
                os.environ["GOOGLE_API_KEY"] = API_KEY
            
            llm_chat = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0,
                max_tokens=10000,
                timeout=None,
                max_retries=2,
                # other params...
            )

                    # Prepare Image Chain
            prompt_func_runnable = RunnableLambda(prompt_func)
            chain = prompt_func_runnable | llm_chat

            # Create Full Prompt
            full_prompt = f"""
                You are an expert in image analysis, 3D modeling, and Blender scripting. 
                Provide a detailed and extensive comparison of the image and the discription.
            """
            
            # Get Agent Chain Result
            vision_result = chain.invoke({
                "text": full_prompt+state["vision"],
                "image": image_b64_2,
            })

            # Ouput Image LLM
            print("\n")
            print("ImageLLM Output:")
            print("\n")
            print(vision_result.content)
            print("\n")

            # Prepare React Agent
            agent = create_react_agent(
                model = llm_chat,
                tools=filtered_tools,
            )

            # Create Full Prompt
            full_prompt = f"""
                You are an expert in image analysis, 3D modeling, and Blender scripting. 
                Improve the Scene in Blender to match the description. Try to minimize the differences.
            """
            
            # Get Agent Result
            try:
                print("Agent Output:\n")
                await agent.ainvoke({
                    "messages": [
                        {"role": "user", "content": full_prompt+(vision_result.content)}
                    ]
                })

            except Exception as e:
                print(f"Error in main execution: {e}")
            


            # Make Viewport Screenshot
            screenshot_code = """
            import bpy

            # Create a new camera object
            cam_data = bpy.data.cameras.new(name="MyCamera")
            cam_object = bpy.data.objects.new("MyCamera", cam_data)

            # Set camera location and rotation
            cam_object.location = (30, 0, 15)
            cam_object.rotation_euler = (1.3, 0, 1.57)

            # Link the camera to the current scene
            bpy.context.collection.objects.link(cam_object)

            # Set the new camera as the active camera
            bpy.context.scene.camera = cam_object

            bpy.context.scene.render.filepath = "C:\\Users\\cross\\Desktop\\Feedback.png"
            bpy.ops.render.render(write_still=True)

                """
            try:
                await agent.ainvoke(
                    {"messages": [{"role": "user", "content": "Execute the following Blender Python Code:\n"+screenshot_code+
                    "\nIf it does not work try to create a camera and reexecute it."}]}
                )
                print("Screenshot taken.")
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
    graph.add_node("gemini_llm",llm_func)
    graph.add_edge(START,"gemini_llm")
    graph.add_edge("gemini_llm",END)
    graph = graph.compile()

    # Get StateGraph Output State
    input_state = MyState(filepath_1=file_path,filepath_2="",userinput=user_input,vision="")
    output_state = await graph.ainvoke(input_state, config={"recursion_limit": 150})


    # Prepare Rendering Loop
    time.sleep(10)
    file_path_loop = "C:\\Users\\cross\\Desktop\\Image.png"
    output_state["filepath_2"] = file_path_loop
    input_state = output_state
    time.sleep(10)

    # Start Rendering Loop
    for i in range(4):
        print("\n")
        print(f"++++++++++++++++++++++++++++++")
        print(f"+ Feedback Loop iteration: {str(i+2)} +")
        print(f"++++++++++++++++++++++++++++++")
        print("\n")
        output_state = await graph.ainvoke(input_state, config={"recursion_limit": 150})
        print(output_state)
        time.sleep(10)
        file_path_loop = "C:\\Users\\cross\\Desktop\\Feedback.png"
        output_state["filepath_2"] = file_path_loop
        input_state = output_state
        time.sleep(10)


if __name__ == "__main__":
    API_KEY = ""
    # Run the example
    asyncio.run(main())
    
