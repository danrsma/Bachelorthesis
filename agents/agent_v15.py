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


async def main():

    # Open Input Window
    app = InputApp()
    app.mainloop()

    # Output Collected Inputs In Terminal
    print("Inputs collected:")
    print("llm_prompt =", app.user_input)
    print("selected_file_path =", app.selected_file_path)
    file_path = app.selected_file_path
    #user_input = app.user_input

    # Loop Until At Least One Input Collected
    while file_path == "":
        app = InputApp()
        app.mainloop()
        print("Inputs collected:")
        print("llm_prompt =", app.user_input)
        print("selected_file_path =", app.selected_file_path)
        file_path = app.selected_file_path
        #user_input = app.user_input

    try:

        pil_image = Image.open(file_path)

    except Exception as e:
        print(f"Error in main execution: {e}")


    image_b64= convert_to_base64(pil_image)


    # Get MCP-Tools From Server
    client = MultiServerMCPClient(
        {
            "blender_mcp": {
                "command": "firejail",
                "args": ["uvx", "blender-mcp", "--private", "--net=none", "--caps.drop=all", "--seccomp", "--private-dev", "--hostname=sandbox"],
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
        model="llama4",
        temperature=0.9,
    )
    agent = create_react_agent(
        model = tools_llm_chat,
        tools=tools
    )

    # Create Full prompt
    full_prompt = f"""
        You are an expert in image analysis, 3D modeling, and Blender scripting.

        Step 1: Provide a detailed and extensive description of the image. Describe every object in the picture accurately. Describe the shape of the lanscape elements.
        Step 2: Create Blender Code of the described Landscape. Create every Object and Shape with math.
        Step 3: Use available tools to execute the code. If errors happen, fix and retry.
    """
    
    # Run Agent With Evrything
    try:
        result = await agent.ainvoke({
            "messages": [
                {"role": "system", "content": full_prompt},
                {"role": "user", "content": f"[IMAGE_BASE64:{image_b64}"}
            ]
        })
        print("Agent Output:\n")
        print(result)
    except Exception as e:
        print(f"Error in unified agent: {e}")


    # Make A Screenshot          
        make_screenshot = """  
        import bpy
        bpy.context.scene.render.filepath = "/home/student-rossmaier/Bachelorthesis/agents/render.png"
        bpy.ops.render.render(write_still=True)
        """

    # Run Agent For Screenshot
    try:
        result = await agent.ainvoke({
            "messages": [
                {"role": "system", "content": make_screenshot},
            ]
        })
        print("Agent Output:\n")
        print(result)
    except Exception as e:
        print(f"Error in unified agent: {e}")


    # Prepare Rendering Loop
    file_path = "/home/student-rossmaier/Bachelorthesis/agents/render.png"

    # Start Rendering Loop
    for i in range(4):
        print("\n")
        print(f"++++++++++++++++++++++++++++++++++++++")
        print(f"+ Rendering Loop iteration: {str(i+2)} +")
        print(f"++++++++++++++++++++++++++++++++++++++")
        print("\n")

        try:

            pil_image = Image.open(file_path)

        except Exception as e:
            print(f"Error in main execution: {e}")


        image_b64_2= convert_to_base64(pil_image)


        # Get MCP-Tools From Server
        client = MultiServerMCPClient(
            {
                "blender_mcp": {
                    "command": "firejail",
                    "args": ["uvx", "blender-mcp", "--private", "--net=none", "--caps.drop=all", "--seccomp", "--private-dev", "--hostname=sandbox"],
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
            model="llama4",
            temperature=0.9,
        )
        agent = create_react_agent(
            model = tools_llm_chat,
            tools=tools
        )

        # Create Full prompt
        full_prompt = f"""
            You are an expert in image analysis, 3D modeling, and Blender scripting.

            Step 1: How does first image compare to the the second one? What are the differences?
            Step 2: The second image is the result of the provided Blender Code. Improve the Blender Code to minimize the differences. Also look at the errors during the first execution and try to avoid them.
            Step 3: Use available tools to execute the code. If errors happen, fix and retry.
        """
        
        # Run Agent With Evrything
        try:
            result = await agent.ainvoke({
                "messages": [
                    {"role": "system", "content": full_prompt},
                    {"role": "user", "content": f"[IMAGE_BASE64:{image_b64}"},
                    {"role": "user", "content": f"[IMAGE_BASE64:{image_b64_2}"}
                ]
            })
            print("Agent Output:\n")
            print(result)
        except Exception as e:
            print(f"Error in unified agent: {e}")


        # Make A Screenshot          
            make_screenshot = """  
            import bpy
            bpy.context.scene.render.filepath = "/home/student-rossmaier/Bachelorthesis/agents/render.png"
            bpy.ops.render.render(write_still=True)
            """

        # Run Agent For Screenshot
        try:
            result = await agent.ainvoke({
                "messages": [
                    {"role": "system", "content": make_screenshot},
                ]
            })
            print("Agent Output:\n")
            print(result)
        except Exception as e:
            print(f"Error in unified agent: {e}")



if __name__ == "__main__":
    # Run the example
    asyncio.run(main())
    
