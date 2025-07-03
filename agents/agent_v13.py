from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.prebuilt import create_react_agent
from PIL import Image
import base64
from io import BytesIO
import asyncio
import tkinter as tk
from tkinter import filedialog


class InputApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Picture or Prompt")
        self.geometry("400x250")
        self.resizable(False, False)

        container = tk.Frame(self, padx=20, pady=20)
        container.pack(fill="both", expand=True)

        input_frame = tk.Frame(container)
        input_frame.pack(fill="x", pady=(0, 15))

        #tk.Label(input_frame, text="ImageLLM:", width=12, anchor="w").grid(row=0, column=0, sticky="w")
        #self.entry1 = tk.Entry(input_frame)
        #self.entry1.grid(row=0, column=1, sticky="ew", padx=5)

        tk.Label(input_frame, text="Prompt:", width=12, anchor="w").grid(row=1, column=0, sticky="w", pady=5)
        self.entry2 = tk.Entry(input_frame)
        self.entry2.grid(row=1, column=1, sticky="ew", padx=5)

        #tk.Label(input_frame, text="MCP-LLM:", width=12, anchor="w").grid(row=2, column=0, sticky="w")
        #self.entry3 = tk.Entry(input_frame)
        #self.entry3.grid(row=2, column=1, sticky="ew", padx=5)

        input_frame.columnconfigure(1, weight=1)

        file_frame = tk.Frame(container)
        file_frame.pack(fill="x", pady=(0, 15))

        tk.Button(file_frame, text="Picture", command=self.choose_file).pack(side="left")
        self.file_path = tk.StringVar()
        self.file_label = tk.Label(file_frame, textvariable=self.file_path, anchor="w")
        self.file_label.pack(side="left", padx=10, fill="x", expand=True)

        submit_btn = tk.Button(container, text="Submit", command=self.submit)
        submit_btn.pack(pady=10)

        # Initialize variables
        #self.user_input1 = None
        self.user_input2 = None
        #self.user_input3 = None
        self.selected_file_path = None

    def choose_file(self):
        path = filedialog.askopenfilename(title="Choose a Picture")
        if path:
            self.file_path.set(path)

    def submit(self):
        #self.user_input1 = str(self.entry1.get())
        self.user_input2 = str(self.entry2.get())
        #self.user_input3 = str(self.entry3.get())
        self.selected_file_path = str(self.file_path.get())

        # Close the window after submit
        self.destroy()

'''
def get_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id is None:
        session_id = "default_session"    
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]
'''
    
def get_history(*args, **kwargs):
    # Try to extract session_id from kwargs if present
    session_id = kwargs.get("session_id") or (args[0] if args else None) or "default_session"
    print(f"get_history called with session_id={session_id}")
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

def convert_to_base64_png(pil_image):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """
    
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")  # You can change the format if needed
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

def convert_to_base64_jpeg(pil_image):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """
    
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")  # You can change the format if needed
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


async def main():
    app = InputApp()
    app.mainloop()
    print("Inputs collected:")
    #print("ImageLLM =", app.user_input1)
    print("llm_prompt =", app.user_input2)
    #print("MCP-LLM =", app.user_input3)
    print("selected_file_path =", app.selected_file_path)
    file_path = app.selected_file_path
    #user_input1 = app.user_input1
    user_input = app.user_input2
    #user_input3 = app.user_input3

    while file_path == "" and user_input == "":
        app = InputApp()
        app.mainloop()
        print("Inputs collected:")
        #print("ImageLLM =", app.user_input1)
        print("llm_prompt =", app.user_input2)
        #print("MCP-LLM =", app.user_input3)
        print("selected_file_path =", app.selected_file_path)
        file_path = app.selected_file_path
        #user_input1 = app.user_input1
        user_input = app.user_input2
        #user_input3 = app.user_input3


    vision_llm = "gemma3:4b"
    code_llm = "hf.co/mradermacher/BlenderLLM-GGUF:Q4_K_M"
    tools_llm = "qwen3:8b"
    
    SESSION_ID = "default_session"
    history = get_history(SESSION_ID)
    if file_path != "":

        prompt = ChatPromptTemplate.from_messages([
        ("system", "You can analyze images and text."),
        MessagesPlaceholder("history"),
        ("human", "Analyze the following image and text:"),
        ("human", "Image: {image}"),
        ("human", "Text: {text}"),
        ("human", "Please describe what you see and read."),
        ])

        vision_llm_chat = ChatOllama(
                model=vision_llm,
                temperature=0.5,
                # other params...
            )   

        image_text_chain = vision_llm_chat | prompt

        vision_memory = RunnableWithMessageHistory(
            image_text_chain,
            get_session_history=lambda _: history,
            input_messages_key="input",   
            history_messages_key="history", 
        )
        try:

            pil_image = Image.open(file_path)

        except Exception as e:
            print(f"Error in main execution: {e}")

        pil_image.show()
        
        image_type = str(file_path.split(".")[-1])
        
        if image_type == "png":
            image_b64= convert_to_base64_png(pil_image)
        else:
            image_b64= convert_to_base64_jpeg(pil_image)
        input_vision = {
            "image": image_b64,
            "text": "Provide a detailed and extensive description of the image. Describe every object in the picture accurately. Describe the shape of the lanscape elements."
        }
        input_vision = prompt.format_prompt(
            image=image_b64,
            text="Provide a detailed and extensive description of the image. Describe every object in the picture accurately. Describe the shape of the landscape elements."
        )

        try:
            result_vision = await vision_memory.invoke(
                input_vision,
                config={"configurable": {"session_id": SESSION_ID}}
            )
        except Exception as e:        
            print(f"Error in main execution: {e}") 
        
        print("\n")
        print("VisionLLM Output:")
        print("\n")
        print(result_vision)
        print("\n")
        
        code_llm_chat = ChatOllama(
            model=code_llm,
            temperature=0.5,
        )
        code_memory = RunnableWithMessageHistory(
            code_llm_chat,
            get_session_history=lambda _: history,
            input_messages_key="input",
            history_messages_key="history",
        )
        try:
            result_code = await code_memory(
                {"input": str(user_input)+str(result_vision)+
                """ Create Blender Code of the described Landscape. 
                Create every Object and Shape with math."""}, 
                config={"configurable": {"session_id": SESSION_ID}}
            )
        except Exception as e:        
            print(f"Error in main execution: {e}") 

        
        print("\n")
        print("CodeLLM Output:")
        print("\n")
        print(result_code)
        print("\n")
        
        client = MultiServerMCPClient(
        {
            "blender_mcp": {
                "command": "uvx",
                # Replace with absolute path to your math_server.py file
                "args": ["blender-mcp"],
                "transport": "stdio",
            }
        }
        )
        try:

            tools = await client.get_tools()

        except Exception as e:
            print(f"Error in main execution: {e}")
        
    
        tools_llm_chat = ChatOllama(
            model=tools_llm,
            temperature=0.5,
        )
        agent = create_react_agent(
            model = tools_llm_chat,
            tools=tools
        )
        tools_memory = RunnableWithMessageHistory(
            agent,
            get_session_history=lambda _: history,
            input_messages_key="input",
            history_messages_key="history",
        )
        try:
            result_tools = await tools_memory.ainvoke({
                "input": "execute_blender_code\n" + result_code +"\nIf it does not work try to fix and reexecute it."},
                config={"configurable": {"session_id": SESSION_ID}}
            )
        except Exception as e:        
            print(f"Error in main execution: {e}")           

        try:
            screenshot_code = """
                import bpy
                bpy.context.scene.render.filepath = "/home/daniel/Bachelorarbeit/agents/render.png"
                bpy.ops.render.render(write_still=True)
            """
            result_tools = await agent.ainvoke(
                {"messages": [{"role": "user", "content": "execute__blender_code\n"+screenshot_code+"\nIf it does not work try to fix and reexecute it."}]}
            )
        except Exception as e:        
            print(f"Error in main execution: {e}")

        loop_result_code = result_code
        #Rerendering Loop
        for i in range(5):
            print("\n")
            print(f"++++++++++++++++++++++++++++++++++++++")
            print(f"+ Rendering Loop iteration: {str(i)} +") 
            print(f"++++++++++++++++++++++++++++++++++++++")
            print("\n")
            history = get_history(SESSION_ID)
            for msg in history.messages:
                print(f"[{msg.type}] {msg.content}")
            print("\n")
            file_path_loop = "/home/daniel/Bachelorarbeit/agents/render.png"
            try:

                pil_image = Image.open(file_path_loop)

            except Exception as e:
                print(f"Error in main execution: {e}")

            pil_image.show()

            image_type = str(file_path.split(".")[-1])
        
            if image_type == "png":
                image_b64= convert_to_base64_png(pil_image)
            else:
                image_b64= convert_to_base64_jpeg(pil_image)
            
            loop_input_vision = {
                "image": image_b64,
                "text": "How does image compare to the discription:"+str(result_vision)+"? What are the differences?"
            }
            try:
                loop_result_vision = await vision_memory.invoke(
                    loop_input_vision,
                    config={"configurable": {"session_id": SESSION_ID}}
                )
            except Exception as e:        
                print(f"Error in main execution: {e}") 

            print("\n")
            print("VisionLLM Output:")
            print("\n")
            print(loop_result_vision)
            print("\n")
    
            try:
                loop_result_code = await code_memory(
                    {"input": str(result_code)+"""The new image is the result of the provided Blender Code. 
                    Improve the Blender Code to minimize the differences. 
                    Also look at the errors during the first execution and try to avoid them."""}, 
                    config={"configurable": {"session_id": SESSION_ID}}
                )
            except Exception as e:        
                print(f"Error in main execution: {e}") 

            print("\n")
            print("CodeLLM Output:")
            print("\n")
            print(loop_result_code.content)
            print("\n")
            
            try:
                loop_result_tools = await tools_memory.ainvoke({
                    "input": "execute_blender_code\n" +loop_result_code.content+"\nIf it does not work try to fix and reexecute it."},
                    config={"configurable": {"session_id": SESSION_ID}}
                )
            except Exception as e:        
                print(f"Error in main execution: {e}")           

            try:
                screenshot_code = """
                    import bpy
                    bpy.context.scene.render.filepath = "/home/daniel/Bachelorarbeit/agents/render.png"
                    bpy.ops.render.render(write_still=True)
                """
                loop_result_tools = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": "execute__blender_code\n"+screenshot_code+"\nIf it does not work try to fix and reexecute it."}]}
                )
            except Exception as e:        
                print(f"Error in main execution: {e}")
            
            result_code = loop_result_code

    else:
        pass 


session_store: dict[str, InMemoryChatMessageHistory] = {}
if __name__ == "__main__":
    # Run the example
    asyncio.run(main())