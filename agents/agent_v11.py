from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor
from langgraph.prebuilt import create_react_agent
import base64
from io import BytesIO
from IPython.display import HTML, display
from PIL import Image
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


class VisionMemoryRunnable:
    def __init__(self, chain, memory):
        self.chain = chain
        self.memory = memory

    def invoke(self, inputs):
        history_text = "\n".join([
            f"{msg.type.capitalize()}: {msg.content}" for msg in self.memory.chat_memory.messages
        ])
        updated_text = history_text + f"\nHuman: {inputs['text']}"
        inputs["text"] = updated_text
        result = self.chain.invoke(inputs)
        self.memory.chat_memory.add_user_message(inputs["text"])
        self.memory.chat_memory.add_ai_message(result)
        return result


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


def plt_img_base64_png(img_base64):
    """
    Disply base64 encoded string as image

    :param img_base64:  Base64 string
    """
    # Create an HTML img tag with the base64 string as the source
    image_html = f'<img src="data:image/png;base64,{img_base64}" />'
    # Display the image by rendering the HTML
    #display(HTML(image_html))

def prompt_func_png(data):
    text = data["text"]
    image = data["image"]
    image_type = data["image_type"]

    image_part = {
        "type": "image_url",
        "image_url": f"data:image/png;base64,{image}",
    }

    content_parts = []

    text_part = {"type": "text", "text": text}

    content_parts.append(image_part)
    content_parts.append(text_part)

    return [HumanMessage(content=content_parts)]

def prompt_func_png_two_images(data):
    text = data["text"]
    image = data["image"]
    image_type = data["image_type"]
    image_loop = data["image_loop"]
    image_type_loop = data["image_type_loop"]

    image_part = {
        "type": "image_url",
        "image_url": f"data:image/png;base64,{image}",
    }
    image_part_loop = {
        "type": "image_url",
        "image_url": f"data:image/png;base64,{image_loop}",
    }

    content_parts = []

    text_part = {"type": "text", "text": text}

    content_parts.append(image_part)
    content_parts.append(image_part_loop)
    content_parts.append(text_part)

    return [HumanMessage(content=content_parts)]

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


def plt_img_base64_jpeg(img_base64):
    """
    Disply base64 encoded string as image

    :param img_base64:  Base64 string
    """
    # Create an HTML img tag with the base64 string as the source
    image_html = f'<img src="data:image/png;base64,{img_base64}" />'
    # Display the image by rendering the HTML
    #display(HTML(image_html))

def prompt_func_jpeg(data):
    text = data["text"]
    image = data["image"]
    image_type = data["image_type"]

    image_part = {
        "type": "image_url",
        "image_url": f"data:image/png;base64,{image}",
    }

    content_parts = []

    text_part = {"type": "text", "text": text}

    content_parts.append(image_part)
    content_parts.append(text_part)

    return [HumanMessage(content=content_parts)]



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
    
    
    if file_path != "":
    
    
        try:

            pil_image = Image.open(file_path)

        except Exception as e:
            print(f"Error in main execution: {e}")

        pil_image.show()
        
        image_type = str(file_path.split(".")[-1])
        
        
        if image_type == "png":
            image_b64= convert_to_base64_png(pil_image)
            plt_img_base64_png(image_b64)
            llm1 = ChatOllama(
                model=vision_llm,
                temperature=0.3,
                # other params...
            )   

            
            chain = prompt_func_png | llm1 | StrOutputParser()
            memory_1 = ConversationBufferMemory(return_messages=True)
            vision_chain = VisionMemoryRunnable(chain, memory_1)
        
        else:
            image_b64= convert_to_base64_jpeg(pil_image)
            plt_img_base64_jpeg(image_b64)
        
            llm1 = ChatOllama(
                model=vision_llm,
                temperature=0.3,
                # other params...
            )   

            
            chain = prompt_func_jpeg | llm1 | StrOutputParser()
            memory_1 = ConversationBufferMemory(return_messages=True)
            vision_chain = VisionMemoryRunnable(chain, memory_1)


        result1 = vision_chain.invoke({
            "text": "Provide a detailed and extensive description of the image. Describe every object in the picture accurately. Describe the shape of the lanscape elements.",
            "image": image_b64,
            "image_type": image_type
        })    

        print("\n")
        print("ImageLLM Output:")
        print("\n")
        print(result1)
        print("\n")
        llm2 = ChatOllama(
            model=code_llm,
            temperature=0.5,
        )
        memory_2 = ConversationBufferMemory(return_messages=True)

        llm2_input = str(user_input)+"\n"+str(result1)+""" Create Blender Code for the described Landscape. Create every Object and Shape with math. Put the following lines at the end:   
            bpy.context.scene.render.filepath = "/home/daniel/Bachelorarbeit/agents/render.png"
            bpy.ops.render.render(write_still=True)
            Do not use the following.
            bpy.ops.object.camera_add()
            camera = bpy.context.object
            camera.name = "Camera"
            bpy.context.scene.camera = camera
            """

        result2 = llm2.invoke(llm2_input)
        memory_2.chat_memory.add_user_message(llm2_input)
        memory_2.chat_memory.add_ai_message(result2.content)

        print("\n")
        print("CodeLLM Output:")
        print("\n")
        print(result2.content)
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
        
        
        llm3 = ChatOllama(
            model=tools_llm,
            temperature=0.1,
        )#.bind_tools(tools)
        memory_3 = ConversationBufferMemory(return_messages=True)
            
        filtered_tools = [
            t for t in tools
            if t.name not in {"get_hyper3d_status", "get_sketchfab_status", "search_sketchfab_models","download_sketchfab_models","generate_hyper3d_model_via_text","generate_hyper3d_model_via_images","poll_rodin_job_status","import_generated_asset"}
        ]
    
        
        agent = create_react_agent(
            model = llm3,
            tools=filtered_tools
        )
        agent_executor_3 = AgentExecutor(agent=agent, tools=filtered_tools, memory=memory_3)

        #3. MCP Action
        try:
            result3 = await agent_executor_3.ainvoke(
                {"messages": [{"role": "user", "content":  "execute_code\n"+result2.content+"\nIf it does not work try to fix and reexecute it."}]}
            )

        except Exception as e:        
            print(f"Error in main execution: {e}")

        
        
        result_code = result2.content
        
        #Rerendering Loop
        for i in range(10):
            file_path_loop = "/home/daniel/Bachelorarbeit/agents/render.png"
            try:

                pil_image_loop = Image.open(file_path_loop)

            except Exception as e:
                print(f"Error in main execution: {e}")

            pil_image_loop.show()
            image_b64_loop= convert_to_base64_png(pil_image_loop)
            plt_img_base64_png(image_b64_loop)

            
            chain = prompt_func_png_two_images | llm1 | StrOutputParser()
            vision_chain = VisionMemoryRunnable(chain, memory_1)

            result_loop1 = vision_chain.invoke({
                "text": "How does image_loop compare to image? What are the differences?",
                "image": image_b64, "image_type": image_type, "image_loop": image_b64_loop, "image_type_loop": "png"
            }) 
        
            print("\n")
            print("ImageLLM Output:")
            print("\n")
            print(result_loop1)
            print("\n")


            llm2_input = str(result_code)+"Image 2 is the result of the provided Blender Code.\n"+str(result_loop1)+"\nRewrite the Blender Code to make the differences smaller."
            result_loop2 = llm2.invoke(llm2_input)
            memory_2.chat_memory.add_user_message(llm2_input)
            memory_2.chat_memory.add_ai_message(result_loop2.content)

            print("\n")
            print("CodeLLM Output:")
            print("\n")
            print(result_loop2.content)
            print("\n")

            #3. MCP Action
            try:
                result_loop3 = await agent_executor_3.ainvoke(
                    {"messages": [{"role": "user", "content":  "execute_code\n"+str(result_loop2.content)+"\nIf it does not work try to fix and reexecute it."}]}
                )

            except Exception as e:        
                print(f"Error in main execution: {e}")

            result_code = result_loop2.content



    else:
        llm2 = ChatOllama(
        model=code_llm,
        temperature=0.5,
        )
        memory_2 = ConversationBufferMemory(return_messages=True)
        llm2_input = str(user_input)+""" Create Blender Code for the described Landscape. Create every Object and Shape with math. Put the following lines at the end:   
            bpy.context.scene.render.filepath = "/home/daniel/Bachelorarbeit/agents/render.png"
            bpy.ops.render.render(write_still=True)
            Do not use the following.
            bpy.ops.object.camera_add()
            camera = bpy.context.object
            camera.name = "Camera"
            bpy.context.scene.camera = camera
            """

        result2 = llm2.invoke(llm2_input)
        memory_2.chat_memory.add_user_message(llm2_input)
        memory_2.chat_memory.add_ai_message(result2.content)  

        print("\n")
        print("CodeLLM Output:")
        print("\n")
        print(result2.content)
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

        
        llm3 = ChatOllama(
            model=tools_llm,
            temperature=0.1,
        )#.bind_tools(tools)
        memory_3 = ConversationBufferMemory(return_messages=True)
            
        filtered_tools = [
            t for t in tools
            if t.name not in {"get_hyper3d_status", "get_sketchfab_status", "search_sketchfab_models","download_sketchfab_models","generate_hyper3d_model_via_text","generate_hyper3d_model_via_images","poll_rodin_job_status","import_generated_asset"}
        ]
    
        
        agent = create_react_agent(
            model = llm3,
            tools=filtered_tools
        )
        agent_executor_3 = AgentExecutor(agent=agent, tools=filtered_tools, memory=memory_3)

        #3. MCP Action
        try:
            result3 = await agent_executor_3.ainvoke(
                {"messages": [{"role": "user", "content":  "execute_code\n"+result2.content+"\nIf it does not work try to fix and reexecute it."}]}
            )

        except Exception as e:        
            print(f"Error in main execution: {e}")

        result_code = result2.content
        memory_1 = ConversationBufferMemory(return_messages=True)
        #Rerendering Loop
        for i in range(10):
            
            file_path_loop = "/home/daniel/Bachelorarbeit/agents/render.png"
            try:

                pil_image_loop = Image.open(file_path_loop)

            except Exception as e:
                print(f"Error in main execution: {e}")

            pil_image_loop.show()
            image_b64_loop= convert_to_base64_png(pil_image_loop)
            plt_img_base64_png(image_b64_loop)

            
            chain = prompt_func_png | llm2 | StrOutputParser()

            vision_chain = VisionMemoryRunnable(chain, memory_1)

            try:
                result_loop1 = await vision_chain.invoke({
                    "text": "How does image compare to the the discription:"+str(user_input)+"? What are the differences?",
                    "image": image_b64, "image_type": image_type, "image_loop": image_b64_loop, "image_type_loop": "png"
                }) 
            except Exception as e:
                print(f"Error in main execution: {e}")

            print("\n")
            print("ImageLLM Output:")
            print("\n")
            print(result_loop1)
            print("\n")

            llm2_input = str(result_code)+"\nImage 2 is the result of the provided Blender Code.\n"+str(result_loop1)+"\nRewrite the Blender Code to make the differences smaller."
            result_loop2 = llm2.invoke(llm2_input)
            memory_2.chat_memory.add_user_message(llm2_input)
            memory_2.chat_memory.add_ai_message(result_loop2.content)



            print("\n")
            print("CodeLLM Output:")
            print("\n")
            print(result_loop2)
            print("\n")

            #3. MCP Action
            try:
                result_loop3 = await agent_executor_3.ainvoke(
                    {"messages": [{"role": "user", "content":  "execute_code\n"+result_loop2+"\nIf it does not work try to fix and reexecute it."}]}
                )

            except Exception as e:        
                print(f"Error in main execution: {e}")

            result_code = result_loop2
        

if __name__ == "__main__":
    # Run the example
    asyncio.run(main())

