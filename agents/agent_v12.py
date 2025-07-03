from langchain_core.messages import HumanMessage
#from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.runnables import RunnableLambda
from langchain.memory import ConversationBufferMemory
from langgraph.prebuilt import create_react_agent
import base64
from io import BytesIO
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


def prompt_func(data):
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
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """
    
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
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
    
    
    if file_path != "":
    
    
        try:

            pil_image = Image.open(file_path)

        except Exception as e:
            print(f"Error in main execution: {e}")

        pil_image.show()
        
        

        image_b64= convert_to_base64(pil_image)

        vision_llm_chat = ChatOllama(
            model=vision_llm,
            temperature=0.5,
            # other params...
        )   

        prompt_func_runnable = RunnableLambda(prompt_func)
        chain = prompt_func_runnable | vision_llm_chat #| StrOutputParser()
        memory = ConversationBufferMemory(return_messages=True)
        vision_chain = VisionMemoryRunnable(chain, memory)
    

        vision_result = vision_chain.invoke({
            "text": "Provide a detailed and extensive description of the image. Describe every object in the picture accurately. Describe the shape of the lanscape elements.",
            "image": image_b64,
        })    

        print("\n")
        print("ImageLLM Output:")
        print("\n")
        print(vision_result)
        print("\n")
        code_llm_chat = ChatOllama(
            model=code_llm,
            temperature=0.5,
        )

        code_llm_chat_input = str(user_input)+"\n"+str(vision_result)+""" Create Blender Code of the described Landscape. 
            Create every Object and Shape with math. 
            """

        code_result = code_llm_chat.invoke(code_llm_chat_input)
        memory.chat_memory.add_user_message(code_llm_chat_input)
        memory.chat_memory.add_ai_message(code_result.content)

        print("\n")
        print("CodeLLM Output:")
        print("\n")
        print(code_result.content)
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
            temperature=0.5,
        )
            
    
        
        agent = create_react_agent(
            model = llm3,
            tools=tools
        )
    
        try:
            tool_result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": "execute__blender_code\n"+code_result.content+"\nIf it does not work try to fix and reexecute it."}]}
            )

        except Exception as e:        
            print(f"Error in main execution: {e}")
        try:
            screenshot_code = """
                import bpy
                bpy.context.scene.render.filepath = "/home/daniel/Bachelorarbeit/agents/render.png"
                bpy.ops.render.render(write_still=True)
            """
            tool_result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": "execute__blender_code\n"+screenshot_code+"\nIf it does not work try to fix and reexecute it."}]}
            )
        except Exception as e:        
            print(f"Error in main execution: {e}")

        
        result_code = code_result.content
        
        #Rerendering Loop
        for i in range(5):
            print("\n")
            print(f"++++++++++++++++++++++++++++++++++++++")
            print(f"+ Rendering Loop iteration: {str(i)} +") 
            print(f"++++++++++++++++++++++++++++++++++++++")
            print("\n")
            print(memory)
            print("\n")
            file_path_loop = "/home/daniel/Bachelorarbeit/agents/render.png"
            try:

                pil_image_loop = Image.open(file_path_loop)
                pil_image_loop.show()
                image_b64_loop= convert_to_base64(pil_image_loop)


                prompt_func_runnable = RunnableLambda(prompt_func)
                chain = prompt_func_runnable | vision_llm_chat #| StrOutputParser()
                vision_chain = VisionMemoryRunnable(chain, memory)

                vision_loop_result = vision_chain.invoke({
                    "text": "How does the new image compare to the inital scene?"+str(vision_result.content)+" What are the differences?",
                    "image": image_b64_loop
                }) 
            
                print("\n")
                print("ImageLLM Output:")
                print("\n")
                print(vision_loop_result)
                print("\n")

            except Exception as e:
                print(f"Error in main execution: {e}")


            code_llm_chat_input_loop = str(result_code)+"""The new image is the result of the provided Blender Code. 
                Improve the Blender Code to minimize the differences. 
                Also look at the errors during the first execution and try to avoid them.
                """
            code_loop_result = code_llm_chat.invoke(code_llm_chat_input)
            memory.chat_memory.add_user_message(code_llm_chat_input_loop)
            memory.chat_memory.add_ai_message(code_loop_result.content)

            print("\n")
            print("CodeLLM Output:")
            print("\n")
            print(code_loop_result.content)
            print("\n")

            #3. MCP Action
            try:
                tool_loop_result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": "execute__blender_code\n"+code_loop_result.content+"\nIf it does not work try to fix and reexecute it."}]}
                )

            except Exception as e:        
                print(f"Error in main execution: {e}")

            try:
                screenshot_code = """
                    import bpy
                    bpy.context.scene.render.filepath = "/home/daniel/Bachelorarbeit/agents/render.png"
                    bpy.ops.render.render(write_still=True)
                """
                tool_loop_result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": "execute__blender_code\n"+screenshot_code+"\nIf it does not work try to fix and reexecute it."}]}
                )
            except Exception as e:        
                print(f"Error in main execution: {e}")
            result_code = code_loop_result.content
            


    else:
        code_llm_chat = ChatOllama(
        model=code_llm,
        temperature=0.5,
        )
        memory = ConversationBufferMemory(return_messages=True)
        code_llm_chat_input = str(user_input)+""" Create Blender Code of the described Landscape. 
            Create every Object and Shape with math. 
            """

        code_result = code_llm_chat.invoke(code_llm_chat_input)
        memory.chat_memory.add_user_message(code_llm_chat_input)
        memory.chat_memory.add_ai_message(code_result.content)  

        print("\n")
        print("CodeLLM Output:")
        print("\n")
        print(code_result.content)
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
        )#.bind_tools(tools)
        memory = ConversationBufferMemory(return_messages=True)
            
    
        
        agent = create_react_agent(
            model = tools_llm_chat,
            tools=tools
        )

        #3. MCP Action
        try:
            tool_result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": "execute__blender_code\n"+code_result.content+"\nIf it does not work try to fix and reexecute it."}]}
            )
        except Exception as e:        
            print(f"Error in main execution: {e}")
        try:
            screenshot_code = """
                import bpy
                bpy.context.scene.render.filepath = "/home/daniel/Bachelorarbeit/agents/render.png"
                bpy.ops.render.render(write_still=True)
            """
            tool_result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": "execute__blender_code\n"+screenshot_code+"\nIf it does not work try to fix and reexecute it."}]}
            )
        except Exception as e:        
            print(f"Error in main execution: {e}")
    

        result_code = code_result.content
        #Rerendering Loop
        for i in range(5):
            print("\n")
            print(f"++++++++++++++++++++++++++++++++++++++")
            print(f"+ Rendering Loop iteration: {str(i)} +") 
            print(f"++++++++++++++++++++++++++++++++++++++")
            print("\n")
            file_path_loop = "/home/daniel/Bachelorarbeit/agents/render.png"
            
            try:
                pil_image_loop = Image.open(file_path_loop)
                pil_image_loop.show()
                image_b64_loop= convert_to_base64(pil_image_loop)

                vision_llm_chat = ChatOllama(
                    model=vision_llm,
                    temperature=0.5,
                    # other params...
                )
                prompt_func_runnable = RunnableLambda(prompt_func)
                chain = prompt_func_runnable | vision_llm_chat #| StrOutputParser()
                memory = ConversationBufferMemory(return_messages=True)
                vision_chain = VisionMemoryRunnable(chain, memory)
                

           
                vision_loop_result = vision_chain.invoke({
                    "text": "How does image compare to the the discription:"+str(user_input)+"? What are the differences?",
                    "image": image_b64_loop
                })
                print("\n")
                print("ImageLLM Output:")
                print("\n")
                print(str(vision_loop_result))
                print("\n")

            except Exception as e:
                print(f"Error in main execution: {e}")
            
            code_llm_chat_input_loop = str(result_code)+"""The new image is the result of the provided Blender Code. 
                Improve the Blender Code to minimize the differences. 
                Also look at the errors during the first execution and try to avoid them.
                """
            code_loop_result = code_llm_chat.invoke(code_llm_chat_input)
            memory.chat_memory.add_user_message(code_llm_chat_input_loop)
            memory.chat_memory.add_ai_message(code_loop_result.content)



            print("\n")
            print("CodeLLM Output:")
            print("\n")
            print(code_loop_result.content)
            print("\n")

            #3. MCP Action
            try:
                tool_loop_result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": "execute__blender_code\n"+code_loop_result.content+"\nIf it does not work try to fix and reexecute it."}]}
                )
            except Exception as e:        
                print(f"Error in main execution: {e}")
            try:
                screenshot_code = """
                    import bpy
                    bpy.context.scene.render.filepath = "/home/daniel/Bachelorarbeit/agents/render.png"
                    bpy.ops.render.render(write_still=True)
                """
                tool_loop_result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": "execute__blender_code\n"+screenshot_code+"\nIf it does not work try to fix and reexecute it."}]}
                )
            except Exception as e:        
                print(f"Error in main execution: {e}")

        

            result_code = code_loop_result.content
        

if __name__ == "__main__":
    # Run the example
    asyncio.run(main())

