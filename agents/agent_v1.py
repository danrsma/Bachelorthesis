from typing import Annotated, List
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
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
        self.title("Input Form with File Selection")
        self.geometry("400x250")
        self.resizable(False, False)

        container = tk.Frame(self, padx=20, pady=20)
        container.pack(fill="both", expand=True)

        input_frame = tk.Frame(container)
        input_frame.pack(fill="x", pady=(0, 15))

        tk.Label(input_frame, text="ImageLLM:", width=12, anchor="w").grid(row=0, column=0, sticky="w")
        self.entry1 = tk.Entry(input_frame)
        self.entry1.grid(row=0, column=1, sticky="ew", padx=5)

        tk.Label(input_frame, text="CodeLLM:", width=12, anchor="w").grid(row=1, column=0, sticky="w", pady=5)
        self.entry2 = tk.Entry(input_frame)
        self.entry2.grid(row=1, column=1, sticky="ew", padx=5)

        tk.Label(input_frame, text="MCP-LLM:", width=12, anchor="w").grid(row=2, column=0, sticky="w")
        self.entry3 = tk.Entry(input_frame)
        self.entry3.grid(row=2, column=1, sticky="ew", padx=5)

        input_frame.columnconfigure(1, weight=1)

        file_frame = tk.Frame(container)
        file_frame.pack(fill="x", pady=(0, 15))

        tk.Button(file_frame, text="Choose Picture", command=self.choose_file).pack(side="left")
        self.file_path = tk.StringVar()
        self.file_label = tk.Label(file_frame, textvariable=self.file_path, anchor="w")
        self.file_label.pack(side="left", padx=10, fill="x", expand=True)

        submit_btn = tk.Button(container, text="Submit", command=self.submit)
        submit_btn.pack(pady=10)

        # Initialize variables
        self.user_input1 = None
        self.user_input2 = None
        self.user_input3 = None
        self.selected_file_path = None

    def choose_file(self):
        path = filedialog.askopenfilename(title="Choose a Picture")
        if path:
            self.file_path.set(path)

    def submit(self):
        self.user_input1 = str(self.entry1.get())
        self.user_input2 = str(self.entry2.get())
        self.user_input3 = str(self.entry3.get())
        self.selected_file_path = str(self.file_path.get())

        # Close the window after submit
        self.destroy()

def convert_to_base64(pil_image):
    """
    Convert PIL images to Base64 encoded strings

    :param pil_image: PIL image
    :return: Re-sized Base64 string
    """

    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")  # You can change the format if needed
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def plt_img_base64(img_base64):
    """
    Disply base64 encoded string as image

    :param img_base64:  Base64 string
    """
    # Create an HTML img tag with the base64 string as the source
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'
    # Display the image by rendering the HTML
    display(HTML(image_html))

def prompt_func(data):
    text = data["text"]
    image = data["image"]


    image_part = {
        "type": "image_url",
        "image_url": f"data:image/jpeg;base64,{image}",
    }

    content_parts = []

    text_part = {"type": "text", "text": text}

    content_parts.append(image_part)
    content_parts.append(text_part)

    return [HumanMessage(content=content_parts)]


async def main():
    #root = Tk()  
    #root.withdraw() 
    #file_path = filedialog.askopenfilename(title="Choose a Picture")  
    app = InputApp()
    app.mainloop()
    print("Inputs collected:")
    print("ImageLLM =", app.user_input1)
    print("CodeLLm =", app.user_input2)
    print("MCP-LLM =", app.user_input3)
    print("selected_file_path =", app.selected_file_path)
    file_path = app.selected_file_path
    user_input1 = app.user_input1
    user_input2 = app.user_input2
    user_input3 = app.user_input3

    try:

        #file_path = "image.jpg"
        pil_image = Image.open(file_path)

    except Exception as e:
        print(f"Error in main execution: {e}")

    image_b64 = convert_to_base64(pil_image)
    plt_img_base64(image_b64)
    
    llm1 = ChatOllama(
        model="qwen2.5vl:7b",
        temperature=0,
        # other params...
    )   

    chain = prompt_func | llm1 | StrOutputParser()

    result1 = chain.invoke(
        {"text": str(user_input1)+"\n"+"Describe the image.", "image": image_b64}
    )
        
    print(result1)
    
    llm2 = ChatOllama(
        model="hf.co/mradermacher/BlenderLLM-GGUF:Q8_0",
        temperature=0,
    )
    result2 = llm2.invoke(
        str(result1)+" Create a Blender 3D Model for the image.\n"+str(user_input2)
    )
    print(result2.content)
    
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
        model="qwen3:8b",
        temperature=0,
    )#.bind_tools(tools)
    
    agent = create_react_agent(
        model = llm3,
        tools=tools
    )

    
    #tool_dict = {tool.name: tool for tool in tools}
    #execute_code = result2.content  # this should be the generated Blender code
    try:
        result3 = await agent.ainvoke(
            {"messages": [{"role": "user", "content":  "execute_blender_code\n"+result2.content+"\n"+str(user_input3)}]}
        )

        #response = await tool_dict["execute_blender_code"].ainvoke({"code": execute_code})
        #print(response)
    except Exception as e:        

        for x in result3:
            print (x)
            for y in result3[x]:
                print (y,':',result3[x][y])
        print(f"Error in main execution: {e}")



if __name__ == "__main__":
    # Run the example
    asyncio.run(main())

