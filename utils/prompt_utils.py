import json
import base64
import os
from utils.utils import rescale_image_loader, encode_image



PROVIDER_SYSTEM_PROMPT = """You are trained to assist code agents writing HTML webpages from a sketch layout. Your job is to act as a human user interacting with a code agent and answer their questions regarding the sketches. To help you provide more accurate answers, we will give you a reference implementation of the intended webpage and a screenshot of the reference implementation, in addition to the sketch design itself.

You should do your best to accurately and concisely answer the agent's questions regarding the sketches. However, you should not say anything more than what the agent asks for. In addition, the agent is **NOT** supposed to know about the reference implementation. You should **NEVER** mention the reference implementation or its screenshot in your response, and you should **NEVER** give out the actual HTML code to the agent.

Limit your response to each agent question to one succinct sentence."""


PROVIDER_TEXT_MESSAGE = """You have access to two images. One is a sketch layout of a webpage drawn in the wireframing conventions, and the other one is a screenshot of a reference implementation. Please note that some images have already been replaced by placeholders (i.e., "rick.jpg") in the screenshot.

In addition, you also have access to the HTML implementation of the reference webpage:
```
{}
```

-------------------------------------------------
Now, please answer the agent's questions based on the information you have. The agent will ask questions about elements in the sketch, and your answers **MUST** be **strictly** based on the provided images and html code.

Remember, you must answer the questions accurately and succinctly. You should **NEVER** make things up or provide any information more than what the agent asks for. The agent is not supposed to know about the reference implementation or its screenshot, so you should **NEVER** mention the reference implementation or the screenshot in your response, nor should you ever give out any HTML content to the agent. For example, if the agent asks for an element, you should answer with what is visible on the rendered webpage instead of the actual HTML tag or id. If the user asks for the color of something, you should describe the color in natural language (e.g., blue) instead of the hexadecimal color code. And if the user asks for the specific texts within a text block or paragraph, you should respond with a concise summary of the paragraph instead of reciting the text verbatim. You may acknowledge the fact that `rick.jpg` is used as image placeholders.

Format your answer to each question as a single sentence without omitting important information.

Agent Question: {}"""



def get_provider_system_message():
    return PROVIDER_SYSTEM_PROMPT

def get_provider_user_message(screenshot, sketch, html, question):
    if os.path.isfile(screenshot):
        screenshot = rescale_image_loader(screenshot)
        
    if os.path.isfile(sketch):
        sketch = rescale_image_loader(sketch)
    
    if sketch is None or screenshot is None:
      return None
    
    text_message = PROVIDER_TEXT_MESSAGE.format(html, question)
    
    return [
        {
            "type": "text",
            "text": text_message,
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{sketch}"
          }
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{screenshot}"
          }
        }
    ]
    
    


AGENT_SYSTEM_PROMPT = '''You are an expert web developer who specializes in HTML and CSS. A user will provide you with a sketch design of the webpage drawn in the wireframing conventions, where images are replaced by boxes with an "X" inside and texts are represented by curly lines. You need to return a single html file that uses HTML and CSS to produce a webpage that strictly follows the sketch design. Include all CSS code in the HTML file itself. If it involves any images, use "rick.jpg" as the placeholder. Some texts are replaced by curly lines as placeholders. You should try your best to infer what these texts should be, but do not hallucinate if you are not sure.

If you are unsure what certain elements are in the provided sketch, you should ask the user to clarify. Once you are confident, output a single HTML file with embedded CSS. Do not hallucinate any dependencies to external files. Pay attention to things like size and position of all the elements, as well as the overall layout.

If you want to ask a clarification question, format your question as: 
Question: """{{YOUR_QUESTION_HERE}}"""

To ask multiple questions in a single turn, you should list your questions as:
Question: """
1. {{First_Question}}
2. {{Second_Question}}
3. {{Third_Question}}
...
"""

If you are ready to write the final HTML code, format your code as
```
{{HTML_CSS_CODE}}
```'''

AGENT_USER_PROMPT = '''Here is a sketch design of a webpage about {topic}. Could you write a HTML+CSS code of this webpage for me?

Remember, If you are uncertain about something, please ask clarification questions.

If you want to ask a clarification question, format your question as: 
Question: """{{YOUR_QUESTION_HERE}}"""

To ask multiple questions in a single turn, you should list your questions as:
Question: """
1. {{First_Question}}
2. {{Second_Question}}
3. {{Third_Question}}
...
"""

If you are ready to write the final HTML code, format your code as
```
{{HTML_CSS_CODE}}
```'''

AGENT_USER_PROMPT_WITHOUT_TOPIC = '''Here is a sketch design of a webpage. Could you write a HTML+CSS code of this webpage for me?

Remember, If you are uncertain about something, please ask clarification questions.

If you want to ask a clarification question, format your question as: 
Question: """{{YOUR_QUESTION_HERE}}"""

To ask multiple questions in a single turn, you should list your questions as:
Question: """
1. {{First_Question}}
2. {{Second_Question}}
3. {{Third_Question}}
...
"""

If you are ready to write the final HTML code, format your code as
```
{{HTML_CSS_CODE}}
```'''


def get_agent_system_message():
    return AGENT_SYSTEM_PROMPT


def get_agent_user_message(sketch_path, topic):
    sketch = encode_image(sketch_path)
    if topic:
      return [
          {
              "type": "text",
              "text": AGENT_USER_PROMPT.format(topic=topic),
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{sketch}"
            }
          },
      ]
    else:
      return [
          {
              "type": "text",
              "text": AGENT_USER_PROMPT_WITHOUT_TOPIC,
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{sketch}"
            }
          },
      ]



DIRECT_SYSTEM_POMPT = '''You are an expert web developer who specializes in HTML and CSS. A user will provide you with a sketch design of the webpage following the wireframing conventions, where images are represented as boxes with an "X" inside, and texts are replaced with curly lines. You need to return a single html file that uses HTML and CSS to produce a webpage that strictly follows the sketch layout. Include all CSS code in the HTML file itself. If it involves any images, use "rick.jpg" as the placeholder name. You should try your best to figure out what text should be placed in each text block. In you are unsure, you may use "lorem ipsum..." as the placeholder text. However, you must make sure that the positions and sizes of these placeholder text blocks matches those on the provided sketch.

Do your best to reason out what each element in the sketch represents and write a HTML file with embedded CSS that implements the design. Do not hallucinate any dependencies to external files. Pay attention to things like size and position of all the elements, as well as the overall layout. You may assume that the page is static and ignore any user interactivity.'''


DIRECT_USER_PROMPT = '''Here is a sketch design of a webpage about {topic}. Could you write a HTML+CSS code of this webpage for me?

Please format your code as
```
{{HTML_CSS_CODE}}
```
Remember to use "rick.jpg" as the placeholder for any images'''

DIRECT_USER_PROMPT_WITHOUT_TOPIC = '''Here is a sketch design of a webpage. Could you write a HTML+CSS code of this webpage for me?

Please format your code as
```
{{HTML_CSS_CODE}}
```
Remember to use "rick.jpg" for as the placeholder any images'''


DIRECT_PROMPT_COMBINED_WITHOUT_TOPIC = '''You are an expert web developer who specializes in HTML and CSS. A user will provide you with a sketch design of the webpage following the wireframing conventions, where images are represented as boxes with an "X" inside, and texts are replaced with curly lines. You need to return a single html file that uses HTML and CSS to produce a webpage that strictly follows the sketch layout. Include all CSS code in the HTML file itself. If it involves any images, use "rick.jpg" as the placeholder name. You should try your best to figure out what text should be placed in each text block. In you are unsure, you may use "lorem ipsum..." as the placeholder text. However, you must make sure that the positions and sizes of these placeholder text blocks matches those on the provided sketch.

Do your best to reason out what each element in the sketch represents and write a HTML file with embedded CSS that implements the design. Do not hallucinate any dependencies to external files. Pay attention to things like size and position of all the elements, as well as the overall layout. You may assume that the page is static and ignore any user interactivity.

Here is a sketch design of a webpage. Could you write a HTML+CSS code of this webpage for me?

Please format your code as
```
{{HTML_CSS_CODE}}
```
Remember to use "rick.jpg" for as the placeholder any images'''


DIRECT_PROMPT_COMBINED = '''You are an expert web developer who specializes in HTML and CSS. A user will provide you with a sketch design of the webpage following the wireframing conventions, where images are represented as boxes with an "X" inside, and texts are replaced with curly lines. You need to return a single html file that uses HTML and CSS to produce a webpage that strictly follows the sketch layout. Include all CSS code in the HTML file itself. If it involves any images, use "rick.jpg" as the placeholder name. You should try your best to figure out what text should be placed in each text block. In you are unsure, you may use "lorem ipsum..." as the placeholder text. However, you must make sure that the positions and sizes of these placeholder text blocks matches those on the provided sketch.

Do your best to reason out what each element in the sketch represents and write a HTML file with embedded CSS that implements the design. Do not hallucinate any dependencies to external files. Pay attention to things like size and position of all the elements, as well as the overall layout. You may assume that the page is static and ignore any user interactivity.

Here is a sketch design of a webpage about {topic}. Could you write a HTML+CSS code of this webpage for me?

Please format your code as:
```
{{HTML_CSS_CODE}}
```
Remember to use "rick.jpg" for as the placeholder any images'''

def get_direct_prompt_combined(topic):
  if topic:
    return DIRECT_PROMPT_COMBINED.format(topic=topic)
  return DIRECT_PROMPT_COMBINED_WITHOUT_TOPIC

def get_direct_user_prompt(topic):
  if topic:
    return DIRECT_USER_PROMPT.format(topic=topic)
  return DIRECT_USER_PROMPT_WITHOUT_TOPIC


def get_direct_system_message():
  return DIRECT_SYSTEM_POMPT


def get_direct_user_message(sketch, topic):
    if os.path.isfile(sketch):
          sketch = rescale_image_loader(sketch)
    if topic:
      return [
          {
              "type": "text",
              "text": DIRECT_USER_PROMPT.format(topic=topic),
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{sketch}"
            }
          },
      ]
    else:
      return [
          {
              "type": "text",
              "text": DIRECT_USER_PROMPT_WITHOUT_TOPIC,
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{sketch}"
            }
          },
      ]
      
      

DESIGN_TO_CODE_USER_PROMPT = """You are an expert web developer who specializes in HTML and CSS.
A user will provide you with a screenshot of a webpage.
You need to return a single html file that uses HTML and CSS to reproduce the given website.
Include all CSS code in the HTML file itself.
If it involves any images, use "rick.jpg" as the placeholder.
Some images on the webpage are replaced with a blue rectangle as the placeholder, use "rick.jpg" for those as well.
Do not hallucinate any dependencies to external files. You do not need to include JavaScript scripts for dynamic interactions.
Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout.
Respond with the content of the HTML+CSS file.

Format your HTML+CSS output as:
```
{{HTML_CSS_CODE}}
```"""

def get_design_to_code_user_prompt():
    return DESIGN_TO_CODE_USER_PROMPT


DESIGN_TO_CODE_USER_PROMPT_2 = """You are an expert web developer who specializes in HTML and CSS.
A user will provide you with a screenshot of a webpage, along with additional layout information about the webpage:

{layout}

You need to return a single html file that uses HTML and CSS to reproduce the given website.
Include all CSS code in the HTML file itself.
If it involves any images, use "rick.jpg" as the placeholder.
Some images on the webpage are replaced with a blue rectangle as the placeholder, use "rick.jpg" for those as well.
Do not hallucinate any dependencies to external files. You do not need to include JavaScript scripts for dynamic interactions.
Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout.
Respond with the content of the HTML+CSS file.

Format your HTML+CSS output as:
```
{{HTML_CSS_CODE}}
```"""

def get_design_to_code_user_prompt2(layout):
    return DESIGN_TO_CODE_USER_PROMPT_2.format(layout=layout)
  

DESIGN_TO_CODE_USER_PROMPT_3 = """You are an expert web developer who specializes in HTML and CSS.
A user will provide you with a layout information about the webpage:

{layout}

And here are the text blocks (idx -> text) that should be inlcuded in the webpage:

{text}

You need to return a single html file that uses HTML and CSS to reproduce the given website.
Include all CSS code in the HTML file itself.
If it involves any images, use "rick.jpg" as the placeholder.
Some images on the webpage are replaced with a blue rectangle as the placeholder, use "rick.jpg" for those as well.
Do not hallucinate any dependencies to external files. You do not need to include JavaScript scripts for dynamic interactions.
Pay attention to things like size, text, position, and color of all the elements, as well as the overall layout.
Respond with the content of the HTML+CSS file.

Format your HTML+CSS output as:
```
{{HTML_CSS_CODE}}
```"""


def get_design_to_code_user_prompt3(layout, text):
    return DESIGN_TO_CODE_USER_PROMPT_3.format(layout=layout, text=text)




DIRECT_TEXT_AUGMENTED_USER_PROMPT = '''Here is a sketch design of a webpage drawn in the wireframing conventions. In addition, here is a list of text blocks that I would like to include in the webpage:

{texts}

Could you write a HTML+CSS code of this webpage for me?

Please format your code as
```
{{HTML_CSS_CODE}}
```
Remember to use "rick.jpg" as the placeholder for any images'''


def get_direct_text_augmented_user_message(sketch, texts):
    if os.path.isfile(sketch):
        sketch = encode_image(sketch)
    
    return [
        {
            "type": "text",
            "text": DIRECT_TEXT_AUGMENTED_USER_PROMPT.format(texts=texts),
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{sketch}"
          }
        },
    ]

def get_direct_text_augmented_user_prompt(texts):
  return DIRECT_TEXT_AUGMENTED_USER_PROMPT.format(texts=texts)


AGENT_TEXT_AUGMENTED_USER_PROMPT = '''Here is a sketch design of a webpage drawn in the wireframing conventions. In addition, here is a list of text blocks that I would like to include in the webpage:

{texts}

Could you write a HTML+CSS code of this webpage for me?

Remember, If you are uncertain about something, please ask clarification questions. Your questions should be thoughtful and specific, and you should ask no more than five questions in each turn.

If you want to ask a clarification question, format your question as: 
Question: """{{YOUR_QUESTION_HERE}}"""

To ask multiple questions in a single turn, you should list your questions as:
Question: """
1. {{First_Question}}
2. {{Second_Question}}
3. {{Third_Question}}
...
"""

If you are ready to write the final HTML code, format your code as
```
{{HTML_CSS_CODE}}
```
Remember to use "rick.jpg" as the placeholder for any images'''

def get_agent_text_augmented_user_message(sketch, texts):
    if os.path.isfile(sketch):
        sketch = encode_image(sketch)
    
    return [
        {
            "type": "text",
            "text": AGENT_TEXT_AUGMENTED_USER_PROMPT.format(texts=texts),
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{sketch}"
          }
        },
    ]

def get_guided_agent_text_augmented_user_message(sketch, texts, guide):
    if os.path.isfile(sketch):
        sketch = encode_image(sketch)
    
    agent_prompt = f"""{AGENT_TEXT_AUGMENTED_USER_PROMPT.format(texts=texts)}
    
    {guide}"""
    
    return [
        {
            "type": "text",
            "text": agent_prompt,
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{sketch}"
          }
        },
    ]


def get_text_augmented_user_prompt(texts):
  return AGENT_TEXT_AUGMENTED_USER_PROMPT.format(texts=texts)

TEXT_AUGMENTED_USER_PROMPT_WITH_QA_PAIRS = '''Here is a sketch design of a webpage drawn in the wireframing conventions. Also, here is a list of text blocks that I would like to include in the webpage:

{texts}

Could you write a HTML+CSS code of this webpage for me?

Here are some additional information for your reference:
{qa_pairs}

Please format your code as
```
{{HTML_CSS_CODE}}
```
Remember to use "rick.jpg" as the placeholder for any images'''

def get_user_message_with_qa_pairs(sketch, texts, qa_pairs):
    if os.path.isfile(sketch):
        sketch = encode_image(sketch)
        
    return [
        {
            "type": "text",
            "text": TEXT_AUGMENTED_USER_PROMPT_WITH_QA_PAIRS.format(texts=texts, qa_pairs=qa_pairs),
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{sketch}"
          }
        },
    ]

def get_user_prompt_with_qa_pairs(texts, qa_pairs):
    return TEXT_AUGMENTED_USER_PROMPT_WITH_QA_PAIRS.format(texts=texts, qa_pairs=qa_pairs)


REFLECT_USER_PROMPT = '''Suppose you are a frontend designer working with a code agent to implement an HTML webpage. You are provided with two images: the first image is the webpage you are hoping to produce, and the second one is the current implementation from the code agent. Note that images have already been replaced with blue rectangles as the placeholder.

Your job is to carefully compare the code agent's implementation against the intended webpage, and provide feedback to help the code agent make its implementation closer to the indended webpage. Your feedback should be specific to the differences in layouts and visual components on the two webpages. Please note that the code agent **DOES NOT** have access to the intended webpage, so you make sure to describe the intended visual components and where exactly the agent got wrong, instead of saying something like "refer to the format of the intended webpage". You should prioritize making sure that the code agent understands the correct layout before giving out any styling advice.

Limit your feedback to a single sentence.

You may compare and analyze the two webpages step by step. Once you are ready, your final feedback using triple quotes:
Feedback: """
{{YOUR_INSTRUCTIONS_HERE}}
"""

If you think the current implementation is close enough to the intended webpage, please output "Generation Complete" as your feedback. I.e.,
Feedback: """
Generation Complete
"""'''


def get_reflection_message(original, current_implementation):
    if os.path.isfile(original):
        sketch = rescale_image_loader(original)
    
    if os.path.isfile(original):
        sketch = rescale_image_loader(current_implementation)
    
    return [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Intended webpage:",
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{original}"
            }
          },
          {
            "type": "text",
            "text": "Code agent's current implementation:",
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{current_implementation}"
            }
          },
          {
            "type": "text",
            "text": REFLECT_USER_PROMPT,
          }
        ]
      }
    ]


def get_guided_reflection_message(original, current_implementation, guide):
    if os.path.isfile(original):
        sketch = rescale_image_loader(original)
    
    if os.path.isfile(original):
        sketch = rescale_image_loader(current_implementation)
    
    guided_reflection_prompt = f"""{REFLECT_USER_PROMPT}

{guide}"""
    
    return [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": "Intended webpage:",
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{original}"
            }
          },
          {
            "type": "text",
            "text": "Code agent's current implementation:",
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{current_implementation}"
            }
          },
          {
            "type": "text",
            "text": guided_reflection_prompt,
          }
        ]
      }
    ]



LAYOUT_SCHEMA = '''{
  "type": "object",
  "properties": {
    "elementType": {
      "type": "string",
      "enum": ["video", "image", "text_block", "form_table", "button", "nav_bar", "header", "footer", "main_body", "section"],
      "description": "Type of the element"
    },
    "display": {
      "type": "string",
      "description": "CSS display type of the element, such as block, inline, flex, and grid"
    },
    "height": {
      "type": "number",
      "description": "the relative height of the element, with respect to its parent"
    },
    "width": {
      "type": "number",
      "description": "the relative width of the element, with respect to its parent"
    },
    "layout": {
      "type": "object",
      "properties": {
        "orientation": {
          "type": "string",
          "enum": ["horizontal", "vertical", "both"],
          "description": "Orientation of the child elements; horizontal for rows and vertical for columns. For grid, this represents the primary content flow direction."
        },
        "distribution": {
          "type": "string",
          "enum": ["evenly spaced", "start-aligned", "end-aligned", "space-around", "space-between"],
          "description": "Optional, describes how elements are spaced within their container, applicable for flex and grid layouts."
        }
      },
      "required": ["orientation"]
    },
    "text": {
      "type": ["string", "null"]
      "description": "The textual content within the element if elementType is text block, otherwise null",
    },
    "style": {
      "type": ["string", "null"],
      "description": "Any additional information on the styling of the element, null if not specified",
    },
    "children": {
      "type": "array",
      "items": {
        "$ref": "#"
      },
      "description": "An array of child elements, each with the same structure as this object, use an empty list if there are no child elements."
    },
  },
  "required": ["elementType", "display", "height", "width", "layout", "children"]
}'''

VISUAL_COMPONENTS = '''{
    "video": "video players and video containers",
    "image": "image elements",
    "text_block": "any components including inner text",
    "form_table": "this includes all forms, tables, search bars, and text input areas",
    "button": "buttons",
    "nav_bar": "navigation bars and menus",
    "header": "the headder section of the webpage",
    "footer": "the footer section of the webpage",
    "main_body": "the main body of the webpage",
    "section": "any layout section in the main body",
}'''


GROUNDED_USER_PROMPT = '''Here is a sketch design of a webpage drawn in the wireframing conventions. Also, here is a list of text blocks that the user would like to include in the webpage:
"""
{texts}
"""

First, carefully analyze the given sketch and record the layout of its visual components in a JSON object. In particular, you should pay attention to the following components:
"""
{components}
"""

Here is the JSON Schema that you should follow when generating the layout structure of the webpage:
"""
{schema}
"""

Please break down the layout of the given sketch and output a JSON layout formatted as:
Page Layout: ```
{{YOUR_JSON_LAYOUT}}
```'''


GROUNDED_SCREENSHOT_PROMPT = '''You are an expert web developer who specializes in HTML and CSS.
A user will provide you with a screenshot of a webpage. You need to break down the structure of the webpage and output a layout tree of all visual components as an JSON object.
In particular, you shou pay attention to the following components:
"""
{components}
"""

Here is the JSON Schema that you should follow when generating the layout structure of the webpage:
"""
{schema}
"""

Please break down the layout of the given screenshot and output a JSON layout formatted as:
Page Layout: ```
{{YOUR_JSON_LAYOUT}}
```'''


HTML_GENERATION_PROMPT = '''Now, generate an HTML impelementation (with embedded CSS) of the webpage based on both the provided sketch and your JSON layout. Format your HTML+CSS code as:
```
{{HTML_CSS_CODE}}
```'''


def get_grounded_user_message(sketch, texts):
    if os.path.isfile(sketch):
        sketch = rescale_image_loader(sketch)
    
    return [
        {
            "type": "text",
            "text": GROUNDED_USER_PROMPT.format(texts=texts, components=VISUAL_COMPONENTS, schema=LAYOUT_SCHEMA),
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{sketch}"
          }
        },
    ]

def get_grounded_user_prompt(texts):
  return GROUNDED_USER_PROMPT.format(texts=texts, components=VISUAL_COMPONENTS, schema=LAYOUT_SCHEMA)

def get_html_generation_prompt():
  return HTML_GENERATION_PROMPT

def get_grounded_screenshot_message(screenshot):
  if os.path.isfile(screenshot):
    screenshot = rescale_image_loader(screenshot)
    
  return [
    {
      "type": "text",
      "text": GROUNDED_SCREENSHOT_PROMPT.format(components=VISUAL_COMPONENTS, schema=LAYOUT_SCHEMA),
    },
    {
      "type": "image_url",
      "image_url": {
        "url": f"data:image/jpeg;base64,{screenshot}"
      }
    },
  ]

GROUNDED_FEEDBACK_MESSAGE = '''Here are some user feedback regarding your current implementation:
{feedback}

Could you revise your JSON layout representation based on this feedback?

Output the updated JSON layout formatted in triple quotes:
Page Layout: ```
{{YOUR_JSON_LAYOUT}}
```'''

def get_grounded_feedback_message(feedback):
    return GROUNDED_FEEDBACK_MESSAGE.format(feedback=feedback)



GROUNDED_QA_PROMPT = '''Here is a sketch design of a webpage drawn in the wireframing conventions. Also, here is a list of text blocks that the user would like to include in the webpage:
"""
{texts}
"""

Here are some additional QAs regarding the sketch design for your reference:
{qa_pairs}

Now, before generating the actual code, let's first carefully analyze the given sketch and record the layout of its visual components in a JSON object. In particular, you should pay attention to the following components:
"""
{components}
"""

Here is the JSON Schema that you should follow when generating the layout structure of the webpage:
"""
{schema}
"""

Please break down the layout of the given sketch and output a JSON layout formatted as:
Page Layout: ```
{{YOUR_JSON_LAYOUT}}
```'''


def get_grounded_qa_prompt(texts, qa_pairs):
  return GROUNDED_USER_PROMPT.format(texts=texts, qa_pairs=qa_pairs, components=VISUAL_COMPONENTS, schema=LAYOUT_SCHEMA)

def get_grounded_qa_message(sketch, texts, qa_pairs):
    if os.path.isfile(sketch):
        sketch = rescale_image_loader(sketch)
    
    return [
        {
            "type": "text",
            "text": GROUNDED_USER_PROMPT.format(texts=texts, qa_pairs=qa_pairs, components=VISUAL_COMPONENTS, schema=LAYOUT_SCHEMA),
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{sketch}"
          }
        },
    ]
