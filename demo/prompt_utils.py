import json
import base64
import os
from utils import rescale_image_loader



def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')


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


CUSTOM_AGENT_SYSTEM_PROMPT = '''You are an expert web developer who specializes in HTML and CSS. A user will provide you with a sketch design of the webpage drawn in the wireframing conventions, where images are replaced by boxes with an "X" inside and texts are represented by curly lines. Your task is to write a single html file that uses HTML and CSS to produce a webpage that strictly follows the sketch design. Include all CSS code in the HTML file itself. If it involves any images, use "rick.jpg" as the placeholder.

Before you start coding, you should ask the user to clarify any ambiguities in the sketch. You should formulate effective questions to better understand the layout, stylistic choices, additional design details on the visual components.

Please format your questions as:
Question: """
1. {{First_Question}}
2. {{Second_Question}}
3. {{Third_Question}}
...
"""

You should **avoid** asking the user to clarify the content of the images and instead use "rick.jpg" as the placeholder for all images. Also, since the webpage is going to be a static HTML page, you should also avoid asking about any detailed user interactions that cannot be achieved from HTML+CSS alone.'''



CUSTOM_AGENT_USER_PROMPT = '''Here is a sketch design of a webpage drawn in the wireframing conventions. The user has also provided some high-level descriptions of the intended webpage:

{description}

Please formulate exactly {num_questions} thoughtful and specific questions to better understand the layout, stylistic choices, and/or additional design details on the visual components.

You should list your questions as:
Question: """
1. {{First_Question}}
2. {{Second_Question}}
3. {{Third_Question}}
...
"""

Remember, all images should use "rick.jpg" as the placeholder, and you should avoid asking users about image contents or detailed user interactivity.'''


def get_agent_system_message():
    return AGENT_SYSTEM_PROMPT


def get_agent_user_message(sketch_path, topic):
    sketch = encode_image(sketch_path)
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


def get_custom_agent_system_message():
    return CUSTOM_AGENT_SYSTEM_PROMPT

def get_custom_agent_user_message(sketch, description, num_questions, focus="any"):
  if os.path.isfile(sketch):
     sketch = encode_image(sketch)
  
  user_prompt = CUSTOM_AGENT_USER_PROMPT.format(description=description, num_questions=num_questions)
  if focus == "style":
    user_prompt += "\n\n Please focus your questions on the color schemes and stylistic choices of visual components."
  elif focus == "layout":
    user_prompt += "\n\n Please focus your questions on the layout, sizing, and positional placement of visual components."
  elif focus == "text content":
    user_prompt += "\n\n Please focus your questions on the font styles and exact text content of textual components."
  elif focus == "design details":
    user_prompt += "\n\n Please focus your questions on the detailed design choices (e.g., alignment, spacing, borders, interactive components, placeholder texts, etc.) of visual components."
    
  return [
      {
          "type": "text",
          "text": user_prompt,
      },
      {
        "type": "image_url",
        "image_url": {
          "url": f"data:image/jpeg;base64,{sketch}"
        }
      },
  ]



DIRECT_SYSTEM_POMPT = '''You are an expert web developer who specializes in HTML and CSS. A user will provide you with a sketch design of the webpage following the wireframing conventions, where images are represented as boxes with an "X" inside, and texts are replaced with curly lines. You need to return a single html file that uses HTML and CSS to produce a webpage that strictly follows the sketch layout. Include all CSS code in the HTML file itself. If it involves any images, use "rick.jpg" as the placeholder name unless specified otherwise. You should try your best to figure out what text should be placed in each text block. In you are unsure, you may use "lorem ipsum..." as the placeholder text. However, you must make sure that the positions and sizes of these placeholder text blocks matches those on the provided sketch.

Do your best to reason out what each element in the sketch represents and write a HTML file with embedded CSS that implements the design. Do not hallucinate any dependencies to external files. Pay attention to things like size and position of all the elements, as well as the overall layout. You may assume that the page is static and ignore any user interactivity.

As a reminder, the "rick.jpg" placeholder is very large (1920 x 1080). So make sure to always specify the correct dimensions for the images in your HTML code, since otherwise the image would likely take up the entire page.

When reading the user-provided sketch, pay attention to whether it is intended to be a regular webpage or a mobile webpage. If the sketch is intended to be a mobile webpage, make sure to enclose all UI components inside a container similar to the size of a mobile screen.'''


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


GROUNDER_SYSTEM_PROMPT = '''You are an intelligent assistant trained to help users better understand the components and layout of a webpage design sketch. The sketch follows the wireframing conventions, where images are represented as boxes with a cross inside, and texts are replaced with curly lines.

Given a sketch design, your job is to break down the sketch layout and output a detailed description of the sketch layout. Your description should include the type, size, and position of all visual components identified from the sketch (including but not limited to images, text blocks, forms/tables, buttons, search/navigation bars, dividers/separators, etc.). Your layout description should be as detailed as possible, but do not hallucinate any information that is not directly obtainable from the sketch. You may assume that the page is static and ignore any user interactivity.

If you are unsure about something, or if a component is unclear, you should ask the user to clarify the uncertainty. You should focus on understanding the higher-level layout and the type, size, & position of each visual component, but you should ignore details in color and styling or the exact texts in each text block. Be specific when you ask a question.

If you want to ask a clarification question, format your question using triple quotes: 
Question: """{{YOUR_QUESTION_HERE}}"""

To ask multiple questions in a single turn, you should list your questions as:
Question: """
1. {{First_Question}}
2. {{Second_Question}}
3. {{Third_Question}}
...
"""

If you are ready to output the final description, wrap your layout description using:
Layout Description: """
{{YOUR_LAYOUT_DESCRIPTION_HERE}}
"""'''


GROUNDER_USER_PROMPT = '''Here is a sketch design of a webpage layout drawn in the wireframing conventions. Please break down the sketch step by step and provide a detailed layout description of the sketch.

Remember, your description should include the type, size, and position of **all** visual components identified from the sketch, but do not hallucinate any information that is not directly obtainable from the sketch. You do not need to report the exact number of lines within each text block, as they are not meant to be accurate. You may also assume that the page is static and ignore any user interactivity.

In order to get the most accurate layout, you should break down layout of the sketch, reason about the type, size, and position of each visual component, and identify places with potential ambiguities. You should ask the user to clarify any ambiguities before reaching a final layout description.

If you want to ask a clarification question, format your question using triple quotes: 
Question: """{{YOUR_QUESTION_HERE}}"""

To ask multiple questions in a single turn, you should list your questions as:
Question: """
1. {{First_Question}}
2. {{Second_Question}}
3. {{Third_Question}}
...
"""

Once you are ready, wrap your final description with a triple quote:
Layout Description: """
{{YOUR_LAYOUT_DESCRIPTION_HERE}}
"""

Your output should be a step-by-step breakdown of the sketch layout, identifying the type, size, and position of each visual component and addressing any uncertainties or ambiguities. You should then ask clarifying questions based on the uncertainties, or output a final description if you deem that everything is clear'''


def init_grounder_messages(sketch):
    if os.path.isfile(sketch):
        sketch = rescale_image_loader(sketch)
    
    return [
      {
        "role": "system",
        "content": GROUNDER_SYSTEM_PROMPT,
      },
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": GROUNDER_USER_PROMPT,
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{sketch}"
            }
          }
        ]
      }
    ]


GROUNDER_REFLECTION_PROMPT = '''Here are the answers to your previous questions:
{answers}

Include the new information into your reasoning of the sketch layout. Are you confident with the information you have about the sketch? Are there any remaining uncertainties?

If you would like get further clarifications on any visual components or the overall composition, format your questions as:
Question: """
1. {{First_Question}}
2. {{Second_Question}}
3. {{Third_Question}}
...
"""

If you are confident about the sketch layout, output your final description with a triple quote:
Layout Description: """
{{YOUR_LAYOUT_DESCRIPTION_HERE}}
"""'''


def get_grounder_reflection_message(answers):
  return {
    "role": "user",
    "content": GROUNDER_REFLECTION_PROMPT.format(answers=answers)
  }
  

MATCHER_USER_PROMPT = '''Here is a sketch of a web page drawn in the wireframing conventions where images are represented as boxes with a cross inside, and texts are replaced with curly lines.

Here are some additional information on the layout of the visual components within the webpage:
Page Layout: """
{layout}
"""

Please match the following blocks of texts to the components of the webpage:
Block Index -> Text: """
{text}
"""

You should assign each block of text to the most suitable webpage component and update the given Page Layout to reflect the assignment. Instead of reciting the full paragraphs of texts, you should use the Block Index as the identifier for a block of text in your updated Page Layout.

Explain your reasonings step by step. If you are unsure about certain text assignments, you should ask the user to provide clarifications. Please be very specific with your questions. You may format your questions as a numerical list wrapped in triple quotes:
Question: """
1. {{First_Question}}
2. {{Second_Question}}
3. {{Third_Question}}
...
"""

If you are confident with your text assignments, output your updated Page Layout using triple quotes:
Page Layout: """
{{PAGE_LAYOUT_WITH_TEXT_ASSIGNMENTS}}
"""

You should explain your thoughts step by step, provide reasonings for each assignment, and address any uncertainties. You should then ask clarification questions based on the identified uncertainties or output the updated page layout if you deem everything is clear.'''

MATCHER_REFLECTION_PROMPT = '''The user responded with the following answers:
{answers}

Integrate the new information into your text assignments. Are you confident with the assignment now?

If you would like to ask for further clarifications, format your questions as a numerical list wrapped in triple quotes:
Question: """
1. {{First_Question}}
2. {{Second_Question}}
3. {{Third_Question}}
...
"""

If you are confident with your text assignments, output your updated Page Layout using triple quotes:
Page Layout: """
{{PAGE_LAYOUT_WITH_TEXT_ASSIGNMENTS}}
"""'''


def init_matcher_messages(sketch, layout, text):
    if os.path.isfile(sketch):
        sketch = rescale_image_loader(sketch)
    
    return [
      {
        "role": "user",
        "content": [
          {
            "type": "text",
            "text": MATCHER_USER_PROMPT.format(layout=layout, text=text),
          },
          {
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{sketch}"
            }
          }
        ]
      }
    ]

def get_matcher_reflection_message(answers):
    return {
        "role": "user",
        "content": MATCHER_REFLECTION_PROMPT.format(answers=answers)
    }



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



def get_agent_user_augmented_user_message(sketch, description):
    if os.path.isfile(sketch):
        sketch = encode_image(sketch)
    
    return [
        {
            "type": "text",
            "text": AGENT_USER_AUGMENTED_USER_PROMPT.format(description=description),
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{sketch}"
          }
        },
    ]

AGENT_USER_AUGMENTED_USER_PROMPT = '''Here is a sketch design of a webpage drawn in the wireframing conventions. The user has also provided some high-level descriptions of the intended webpage:

{description}

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


def get_user_augmented_message_with_qa_pairs(sketch, description, qa_pairs):
    if os.path.isfile(sketch):
        sketch = encode_image(sketch)
        
    return [
        {
            "type": "text",
            "text": USER_AUGMENTED_USER_PROMPT_WITH_QA_PAIRS.format(description=description, qa_pairs=qa_pairs),
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{sketch}"
          }
        },
    ]


USER_AUGMENTED_USER_PROMPT_WITH_QA_PAIRS = '''Here is a sketch design of a webpage drawn in the wireframing conventions. The user has also provided some high-level descriptions of the intended webpage:

{description}

Could you write a HTML+CSS code of this webpage for me?

Here are some additional information for your reference:
{qa_pairs}

Please format your code as
```
{{HTML_CSS_CODE}}
```
Remember to use "rick.jpg" as the placeholder for any images unless specified otherwise'''


def get_direct_user_augmented_user_message(sketch, description):
    if os.path.isfile(sketch):
        sketch = encode_image(sketch)
    
    return [
        {
            "type": "text",
            "text": USER_AUGMENTED_DIRECT_MESSAGE.format(description=description),
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{sketch}"
          }
        },
    ]

USER_AUGMENTED_DIRECT_MESSAGE = '''Here is a sketch design of a webpage drawn in the wireframing conventions. The user has also provided some high-level descriptions of the intended webpage:

{description}

Could you write a HTML+CSS code of this webpage for me?

Please format your code as
```
{{HTML_CSS_CODE}}
```
Remember to use "rick.jpg" as the placeholder for any images'''