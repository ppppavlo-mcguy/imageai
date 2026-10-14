"""Sequence builders: every training example is one causal sequence of ids from a single vocabulary.

    caption-ish  <img> img1... img2... <sep> <ask> QUESTION <sep> <cap|ascii> ANSWER <eos>
    draw it      <img> INSTRUCTION <sep> img1... img2... <eos>
    write code   <code> INSTRUCTION <sep> SNIPPET <eos>
    ascii2code   <ascii> ART <sep> <code> INSTRUCTION <sep> SNIPPET <eos>
    img2code     <img> img1... img2... <sep> <code> INSTRUCTION <sep> SNIPPET <eos>
    code2img     <code> SNIPPET <sep> <img> INSTRUCTION <sep> img1... img2... <eos>
    plain lm     <bos> TEXT <eos>

The mode tag is written first, so the model decides what it is doing before it reads the rest.
The combination tasks read one modality first, then switch to another with a second tag; that is
why every builder below starts with the tag of its *input*.
"""
from __future__ import annotations

from vocab import SPECIALS, IMG, SEP, ASK, CAP, ASCII, CODE, BOS, EOS

CAPTION_QUESTIONS = [
    "what is on this picture?",
    "describe this image.",
    "what do you see here?",
    "name the main object in this picture.",
    "what is in the image?",
]
ASCII_QUESTIONS = [
    "turn this into an ascii text art",
    "convert this picture into ascii art",
    "make ascii art of this image",
    "render this image as ascii text art",
]
IMAGE_INSTRUCTIONS = [
    "generate the image of {c}",
    "draw {c}",
    "show me {c}",
    "make a picture of {c}",
    "generate {c}",
    "create an image showing {c}",
]
CODE_INSTRUCTIONS = [
    "{i}",
    "please {i}",
    "can you {i}",
    "help me {i}",
    "i need code that does this: {i}",
]

# ------------------------------------------------------------- combination tasks
ASCII2CODE_INSTRUCTIONS = [
    "turn this ascii art into a website",
    "make a web page that shows this ascii art",
    "convert this ascii art into html",
    "build a page for this ascii art",
    "wrap this ascii art in a web page",
]
IMG2CODE_INSTRUCTIONS = [
    "build a web page that looks like this",
    "recreate this image as a website",
    "write html for the page in this picture",
    "make a website out of this picture",
    "design a web page for this image",
]
# "fix this, i got an error" - the input is a snippet plus the exact error text
FIX_QUESTIONS = [
    "i got this error: {e}",
    "fix this, i got an error: {e}",
    "this breaks with: {e}",
    "can you fix it? it fails with {e}",
    "why does this fail: {e}",
]

CODE2IMG_INSTRUCTIONS = [
    "turn this website into an image preview",
    "show me a preview image of this page",
    "render this code as an image",
    "draw what this page looks like",
    "make an image preview of this website",
]


def img_prefix(layout, l1, l2):
    """Image codes -> token ids. Level-1 and level-2 codes live in their own id blocks, so the
    offsets must be added here; the sampler restricts generation to exactly those blocks, and every
    consumer (data.py, generate.py, web/tasks.js) has to agree on that mapping."""
    out = [IMG]
    out += [layout.img1_base + int(c) for c in l1]
    if layout.l2:
        out += [layout.img2_base + int(c) for c in l2]
    return out


def img2text(bpe, layout, l1, l2, question: str, answer: str, kind: str = "cap"):
    tag = ASCII if kind == "ascii" else CAP
    return (img_prefix(layout, l1, l2) + [SEP, ASK] + bpe.encode(question) + [SEP, tag]
            + bpe.encode(answer) + [EOS])


def text2img(bpe, layout, instruction: str, l1, l2):
    return ([IMG] + bpe.encode(instruction) + [SEP] + img_prefix(layout, l1, l2)[1:] + [EOS])


def code_seq(bpe, instruction: str, code: str):
    return [CODE] + bpe.encode(instruction) + [SEP] + bpe.encode(code) + [EOS]


def ascii2code_seq(bpe, art: str, instruction: str, code: str):
    """<ascii> ART <sep> <code> INSTRUCTION <sep> SNIPPET <eos>"""
    return [ASCII] + bpe.encode(art) + [SEP, CODE] + bpe.encode(instruction) \
        + [SEP] + bpe.encode(code) + [EOS]


def img2code_seq(bpe, layout, l1, l2, instruction: str, code: str):
    """<img> img1... img2... <sep> <code> INSTRUCTION <sep> SNIPPET <eos>"""
    return img_prefix(layout, l1, l2) + [SEP, CODE] + bpe.encode(instruction) \
        + [SEP] + bpe.encode(code) + [EOS]


def fix_seq(bpe, code: str, question: str, fixed: str):
    """<code> SNIPPET <sep> <ask> i got this error: MESSAGE <sep> FIXED <eos>"""
    return [CODE] + bpe.encode(code) + [SEP, ASK] + bpe.encode(question) + [SEP] \
        + bpe.encode(fixed) + [EOS]


def code2img_seq(bpe, layout, code: str, instruction: str, l1, l2):
    """<code> SNIPPET <sep> <img> INSTRUCTION <sep> img1... img2... <eos>"""
    return ([CODE] + bpe.encode(code) + [SEP, IMG] + bpe.encode(instruction) + [SEP]
            + img_prefix(layout, l1, l2)[1:] + [EOS])


def lm_seq(bpe, text: str, tag: int | None = None):
    out = [BOS] + ([tag] if tag is not None else []) + bpe.encode(text) + [EOS]
    return out


def prompt_for_image(instruction_template: str, caption: str, rng) -> str:
    return instruction_template.format(c=caption)


def prompt_for_code(template: str, instruction: str) -> str:
    return template.format(i=instruction)


# --------------------------------------------------------------- inference-time
def ask_image_seq(bpe, text: str):
    """`<img> generate the image of a white car <sep>` ready for image sampling."""
    return [IMG] + bpe.encode(text) + [SEP]


def ask_vision_seq(bpe, layout, l1, l2, question: str):
    return img_prefix(layout, l1, l2) + [SEP, ASK] + bpe.encode(question) + [SEP]


def ask_ascii2code_seq(bpe, art: str, instruction: str):
    """`<ascii> ART <sep> <code> INSTRUCTION <sep>` ready for text sampling."""
    return [ASCII] + bpe.encode(art) + [SEP, CODE] + bpe.encode(instruction) + [SEP]


def ask_fix_seq(bpe, code: str, question: str):
    """`<code> SNIPPET <sep> <ask> i got this error: … <sep>` ready for text sampling."""
    return [CODE] + bpe.encode(code) + [SEP, ASK] + bpe.encode(question) + [SEP]


def ask_code2img_seq(bpe, code: str, instruction: str):
    """`<code> SNIPPET <sep> <img> INSTRUCTION <sep> ` ready for image sampling."""
    return [CODE] + bpe.encode(code) + [SEP, IMG] + bpe.encode(instruction) + [SEP]


def ask_img2code_seq(bpe, layout, l1, l2, instruction: str):
    return img_prefix(layout, l1, l2) + [SEP, CODE] + bpe.encode(instruction) + [SEP]
