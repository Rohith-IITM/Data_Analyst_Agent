from typing import Annotated


from langchain_core.tools import tool
from langchain_experimental.utilities import PythonREPL
import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
from typing import List, Dict, Any, Optional
import time
import logging
from urllib.parse import urlparse
import re
from typing import Annotated
import os
from dotenv import load_dotenv

load_dotenv()



# Warning: This executes code locally, which can be unsafe when not sandboxed

repl = PythonREPL()


@tool
def python_repl_tool(
    code: str
):
    """Use this to execute python code. If you want to see the output of a value,
    you should print it out with `print(...)`. This is visible to the user."""
    try:
        result = repl.run(code)
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"
    print(f"Executed code: {code}")
    print(f"Result: {result}")
    result_str = f"Stdout: {result}"
    return (
        result_str + "\n\nIf you have completed all tasks, respond with FINAL ANSWER."
    )