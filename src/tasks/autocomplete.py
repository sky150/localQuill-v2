# Override base run. Streaming no clue how.
from .base import Task

class AutocompleteTask(Task):
    name = "autocomplete"