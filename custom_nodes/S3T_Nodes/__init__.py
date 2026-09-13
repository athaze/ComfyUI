
#__init__.pys
from .nodes import Speech_To_Text , Text_to_Text , Text_to_Speech
from comfy_api.latest import ComfyExtension, io
from typing_extensions import override

class S3TExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [Speech_To_Text,Text_to_Text,Text_to_Speech]

async def comfy_entrypoint() ->  S3TExtension:
    return S3TExtension()
