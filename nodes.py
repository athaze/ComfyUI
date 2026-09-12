import comfy.model_management

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
EXTENSION_WEB_DIRS = {}
LOADED_MODULE_DIRS = {}

def before_node_execution():
    comfy.model_management.throw_exception_if_processing_interrupted()

def interrupt_processing(value=True):
    comfy.model_management.interrupt_current_processing(value)

async def init_extra_nodes(init_custom_nodes=True, init_api_nodes=True):
    pass

def init_builtin_extra_nodes():
    pass

def init_external_custom_nodes():
    pass
