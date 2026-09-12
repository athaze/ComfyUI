from comfy.cli_args import args, LatentPreviewMethod

default_preview_method = args.preview_method
MAX_PREVIEW_RESOLUTION = args.preview_size

def set_preview_method(override: str = None):
    if override and override != "default":
        method = LatentPreviewMethod.from_string(override)
        if method is not None:
            args.preview_method = method
            return
    args.preview_method = default_preview_method
