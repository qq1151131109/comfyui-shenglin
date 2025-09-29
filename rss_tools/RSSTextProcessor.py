import re
import json

class RSSTextProcessor:
    """
    RSS Text Processor Node
    
    Processes RSS text content with various text manipulation options.
    """
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("processed_text",)
    FUNCTION = "execute"
    CATEGORY = "RSS Content Processing"

    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_text": ("STRING", {
                    "forceInput": True
                }),
                "max_length": ("INT", {
                    "default": 500,
                    "min": 50,
                    "max": 5000,
                    "step": 10
                }),
                "remove_html": ("BOOLEAN", {"default": True}),
                "remove_special_chars": ("BOOLEAN", {"default": False}),
                "case_conversion": (["none", "upper", "lower", "title"], {"default": "none"}),
            },
            "optional": {
                "word_replacement": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "old_word:new_word\nstock:investment"
                }),
                "add_prefix": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
                "add_suffix": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
            }
        }

    def execute(self, input_text, max_length, remove_html, remove_special_chars, 
                case_conversion, word_replacement="", add_prefix="", add_suffix=""):
        return (self.process_text(
            input_text, max_length, remove_html, remove_special_chars,
            case_conversion, word_replacement, add_prefix, add_suffix
        ),)
    
    def process_text(self, input_text, max_length, remove_html, remove_special_chars,
                    case_conversion, word_replacement, add_prefix, add_suffix):
        try:
            processed_text = input_text
            
            # Remove HTML tags
            if remove_html:
                processed_text = re.sub(r'<[^>]+>', '', processed_text)
                processed_text = re.sub(r'&[a-zA-Z]+;', ' ', processed_text)
            
            # Remove special characters
            if remove_special_chars:
                processed_text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', processed_text)
            
            # Clean up whitespace
            processed_text = re.sub(r'\s+', ' ', processed_text).strip()
            
            # Word replacement
            if word_replacement.strip():
                replacement_lines = word_replacement.strip().split('\n')
                for line in replacement_lines:
                    if ':' in line:
                        old_word, new_word = line.split(':', 1)
                        old_word = old_word.strip()
                        new_word = new_word.strip()
                        if old_word and new_word:
                            processed_text = processed_text.replace(old_word, new_word)
            
            # Case conversion
            if case_conversion == "upper":
                processed_text = processed_text.upper()
            elif case_conversion == "lower":
                processed_text = processed_text.lower()
            elif case_conversion == "title":
                processed_text = processed_text.title()
            
            # Truncate to max length
            if len(processed_text) > max_length:
                processed_text = processed_text[:max_length] + "..."
            
            # Add prefix and suffix
            if add_prefix:
                processed_text = add_prefix + processed_text
            if add_suffix:
                processed_text = processed_text + add_suffix
            
            return processed_text
            
        except Exception as e:
            return f"Error processing text: {str(e)}"

NODE_CLASS_MAPPINGS = {
    "RSSTextProcessor": RSSTextProcessor
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RSSTextProcessor": "RSS文本处理器"
}