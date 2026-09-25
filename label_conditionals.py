
PERSIAN_COMMA = "،"

def label_conditionals(s):
    """Finds conditionals in a Persian string"""
    
    if not PERSIAN_COMMA in s:
        # no conditional
        return
    
    # simple case: markers for conditionals
    