def format_number(num):
    if num is None:
        return "N/A"
    
    if num >= 1e9:
        return f"{num / 1e9:.2f} G"
    elif num >= 1e6:
        return f"{num / 1e6:.2f} M"
    elif num >= 1e3:
        return f"{num / 1e3:.2f} K"
    else:
        return str(num)