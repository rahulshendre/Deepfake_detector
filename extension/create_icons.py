"""Run this script to generate extension icons: python3 create_icons.py"""
from PIL import Image, ImageDraw

def create_icon(size):
    img = Image.new('RGBA', (size, size), (26, 26, 46, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw a simple magnifying glass shape
    center = size // 2
    radius = size // 3
    
    # Circle (lens)
    draw.ellipse(
        [center - radius, center - radius - size//8, 
         center + radius, center + radius - size//8],
        outline=(0, 217, 255, 255),
        width=max(2, size // 16)
    )
    
    # Handle
    handle_start = (center + radius - size//10, center + radius - size//6)
    handle_end = (center + radius + size//4, center + radius + size//3)
    draw.line([handle_start, handle_end], fill=(0, 255, 136, 255), width=max(2, size // 12))
    
    return img

# Generate icons
for size in [16, 48, 128]:
    icon = create_icon(size)
    icon.save(f'icon{size}.png')
    print(f'Created icon{size}.png')

print('Done! Icons created.')
