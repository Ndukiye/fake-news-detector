from PIL import Image, ImageDraw, ImageFont
import math

def create_shield_icon(size=128):
    """Create a shield icon with checkmark for the extension"""
    # Create new image with transparent background
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Shield parameters
    center_x, center_y = size // 2, size // 2
    shield_width = size * 0.6
    shield_height = size * 0.8
    
    # Draw shield shape (simplified)
    shield_points = [
        (center_x, size * 0.1),  # Top point
        (center_x + shield_width // 2, size * 0.25),  # Top right
        (center_x + shield_width // 2, center_y),  # Middle right
        (center_x + shield_width // 3, size * 0.85),  # Bottom right curve
        (center_x, size * 0.9),  # Bottom center
        (center_x - shield_width // 3, size * 0.85),  # Bottom left curve
        (center_x - shield_width // 2, center_y),  # Middle left
        (center_x - shield_width // 2, size * 0.25),  # Top left
    ]
    
    # Draw shield with gradient effect
    draw.polygon(shield_points, fill=(102, 126, 234, 255), outline=(74, 85, 104, 255))
    
    # Draw checkmark
    check_start = (center_x - size * 0.15, center_y)
    check_middle = (center_x - size * 0.05, center_y + size * 0.1)
    check_end = (center_x + size * 0.15, center_y - size * 0.1)
    
    draw.line([check_start, check_middle, check_end], fill=(255, 255, 255, 255), width=size // 20)
    
    return img

# Create icons in different sizes
for size in [16, 32, 48, 128]:
    icon = create_shield_icon(size)
    out_path = f'extension/icons/icon{size}.png'
    # Ensure directory exists
    import os
    os.makedirs('extension/icons', exist_ok=True)
    icon.save(out_path)
    print(f"Created {out_path}")

print("Extension icons created successfully!")
