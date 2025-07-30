# Simple icon generator for Chrome extension
# Run this file to create the required icon files

try:
    from PIL import Image, ImageDraw, ImageFont
    import os
    
    def create_icon(size, filename):
        # Create image with gradient-like background
        img = Image.new('RGBA', (size, size), (102, 126, 234, 255))  # Blue background
        draw = ImageDraw.Draw(img)
        
        # Add a simple robot-like design
        # Draw a circle for the "head"
        circle_size = size // 2
        circle_pos = (size//4, size//4, size//4 + circle_size, size//4 + circle_size)
        draw.ellipse(circle_pos, fill=(255, 255, 255, 255), outline=(118, 75, 162, 255), width=2)
        
        # Add "eyes"
        eye_size = size // 8
        left_eye = (size//3, size//3, size//3 + eye_size, size//3 + eye_size)
        right_eye = (2*size//3 - eye_size, size//3, 2*size//3, size//3 + eye_size)
        draw.ellipse(left_eye, fill=(102, 126, 234, 255))
        draw.ellipse(right_eye, fill=(102, 126, 234, 255))
        
        # Add a "mouth"
        mouth_width = size // 4
        mouth_y = size // 2
        draw.rectangle((size//2 - mouth_width//2, mouth_y, size//2 + mouth_width//2, mouth_y + 2), 
                      fill=(118, 75, 162, 255))
        
        # Save the image
        img.save(filename, 'PNG')
        print(f"Created {filename}")
    
    # Create icons directory if it doesn't exist
    icons_dir = "icons"
    if not os.path.exists(icons_dir):
        os.makedirs(icons_dir)
    
    # Generate all required icon sizes
    sizes = [16, 32, 48, 128]
    for size in sizes:
        filename = os.path.join(icons_dir, f"icon{size}.png")
        create_icon(size, filename)
    
    print("All icons created successfully!")
    print("You can now reload the Chrome extension.")

except ImportError:
    print("PIL (Pillow) is not installed. Creating simple text-based icons instead...")
    
    # Alternative method: Create very simple SVG icons and convert them
    import os
    
    def create_simple_icon(size, filename):
        # Create a simple SVG content
        svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#667eea;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#764ba2;stop-opacity:1" />
    </linearGradient>
  </defs>
  <rect width="{size}" height="{size}" fill="url(#grad1)" rx="{size//8}"/>
  <circle cx="{size//3}" cy="{size//3}" r="{size//16}" fill="white"/>
  <circle cx="{2*size//3}" cy="{size//3}" r="{size//16}" fill="white"/>
  <rect x="{size//3}" y="{size//2}" width="{size//3}" height="{size//16}" fill="white"/>
  <text x="{size//2}" y="{3*size//4}" text-anchor="middle" fill="white" font-size="{size//8}" font-family="Arial">AI</text>
</svg>'''
        
        with open(filename, 'w') as f:
            f.write(svg_content)
        print(f"Created {filename} (SVG format)")
    
    # Create icons directory if it doesn't exist
    icons_dir = "icons"
    if not os.path.exists(icons_dir):
        os.makedirs(icons_dir)
    
    # Generate all required icon sizes as SVG
    sizes = [16, 32, 48, 128]
    for size in sizes:
        filename = os.path.join(icons_dir, f"icon{size}.svg")
        create_simple_icon(size, filename)
    
    print("\nSVG icons created! For better compatibility, please:")
    print("1. Install Pillow: pip install Pillow")
    print("2. Run this script again to generate PNG icons")
    print("3. Or manually convert SVG files to PNG using an online converter")
