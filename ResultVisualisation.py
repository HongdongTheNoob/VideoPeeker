from PIL import Image, ImageDraw, ImageFont

def text_to_image(text, font_size=96, image_size=(400, 200), background_color=(255, 255, 255), text_color=(0, 0, 0), font_path="C:/Users/hongdong.qin/AppData/Local/Microsoft/Windows/Fonts/Inconsolata-Regular.ttf"):
    # Create a new image with the specified background color
    image = Image.new("RGB", image_size, background_color)
    draw = ImageDraw.Draw(image)

    # Load a font
    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    # Calculate text size and position
    text_width, text_height = draw.textsize(text, font=font)
    text_x = (image_size[0] - text_width) // 2
    text_y = (image_size[1] - text_height) // 2

    # Draw the text on the image
    draw.text((text_x, text_y), text, fill=text_color, font=font)

    return image

