from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
import io
import random
import string

font = ImageFont.truetype("DejaVuSans.ttf", 20)

class Transformer:
    def __init__(self):
        pass

    def transform(self, image, method, **kwargs):
        if method=="screenshot":
            return self.transform(self.transform(image, "crop"), 'jpeg')
    
        elif method == 'jpeg':
            quality = kwargs.get('quality', 30)

            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG', quality=quality)
            img_byte_arr = img_byte_arr.getvalue()
            
            return Image.open(io.BytesIO(img_byte_arr))

        elif method == 'crop':
            width, height = image.size

            # upto 20% crop by default, sum of both sides
            # horizontal = random() * 0.2
            # vertical = random() * 0.2

            # left = horizontal * random()
            # top = vertical * random()
            # right = 1 - (horizontal - left)
            # bottom = 1 - (vertical - top)

            left = 0.2
            top = 0.2
            right = 0.8
            bottom = 0.8

            left = kwargs.get('left', left) * width
            top = kwargs.get('top', top)    * height
            right = kwargs.get('right', right)   * width
            bottom = kwargs.get('bottom', bottom)* height

            return image.crop((left, top, right, bottom))
        
        elif method=="brightness":
            factor = random.uniform(0.7, 1.3)
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(factor)
            
        elif method=="contrast":
            factor = random.uniform(0.7, 1.3)
            enhancer = ImageEnhance.Contrast(image)
            return enhancer.enhance(factor)
        
        elif method=="blur":
            radius = kwargs.get('radius', 2)
            return image.filter(ImageFilter.GaussianBlur(radius=radius))
        
        elif method=="median":
            size = kwargs.get('size', 3)
            return image.filter(ImageFilter.MedianFilter(size=size))
        
        elif method == "double screenshot":
            return self.transform(self.transform(image, "screenshot"), "screenshot")
        elif method == "erase":
            draw = ImageDraw.Draw(image)
            size = round(0.2 * min(image.size))
            coord1 = (random.randint(0, image.size[0]-size), random.randint(0, image.size[1]-size))
            coord2 = (coord1[0]+size, coord1[1]+size)
            draw.rectangle((coord1, coord2), fill="gray")
            return image
        elif method == "text":
            draw = ImageDraw.Draw(image)

            random_text = ''.join(random.choices(string.ascii_letters, k=10))
            text = kwargs.get('text', random_text)

            position = (random.randint(int(image.size[0] * 0.1), int(image.size[0] * 0.3)),
                        random.randint(int(image.size[1] * 0.1), int(image.size[1] * 0.9)))
            draw.text(position, text, fill="blue", align="center", font=font)
            return image
        else:
            raise ValueError('Invalid method')