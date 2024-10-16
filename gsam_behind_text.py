# import os
# import cv2
# import json
# import torch
# import numpy as np
# import supervision as sv
# import pycocotools.mask as mask_util
# from pathlib import Path
# from torchvision.ops import box_convert
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
# from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
# from PIL import Image, ImageDraw, ImageFont
# import os
# import gradio as gr
# from PIL import Image
# # from gsam import GroundedSAM2Processor
# import numpy as np


# class GroundedSAM2Processor:
#     def __init__(self, text_prompt, img_path, sam2_checkpoint, sam2_model_config,
#                  grounding_dino_config, grounding_dino_checkpoint, box_threshold=0.05,
#                  text_threshold=0.05, device="cuda", output_dir="outputs/grounded_sam2_local_demo",
#                  dump_json_results=True):
        
#         self.text_prompt = text_prompt
#         self.img_path = img_path
#         self.sam2_checkpoint = sam2_checkpoint
#         self.sam2_model_config = sam2_model_config
#         self.grounding_dino_config = grounding_dino_config
#         self.grounding_dino_checkpoint = grounding_dino_checkpoint
#         self.box_threshold = box_threshold
#         self.text_threshold = text_threshold
#         self.device = device
#         self.output_dir = Path(output_dir)
#         self.dump_json_results = dump_json_results

#         self.output_dir.mkdir(parents=True, exist_ok=True)
#         self.sam2_model = None
#         self.sam2_predictor = None
#         self.grounding_model = None
#         self.image_source = None
#         self.image = None

#     def load_models(self):
#         # Load SAM2 and GroundingDINO models
#         self.sam2_model = build_sam2(self.sam2_model_config, self.sam2_checkpoint, device=self.device)
#         self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
#         self.grounding_model = load_model(
#             model_config_path=self.grounding_dino_config,
#             model_checkpoint_path=self.grounding_dino_checkpoint,
#             device=self.device
#         )

#     def predict(self):
#         # Load the image
#         self.image_source, self.image = load_image(self.img_path)

#         # Set the image for SAM2 predictor
#         self.sam2_predictor.set_image(self.image_source)

#         # Predict bounding boxes and labels using GroundingDINO
#         boxes, confidences, labels = predict(
#             model=self.grounding_model,
#             image=self.image,
#             caption=self.text_prompt,
#             box_threshold=self.box_threshold,
#             text_threshold=self.text_threshold,
#         )

#         # Post-process the boxes for SAM2 input
#         h, w, _ = self.image_source.shape
#         boxes = boxes * torch.Tensor([w, h, w, h])
#         input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

#         # Predict masks using SAM2
#         masks, scores, logits = self.sam2_predictor.predict(
#             point_coords=None,
#             point_labels=None,
#             box=input_boxes,
#             multimask_output=False,
#         )

#         return masks, labels, confidences, input_boxes

#     def add_text_behind_subject(self, image, mask, text, font_path, font_size, text_color, shadow_color, shadow_offset):
#         """
#         Add text behind the subject using the mask.
#         """
#         # Convert image and mask to numpy arrays for processing
#         image_np = np.array(image)
#         mask_np = np.array(mask)

#         # Get the bounding box of the subject mask to calculate where to place the text
#         subject_indices = np.where(mask_np == 255)
#         top_y = min(subject_indices[0])  # Top-most point of the subject
#         left_x = min(subject_indices[1])
#         right_x = max(subject_indices[1])

#         # Place the text slightly below the top of the subject
#         text_position = (left_x - 30, top_y - 19)  # Adjust position slightly below the subject mask

#         # Create an image for drawing the text
#         text_layer = Image.new("RGBA", image.size, (255, 255, 255, 0))  # Transparent layer
#         draw = ImageDraw.Draw(text_layer)
#         font = ImageFont.truetype(font_path, font_size)

#         # Draw shadow first (offset)
#         shadow_position = (text_position[0] + shadow_offset[0], text_position[1] + shadow_offset[1])
#         draw.text(shadow_position, text, fill=shadow_color + (255,), font=font)

#         # Draw the actual text over the shadow
#         draw.text(text_position, text, fill=text_color + (255,), font=font)

#         # Convert the text layer to a numpy array (RGBA)
#         text_np = np.array(text_layer)

#         # Remove the alpha channel from text_np to make it compatible with the image (convert from RGBA to RGB)
#         text_np_rgb = text_np[:, :, :3]

#         # Create a mask where the text is drawn
#         text_mask = text_np[:, :, 3]  # Extract the alpha channel as mask

#         # Overlay the text on the original image while using the subject mask to hide it
#         for c in range(3):  # For each color channel
#             image_np[:, :, c] = np.where(mask_np == 255, image_np[:, :, c], np.where(text_mask == 255, text_np_rgb[:, :, c], image_np[:, :, c]))

#         # Convert the result back to PIL image
#         final_image = Image.fromarray(image_np.astype(np.uint8))
#         return final_image



# # Gradio function to process the image and text prompt
# def process_image(image, text_prompt, target_text, font_size, text_color):
#     image_path = "input_image.jpg"
#     image.save(image_path)
#     text_color_rgb = tuple(int(float(value)) for value in text_color.strip("rgba()").split(",")[:3])
#     print(">>>>>>>>>>>",text_color_rgb)

#     # Ensure the text color is a valid hex code
#     # # Check if the input is in the format of a hex code
#     # if isinstance(text_color, str) and text_color.startswith("#") and len(text_color) == 7:  # Check for #RRGGBB format
#     #     try:
#     #         # Convert hex color from color picker to RGB tuple
#     #         text_color_rgb = tuple(int(text_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
#     #     except ValueError:
#     #         # If the conversion fails, default to black
#     #         text_color_rgb = (0, 0, 0)

#     # # Check if the input is in normal RGB format as a tuple (R, G, B)
#     # elif isinstance(text_color, tuple) and len(text_color) == 4 and all(isinstance(val, int) and 0 <= val <= 255 for val in text_color):
#     #     print(">>>>>>>>>>>>>>>>>>>>>>>")
#     #     text_color_rgb = text_color[:3]

#     # # If the format is wrong, default to black
#     # else:
#     #     text_color_rgb = (0, 0, 0)

#     # Instantiate the processor with the necessary configurations
#     processor = GroundedSAM2Processor(
#         text_prompt=text_prompt,
#         img_path=image_path,
#         sam2_checkpoint="./checkpoints/sam2.1_hiera_large.pt",
#         sam2_model_config="configs/sam2.1/sam2.1_hiera_l.yaml",
#         grounding_dino_config="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
#         grounding_dino_checkpoint="gdino_checkpoints/groundingdino_swint_ogc.pth"
#     )

#     # Load models
#     processor.load_models()

#     # Run prediction to get masks
#     masks, labels, confidences, input_boxes = processor.predict()

#     # Convert mask to grayscale format
#     mask = (masks[0] * 255).astype(np.uint8)
#     mask = np.moveaxis(mask, 0, -1)[:, :, 0]
#     mask_image = Image.fromarray(mask).convert("L")

#     # Add text behind the subject
#     final_image = processor.add_text_behind_subject(
#         image=Image.open(image_path).convert("RGB"),
#         mask=mask_image,
#         text=target_text,
#         font_path="/root/ws/image2txt/Arial.ttf",  # Adjust the font path
#         font_size=font_size,
#         text_color=text_color_rgb,  # Use the selected color from the color picker
#         shadow_color=(150, 0, 0),  # Shadow color
#         shadow_offset=(3, 3)
#     )

#     # Save and return the final image
#     final_image_path = "final_image_with_text_behind_subject.png"
#     final_image.save(final_image_path)
    
#     return final_image_path


# # Create Gradio interface
# interface = gr.Interface(
#     fn=process_image,
#     inputs=[
#         gr.Image(type="pil", label="Upload Image"),
#         gr.Textbox(label="Text Prompt for Object Detection"),
#         gr.Textbox(label="Text to Add Behind Subject"),
#         gr.Slider(minimum=10, maximum=100, value=25, label="Font Size"),  # Added slider for font size
#         gr.ColorPicker(label="Text Color")  # Added color picker for selecting text color
#     ],
#     outputs=gr.Image(type="filepath", label="Final Image with Text Behind Subject"),
#     title="GroundedSAM2 Text Behind Subject"
# )

# # Launch the app
# interface.launch()

#NOTE: Working
# import os
# import cv2
# import json
# import torch
# import numpy as np
# import supervision as sv
# import pycocotools.mask as mask_util
# from pathlib import Path
# from torchvision.ops import box_convert
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
# from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
# from PIL import Image, ImageDraw, ImageFont
# import gradio as gr
# import numpy as np

# class GroundedSAM2Processor:
#     def __init__(self, text_prompt, img_path, sam2_checkpoint, sam2_model_config,
#                  grounding_dino_config, grounding_dino_checkpoint, box_threshold=0.05,
#                  text_threshold=0.05, device="cuda", output_dir="outputs/grounded_sam2_local_demo",
#                  dump_json_results=True):
        
#         self.text_prompt = text_prompt
#         self.img_path = img_path
#         self.sam2_checkpoint = sam2_checkpoint
#         self.sam2_model_config = sam2_model_config
#         self.grounding_dino_config = grounding_dino_config
#         self.grounding_dino_checkpoint = grounding_dino_checkpoint
#         self.box_threshold = box_threshold
#         self.text_threshold = text_threshold
#         self.device = device
#         self.output_dir = Path(output_dir)
#         self.dump_json_results = dump_json_results

#         self.output_dir.mkdir(parents=True, exist_ok=True)
#         self.sam2_model = None
#         self.sam2_predictor = None
#         self.grounding_model = None
#         self.image_source = None
#         self.image = None

#     def load_models(self):
#         # Load SAM2 and GroundingDINO models
#         self.sam2_model = build_sam2(self.sam2_model_config, self.sam2_checkpoint, device=self.device)
#         self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
#         self.grounding_model = load_model(
#             model_config_path=self.grounding_dino_config,
#             model_checkpoint_path=self.grounding_dino_checkpoint,
#             device=self.device
#         )

#     def predict(self):
#         # Load the image
#         self.image_source, self.image = load_image(self.img_path)

#         # Set the image for SAM2 predictor
#         self.sam2_predictor.set_image(self.image_source)

#         # Predict bounding boxes and labels using GroundingDINO
#         boxes, confidences, labels = predict(
#             model=self.grounding_model,
#             image=self.image,
#             caption=self.text_prompt,
#             box_threshold=self.box_threshold,
#             text_threshold=self.text_threshold,
#         )

#         # Post-process the boxes for SAM2 input
#         h, w, _ = self.image_source.shape
#         boxes = boxes * torch.Tensor([w, h, w, h])
#         input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

#         # Predict masks using SAM2
#         masks, scores, logits = self.sam2_predictor.predict(
#             point_coords=None,
#             point_labels=None,
#             box=input_boxes,
#             multimask_output=False,
#         )

#         return masks, labels, confidences, input_boxes

#     def add_text_behind_subject(self, image, mask, text, font_path, font_size, text_color, shadow_color, shadow_offset):
#         """
#         Add text behind the subject using the mask.
#         """
#         # Convert image and mask to numpy arrays for processing
#         image_np = np.array(image)
#         mask_np = np.array(mask)

#         # Get the bounding box of the subject mask to calculate where to place the text
#         subject_indices = np.where(mask_np == 255)
#         top_y = min(subject_indices[0])  # Top-most point of the subject
#         left_x = min(subject_indices[1])
#         right_x = max(subject_indices[1])

#         # Place the text slightly below the top of the subject
#         text_position = (left_x - 30, top_y - 19)  # Adjust position slightly below the subject mask

#         # Create an image for drawing the text
#         text_layer = Image.new("RGBA", image.size, (255, 255, 255, 0))  # Transparent layer
#         draw = ImageDraw.Draw(text_layer)
#         font = ImageFont.truetype(font_path, font_size)

#         # Draw shadow first (offset)
#         shadow_position = (text_position[0] + shadow_offset[0], text_position[1] + shadow_offset[1])
#         draw.text(shadow_position, text, fill=shadow_color + (255,), font=font)

#         # Draw the actual text over the shadow
#         draw.text(text_position, text, fill=text_color + (255,), font=font)

#         # Convert the text layer to a numpy array (RGBA)
#         text_np = np.array(text_layer)

#         # Remove the alpha channel from text_np to make it compatible with the image (convert from RGBA to RGB)
#         text_np_rgb = text_np[:, :, :3]

#         # Create a mask where the text is drawn
#         text_mask = text_np[:, :, 3]  # Extract the alpha channel as mask

#         # Overlay the text on the original image while using the subject mask to hide it
#         for c in range(3):  # For each color channel
#             image_np[:, :, c] = np.where(mask_np == 255, image_np[:, :, c], np.where(text_mask == 255, text_np_rgb[:, :, c], image_np[:, :, c]))

#         # Convert the result back to PIL image
#         final_image = Image.fromarray(image_np.astype(np.uint8))
#         return final_image


# # Function to map font names to file paths
# def get_font_path(font_name):
#     font_dir = "/root/ws/image2txt/fonts/fonts"  # Directory where fonts are stored
#     font_map = {
#         "Arial": "Arial.ttf",
#         "Verdana_Bold": "Verdana_Bold.ttf",
#         "Arial_Italic": "Arial_Italic.ttf"
#     }
#     return os.path.join(font_dir, font_map[font_name])


# # Gradio function to process the image and text prompt
# def process_image(image, text_prompt, target_text, font_size, text_color, font_name):
#     image_path = "input_image.jpg"
#     image.save(image_path)
#     text_color_rgb = tuple(int(float(value)) for value in text_color.strip("rgba()").split(",")[:3])

#     # Get the font path based on the selected font
#     font_path = get_font_path(font_name)

#     # Instantiate the processor with the necessary configurations
#     processor = GroundedSAM2Processor(
#         text_prompt=text_prompt,
#         img_path=image_path,
#         sam2_checkpoint="./checkpoints/sam2.1_hiera_large.pt",
#         sam2_model_config="configs/sam2.1/sam2.1_hiera_l.yaml",
#         grounding_dino_config="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
#         grounding_dino_checkpoint="gdino_checkpoints/groundingdino_swint_ogc.pth"
#     )

#     # Load models
#     processor.load_models()

#     # Run prediction to get masks
#     masks, labels, confidences, input_boxes = processor.predict()

#     # Convert mask to grayscale format
#     mask = (masks[0] * 255).astype(np.uint8)
#     mask = np.moveaxis(mask, 0, -1)[:, :, 0]
#     mask_image = Image.fromarray(mask).convert("L")

#     # Add text behind the subject
#     final_image = processor.add_text_behind_subject(
#         image=Image.open(image_path).convert("RGB"),
#         mask=mask_image,
#         text=target_text,
#         font_path=font_path,  # Select the appropriate font path
#         font_size=font_size,
#         text_color=text_color_rgb,  # Use the selected color from the color picker
#         shadow_color=(150, 0, 0),  # Shadow color
#         shadow_offset=(3, 3)
#     )

#     # Save and return the final image
#     final_image_path = "final_image_with_text_behind_subject.png"
#     final_image.save(final_image_path)
    
#     return final_image_path


# # Create Gradio interface
# interface = gr.Interface(
#     fn=process_image,
#     inputs=[
#         gr.Image(type="pil", label="Upload Image"),
#         gr.Textbox(label="Text Prompt for Object Detection"),
#         gr.Textbox(label="Text to Add Behind Subject"),
#         gr.Slider(minimum=10, maximum=100, value=25, label="Font Size"),  # Slider for font size
#         gr.ColorPicker(label="Text Color"),  # Color picker for text color
#         gr.Dropdown(label="Font Type", choices=["Arial", "Verdana_Bold", "Arial_Italic"], value="Arial")  # Dropdown for font type
#     ],
#     outputs=gr.Image(type="filepath", label="Final Image with Text Behind Subject"),
#     title="GroundedSAM2 Text Behind Subject"
# )

# # Launch the app
# interface.launch()

import os
import cv2
import json
import torch
import numpy as np
import supervision as sv
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from PIL import Image, ImageDraw, ImageFont
import gradio as gr
import numpy as np
import math

class GroundedSAM2Processor:
    def __init__(self, text_prompt, img_path, sam2_checkpoint, sam2_model_config,
                 grounding_dino_config, grounding_dino_checkpoint, box_threshold=0.05,
                 text_threshold=0.05, device="cuda", output_dir="outputs/grounded_sam2_local_demo",
                 dump_json_results=True):
        
        self.text_prompt = text_prompt
        self.img_path = img_path
        self.sam2_checkpoint = sam2_checkpoint
        self.sam2_model_config = sam2_model_config
        self.grounding_dino_config = grounding_dino_config
        self.grounding_dino_checkpoint = grounding_dino_checkpoint
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = device
        self.output_dir = Path(output_dir)
        self.dump_json_results = dump_json_results

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sam2_model = None
        self.sam2_predictor = None
        self.grounding_model = None
        self.image_source = None
        self.image = None

    def load_models(self):
        # Load SAM2 and GroundingDINO models
        self.sam2_model = build_sam2(self.sam2_model_config, self.sam2_checkpoint, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        self.grounding_model = load_model(
            model_config_path=self.grounding_dino_config,
            model_checkpoint_path=self.grounding_dino_checkpoint,
            device=self.device
        )

    def predict(self):
        # Load the image
        self.image_source, self.image = load_image(self.img_path)

        # Set the image for SAM2 predictor
        self.sam2_predictor.set_image(self.image_source)

        # Predict bounding boxes and labels using GroundingDINO
        boxes, confidences, labels = predict(
            model=self.grounding_model,
            image=self.image,
            caption=self.text_prompt,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
        )

        # Post-process the boxes for SAM2 input
        h, w, _ = self.image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        input_boxes = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()

        # Predict masks using SAM2
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=False,
        )

        return masks, labels, confidences, input_boxes

    def add_text_behind_subject(self, image, mask, text, font_path, font_size, text_color,
                                text_position, text_shape, shadow_color, shadow_offset):
        """
        Add text behind the subject using the mask, with options for different shapes and positions.
        """
        image_np = np.array(image)
        mask_np = np.array(mask)

        # Get the bounding box of the subject mask to calculate where to place the text
        subject_indices = np.where(mask_np == 255)
        top_y = min(subject_indices[0])
        left_x = min(subject_indices[1])
        right_x = max(subject_indices[1])

        if text_position == 'left':
            position = (left_x - 30, top_y - 19)
        elif text_position == 'right':
            position = (right_x + 30, top_y - 19)
        else:
            position = ((left_x + right_x) // 2 - 30, top_y - 19)

        text_layer = Image.new("RGBA", image.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(text_layer)
        font = ImageFont.truetype(font_path, font_size)

        if text_shape == 'circle':
            self._draw_text_in_circle(draw, position, text, font, text_color)
        elif text_shape == 'elliptical':
            self._draw_text_in_ellipse(draw, position, text, font, text_color)
        else:
            draw.text(position, text, fill=text_color + (255,), font=font)

        text_np = np.array(text_layer)
        text_np_rgb = text_np[:, :, :3]
        text_mask = text_np[:, :, 3]

        for c in range(3):
            image_np[:, :, c] = np.where(mask_np == 255, image_np[:, :, c],
                                         np.where(text_mask == 255, text_np_rgb[:, :, c], image_np[:, :, c]))

        final_image = Image.fromarray(image_np.astype(np.uint8))
        return final_image


    def _draw_text_in_circle(self, draw, center_position, text, font, text_color, radius=100):
        """
        Helper function to draw text along a circular path.
        """
        x_center, y_center = center_position
        angle_per_char = 360 / len(text)  # Distribute characters evenly around the circle

        for i, char in enumerate(text):
            # Calculate the angle for this character
            angle = math.radians(i * angle_per_char)

            # Get the bounding box for this character to center it
            char_bbox = draw.textbbox((0, 0), char, font=font)
            char_width = char_bbox[2] - char_bbox[0]
            char_height = char_bbox[3] - char_bbox[1]

            # Compute character position along the circle
            x = x_center + radius * math.cos(angle) - char_width / 2
            y = y_center + radius * math.sin(angle) - char_height / 2

            draw.text((x, y), char, fill=text_color + (255,), font=font)

    def _draw_text_in_ellipse(self, draw, center_position, text, font, text_color, width=250, height=200):
        """
        Helper function to draw text along an elliptical path.
        """
        x_center, y_center = center_position
        angle_per_char = 360 / len(text)  # Distribute characters evenly around the ellipse

        for i, char in enumerate(text):
            # Calculate the angle for this character
            angle = math.radians(i * angle_per_char)

            # Get the bounding box for this character to center it
            char_bbox = draw.textbbox((0, 0), char, font=font)
            char_width = char_bbox[2] - char_bbox[0]
            char_height = char_bbox[3] - char_bbox[1]

            # Compute character position along the ellipse using parametric equations
            x = x_center + width * math.cos(angle) - char_width / 2
            y = y_center + height * math.sin(angle) - char_height / 2

            draw.text((x, y), char, fill=text_color + (255,), font=font)



# Function to map font names to file paths
def get_font_path(font_name):
    font_dir = "/root/ws/image2txt/fonts/fonts"  # Directory where fonts are stored
    font_map = {
        "Arial": "Arial.ttf",
        "Verdana_Bold": "Verdana_Bold.ttf",
        "Arial_Italic": "Arial_Italic.ttf"
    }
    return os.path.join(font_dir, font_map[font_name])


# Gradio function to process the image and text prompt
def process_image(image, text_prompt, target_text, font_size, text_color, font_name,
                  text_position, text_shape):
    image_path = "input_image.jpg"
    image.save(image_path)
    text_color_rgb = tuple(int(float(value)) for value in text_color.strip("rgba()").split(",")[:3])

    font_path = get_font_path(font_name)

    processor = GroundedSAM2Processor(
        text_prompt=text_prompt,
        img_path=image_path,
        sam2_checkpoint="./checkpoints/sam2.1_hiera_large.pt",
        sam2_model_config="configs/sam2.1/sam2.1_hiera_l.yaml",
        grounding_dino_config="grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        grounding_dino_checkpoint="gdino_checkpoints/groundingdino_swint_ogc.pth"
    )

    processor.load_models()

    masks, labels, confidences, input_boxes = processor.predict()

    mask = (masks[0] * 255).astype(np.uint8)
    mask = np.moveaxis(mask, 0, -1)[:, :, 0]
    mask_image = Image.fromarray(mask).convert("L")

    final_image = processor.add_text_behind_subject(
        image=Image.open(image_path).convert("RGB"),
        mask=mask_image,
        text=target_text,
        font_path=font_path,
        font_size=font_size,
        text_color=text_color_rgb,
        text_position=text_position,  # Position (left, center, right)
        text_shape=text_shape,  # Shape (circle, elliptical, etc.)
        shadow_color=(150, 0, 0),
        shadow_offset=(3, 3)
    )

    final_image_path = "final_image_with_text_behind_subject.png"
    final_image.save(final_image_path)
    
    return final_image_path


# Create Gradio interface
interface = gr.Interface(
    fn=process_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(label="Text Prompt for Object Detection"),
        gr.Textbox(label="Text to Add Behind Subject"),
        gr.Slider(minimum=10, maximum=100, value=25, label="Font Size"),
        gr.ColorPicker(label="Text Color"),
        gr.Dropdown(label="Font Type", choices=["Arial", "Verdana_Bold", "Arial_Italic"], value="Arial"),
        gr.Dropdown(label="Text Position", choices=["left", "center", "right"], value="center"),  # Dropdown for position
        gr.Dropdown(label="Text Shape", choices=["default", "circle", "elliptical"], value="default")  # Dropdown for shape
    ],
    outputs=gr.Image(type="filepath", label="Final Image with Text Behind Subject"),
    live=False
)

interface.launch()
