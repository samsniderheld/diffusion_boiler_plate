import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image


class Face_Enhancer_Pipeline():
    def __init__(self, model_path):
        self.model_path = model_path

    def load_pipeline(self):
        self.base_optionsbase_options = python.BaseOptions(model_asset_path=self.model_path)
        self.options = vision.FaceDetectorOptions(base_options=self.base_options)
        self.detector = vision.FaceDetector.create_from_options(self.options)

    def detect_face(img_path):
        image = mp.Image.create_from_file(img_path)
        detection_result = self.detector.detect(image)
        return detection_result.detections[0].bounding_box
    
    def crop_face(img_path, bounding_box, buffer):
        start_img = Image.open(img_path)
        p1_x = bounding_box.origin_x - buffer
        p1_y = bounding_box.origin_y - buffer
        p2_x = p1_x+bounding_box.width + buffer
        p2_y = p1_y+bounding_box.height + buffer
        cropped = start_img.crop((p1_x,p1_y, p2_x, p2_y))

        return cropped, start_img, p1_x, p1_y


    def enhance_face(self, img_path, buffer, pipeline, prompt,negative_prompt, controlnet_scale, control_guidance_start, control_guidance_end, lora_weights, cfg, steps, seed=None, width=1024, height=1024,clip_skip=0,img2img_str=1):
        
        bbox = self.detect_face(img_path)
        face, start_img, x,y = self.crop_face(img_path, bbox, buffer)
        cropped_file = "cropped_face.jpg"
        face.save(cropped_file)
        cropped_size = face.size

        enhanced_face, params, control_imgs = pipeline.generate_img(
            prompt = prompt,
            negative_prompt = negative_prompt,
            controlnet_image_path = cropped_file,
            controlnet_scale = controlnet_scale,
            control_guidance_start = control_guidance_start,
            control_guidance_end = control_guidance_end,
            cfg = cfg,
            seed = seed,
            steps = steps,
            width = width,
            height = height,
            clip_skip = clip_skip,
            lora_weights = lora_weights,
            img2img_str = img2img_str
        )

        enhanced_face_resized = enhanced_face.resize(cropped_size)
        out_img = start_img.copy()
        out_img.paste(enhanced_face_resized,(x,y))

        

        return out_img, enhanced_face