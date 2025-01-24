import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
import os
load_dotenv()
GOOGLE_GENAI_API_KEY = os.getenv("GOOGLE_GENAI_API_KEY") # Gemini API key has been loaded into the .env file

class GeminiModel():
    def __init__(self):
        # set up the model
        genai.configure(api_key=GOOGLE_GENAI_API_KEY)

        generation_config = {
            "temperature": 0.5,
        }

        system_instructions = """You are a super intelligent AI model that is in control of a robot arm. Your input is a scene with a robotic arm and you will output commands for the robotic arm.
                                 You will follow ALL instructions exactly as expreessed.
                                 You are paired with a robot camera. This robot camera is stationary. Your main task is to infer the user's intentions based off their gaze.
                                 In other words, figure out what the user wants to do with the object that they are staring at."""

        self.model = genai.GenerativeModel(
            model_name = 'gemini-2.0-flash-exp',
            generation_config=generation_config,
            system_instruction=system_instructions,
        )

    def inference3D(self, scene):
        if type(scene) is not Image.Image:
            scene = Image.fromarray(scene)

        # Analyze the image using Gemini
        response = self.model.generate_content(
            [
                scene,
                """
                # Background
                A user is looking at the screen. There is a bright green dot on the screen where the user is looking. Based on the object that the user is looking at, think to yourself:
                    1. What is the user looking at? Detect the 3D bounding box of this object.
                    2. In the context of the scene, what is the important and or relevance of this object?
                    3. Are there other objects In this scene that may or may not interact with the object the user is looking at?
                    4. What could the user possibly want to do with the object?
                    5. What is the most likely action that the user would want to do with this object?
                
                # Instructions:
                Following the chain of thought in the above background section, deduce what the user wants to do with the object.
                Identify the IMMEDIATE next step to achieve what you believe the user wants to do. If there are many possibilities of what the user maay want to do, take the next logical action that overlaps with many different possible end goals.
                
                Then, Detect the 3D bounding box of said object.

                Output a json list where the next course of action uses the key "action" and its 3D bounding box in "box_3d"
                The 3D bounding box format should be [x_center, y_center, z_center, x_size, y_size, z_size, roll, pitch, yaw].
                """
            ],
        )

        print(response.text)
        return response.text

    def inference2D(self, scene):
        if type(scene) is not Image.Image:
            height, width = scene.shape[:2]
            scene = Image.fromarray(scene)

        scene = scene.resize((800, int(800 * scene.size[1] / scene.size[0])), Image.Resampling.LANCZOS) # Resizing to speed-up rendering

        # Analyze the image using Gemini
        response = self.model.generate_content(
            [
                scene,
                """
                # Background
                A user is looking at the screen. There is a bright green dot on the screen where the user is looking. Based on the object that the user is looking at, think to yourself:
                    1. What is the user looking at? Detect the 3D bounding box of this object.
                    2. In the context of the scene, what is the important and or relevance of this object?
                    3. Are there other objects In this scene that may or may not interact with the object the user is looking at?
                    4. What could the user possibly want to do with the object?
                    5. What is the most likely action that the user would want to do with this object?
                
                # Instructions:
                Following the chain of thought in the above background section, deduce what the user wants to do with the object.
                Identify the IMMEDIATE next step to achieve what you believe the user wants to do. If there are many possibilities of what the user maay want to do, take the next logical action that overlaps with many different possible end goals.
                
                Then, point to said object.

                Output following the json format: [{"action": , "point": }, ...].
                The points are in [y, x] format normalized to 0-1000.
                """
            ],
        )

        # Check response
        print(response.text)
        return response.text
