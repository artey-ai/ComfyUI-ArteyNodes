"""
@author: Jeff Crouse
@title: Artey Nodes
@nickname: Artey Nodes
@description: Custom nodes for the Artey.ai project
"""

from nodes import SaveImage
from pathlib import Path
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
from botocore.config import Config
import uuid
import os
import torch
import numpy as np
from PIL import Image, ImageSequence, ImageOps
import requests
from io import BytesIO

custom_config = Config(
    retries={
        'max_attempts': 10,        # Maximum number of retry attempts
        'mode': 'standard'         # Retry mode ('standard', 'adaptive', or 'legacy')
    }
)

# https://github.com/comfyanonymous/ComfyUI/blob/75a818c720c1ef0646ad5bf03af740e5d479e882/nodes.py#L1484
class S3Upload(SaveImage):
	
	def __init__(self):
		super().__init__()
		try:
			sts = boto3.client('sts')
			response = sts.get_caller_identity()
			print("[Upload to S3] Authorized as:", response["Arn"])

		except NoCredentialsError:
			print("[Upload to S3] No AWS credentials found.")
		except PartialCredentialsError:
			print("[Upload to S3] Incomplete AWS credentials found.")
		except Exception as e:
			print(f"[Upload to S3] Authorization failed: {e}")


	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"image": ("IMAGE",),
				"bucket_name": ("STRING", {"default": "artey-userassets-01"})
			},
			"hidden": {"extra_pnginfo": "EXTRA_PNGINFO", "prompt": "PROMPT"},
		}

	RETURN_TYPES = ()
	#RETURN_NAMES = ()
	CATEGORY = "ArteyNodes"
	FUNCTION = "execute"
	OUTPUT_NODE = True


	def upload_to_s3(self, image_path, bucket_name):
		try:
			s3 = boto3.client('s3', config=custom_config)

			s3.head_bucket(Bucket=bucket_name)
			print(f"[Upload to S3] Bucket '{bucket_name}' exists and is accessible.")

			id = uuid.uuid4()
			ext = os.path.splitext(image_path)[1]
			object_name = f"{id}{ext}"
			
			print(f"[Upload to S3] Uploading {image_path} to {object_name} ")

			s3.upload_file(image_path, bucket_name, object_name)
			url = f"https://{bucket_name}.s3.amazonaws.com/{object_name}"
			print(f"[Upload to S3] Image successfully uploaded: {url}")

			return url

		except NoCredentialsError:
			print("[Upload to S3] NoCredentialsError AWS credentials not available")

		except FileNotFoundError as e:
			print(f"[Upload to S3] FileNotFoundError {str(e)}")

		except PartialCredentialsError:
			print("[Upload to S3] Incomplete AWS credentials found.")

		except ClientError as e:
			print(f"[Upload to S3] Bucket access error: {e}")

		return None


	def execute(self, image=None, bucket_name=None, prompt=None, extra_pnginfo=None):
		
		data = {
			"ui": {
				"images": [],
			}
		}
		if image is not None:

			saved = super().save_images(image, "S3", prompt, extra_pnginfo)

			image = saved["ui"]["images"][0]
			image_path = Path(self.output_dir).joinpath(image["subfolder"], image["filename"])
			image["url"] = self.upload_to_s3(image_path, bucket_name)

			data["ui"]["images"] = [image]

		return data


# https://github.com/tsogzark/ComfyUI-load-image-from-url/blob/master/nodes/load_image_url_node.py
class LoadFromURL:

	@classmethod
	def INPUT_TYPES(cls):
		return {
			"required": {
				"url": ("STRING", {"multiline": True, "dynamicPrompts": False})
			}
		}
	RETURN_TYPES = ("IMAGE", "MASK")
	FUNCTION = "load"
	CATEGORY = "image"

	def load(self, url):
		print(url)
		response = requests.get(url)
		img = Image.open(BytesIO(response.content))
		img_out, mask_out = self.pil2tensor(img)
		return (img_out, mask_out)

	def pil2tensor(self, img):
		output_images = []
		output_masks = []
		for i in ImageSequence.Iterator(img):
			i = ImageOps.exif_transpose(i)
			if i.mode == 'I':
				i = i.point(lambda i: i * (1 / 255))
			image = i.convert("RGB")
			image = np.array(image).astype(np.float32) / 255.0
			image = torch.from_numpy(image)[None,]
			if 'A' in i.getbands():
				mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
				mask = 1. - torch.from_numpy(mask)
			else:
				mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
			output_images.append(image)
			output_masks.append(mask.unsqueeze(0))

		if len(output_images) > 1:
			output_image = torch.cat(output_images, dim=0)
			output_mask = torch.cat(output_masks, dim=0)
		else:
			output_image = output_images[0]
			output_mask = output_masks[0]

		return (output_image, output_mask)

NODE_CLASS_MAPPINGS = { 
	"Upload to S3" : S3Upload,
	"Load from URL": LoadFromURL
}