import cloudinary.uploader
import uuid

def upload_image(image_path):
    id = str(uuid.uuid4())
    with open(image_path, "rb") as image_file:
        response = cloudinary.uploader.upload(image_file, public_id=id, unique_filename=True, overwrite=False)
        secure_url = response.get("secure_url")
    return secure_url