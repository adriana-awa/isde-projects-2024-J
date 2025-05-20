from fastapi import Request, UploadFile
from typing import Optional

class UploadForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.errors: list = []
        self.file: Optional[UploadFile] = None
        self.model_id: Optional[str] = None

    async def load_data(self):
        form = await self.request.form()
        self.file = form["file"]
        self.model_id = form.get("model_id")

    def is_valid(self):
        if not self.file:
            self.errors.append("No file uploaded")
            return False
        if not self.model_id:
            self.errors.append("No model selected") 
            return False
        return True