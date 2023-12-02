from django.db import models
from django.conf import settings
class UploadedData(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    csv_file = models.FileField(upload_to='uploaded_files/')
    model_type = models.CharField(max_length=100)