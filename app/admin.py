from django.contrib import admin
from .models import UploadedData

class UploadedDataAdmin(admin.ModelAdmin):
    list_display = ('user', 'csv_file', 'model_type')

admin.site.register(UploadedData, UploadedDataAdmin)