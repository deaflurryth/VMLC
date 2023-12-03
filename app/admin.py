from django.contrib import admin
from .models import UploadedData


class UploadedDataAdmin(admin.ModelAdmin):
    list_display = ('user', 'csv_file', 'model_type')
    fields = ('user', 'csv_file', 'model_type', 'upload_date',
              'MSE','MAE','RSQUARE','ACCURACY',
              'RECALL','PRECISION','F1','ROC',
              'SILHUETTE','DAVIES','INTERIA', 'target_variable') 
    readonly_fields = ('upload_date',)

admin.site.register(UploadedData, UploadedDataAdmin)