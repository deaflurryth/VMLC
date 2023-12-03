from django.db import models
from django.contrib.auth.models import User

class UploadedData(models.Model):
    LINEAR_REGRESSION = 'linear_regression'
    DECISION_TREE = 'decision_tree'
    #для селекторов
    MODEL_CHOICES = [
        (LINEAR_REGRESSION, 'Линейная Регрессия'),
        (DECISION_TREE, 'Решающее Дерево'),
    ]
    YES_NO = [
        ('yes', 'да'),
        ('no', 'нет'),
    ]
    #Основные поля
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    csv_file = models.FileField(upload_to='uploaded_files/')
    model_type = models.CharField(max_length=100, choices=MODEL_CHOICES)
    upload_date = models.DateTimeField(auto_now_add=True)
    target_variable = models.CharField(max_length=100, default='y', blank=False)

    #Метрики
    MSE = models.CharField(max_length=33, null=True, blank=True)
    MAE = models.CharField(max_length=33, null=True, blank=True)
    RSQUARE = models.CharField(max_length=33, null=True, blank=True)
    ACCURACY = models.CharField(max_length=33, null=True, blank=True)
    RECALL = models.CharField(max_length=33, null=True, blank=True)
    PRECISION = models.CharField(max_length=33, null=True, blank=True)
    F1 = models.CharField(max_length=33, null=True, blank=True)
    ROC = models.CharField(max_length=33, null=True, blank=True)
    SILHUETTE = models.CharField(max_length=33, null=True, blank=True)
    DAVIES = models.CharField(max_length=33, null=True, blank=True)
    INTERIA = models.CharField(max_length=33, null=True, blank=True)