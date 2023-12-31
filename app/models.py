from django.db import models
from django.contrib.auth.models import User

class UploadedData(models.Model):
    LINEAR_REGRESSION = 'linear_regression'
    DECISION_TREE = 'decision_tree'
    LOGISTIC_REGRESSION = 'logistic_regression'
    SVM = 'svm'
    KNN = 'knn'
    KMEANS = 'kmeans'
    GRADIENT_BOOSTING = 'gradient_boosting'
    NEURAL_NETWORK = 'neural_network'
    #для селекторов
    MODEL_CHOICES = [
        (LINEAR_REGRESSION, 'Линейная Регрессия'),
        (DECISION_TREE, 'Решающее Дерево'),
        (LOGISTIC_REGRESSION, 'Логистическая Регрессия'),
        (SVM, 'Метод Опорных Векторов'),
        (KNN, 'K-Ближайших Соседей'),
        (KMEANS, 'Методы Кластеризации'),
        (GRADIENT_BOOSTING, 'Градиентный Бустинг'),
        (NEURAL_NETWORK, 'Нейронные Сети'),
    ]
    YES_NO = [
        ('yes', 'да'),
        ('no', 'нет'),
    ]
    #Основные поля
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    csv_file = models.FileField(upload_to='uploaded_files/')
    graph_file = models.FileField(upload_to='graphs/', null=True, blank=True)
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