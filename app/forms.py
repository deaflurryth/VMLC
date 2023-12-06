from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

class UploadCsvForm(forms.Form):
    MODEL_CHOICES = [
        ('linear_regression', 'Линейная Регрессия'),
        ('decision_tree', 'Решающее Дерево'),
        ('logistic_regression', 'Логистическая Регрессия'),
        ('svm', 'Метод Опорных Векторов'),
        ('knn', 'K-Ближайших Соседей'),
        ('kmeans', 'Методы Кластеризации'),
        ('gradient_boosting', 'Градиентный Бустинг'),
        ('neural_network', 'Нейронные Сети'),
    ]

    csv_file = forms.FileField(label='Загрузка данных')
    model_choice = forms.ChoiceField(choices=MODEL_CHOICES, label='Модель')
    target_variable = forms.CharField(max_length=100, label='Целевая переменная')

class RegisterForm(UserCreationForm):
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ["username", "email", "password1", "password2"]

class LoginForm(forms.Form):
    username = forms.CharField()
    password = forms.CharField(widget=forms.PasswordInput)