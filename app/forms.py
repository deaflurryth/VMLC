from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

class UploadCsvForm(forms.Form):
    MODEL_CHOICES = [
        ('linear_regression', 'Линейная Регрессия'),
        ('decision_tree', 'Решающее Дерево')
    ]

    csv_file = forms.FileField(label='Загрузите данные .csv')
    model_choice = forms.ChoiceField(choices=MODEL_CHOICES, label='Выберите модель')
    target_variable = forms.CharField(max_length=100, label='Целевая переменная(y)')

class RegisterForm(UserCreationForm):
    email = forms.EmailField()

    class Meta:
        model = User
        fields = ["username", "email", "password1", "password2"]

class LoginForm(forms.Form):
    username = forms.CharField()
    password = forms.CharField(widget=forms.PasswordInput)