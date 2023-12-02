from django import forms

class UploadCsvForm(forms.Form):
    MODEL_CHOICES = [
        ('linear_regression', 'Линейная Регрессия'),
        ('decision_tree', 'Решающее Дерево')
    ]

    csv_file = forms.FileField(label='Загрузите данные .csv')
    model_choice = forms.ChoiceField(choices=MODEL_CHOICES, label='Выберите модель')