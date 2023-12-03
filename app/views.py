from django.shortcuts import render, redirect
from .forms import UploadCsvForm
from .models import UploadedData
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from .forms import LoginForm, RegisterForm

# models
import pandas as pd
from sklearn import linear_model 
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


####################################
@login_required
def index(request):
    if request.method == 'POST':
        form = UploadCsvForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.cleaned_data['csv_file']
            model_choice = form.cleaned_data['model_choice']
            target_variable = form.cleaned_data['target_variable']

            instance = UploadedData(csv_file=uploaded_file, user=request.user, model_type=model_choice, target_variable=target_variable)
            instance.save()
            file_path = instance.csv_file.path
            df = pd.read_csv(file_path)

            # Определение целевой переменной и признаков
            y = df[target_variable]
            X = df.drop(columns=[target_variable])

            # Выбор и обучение модели
            if model_choice == 'linear_regression':
                model = linear_model.LinearRegression()
                model.fit(X, y)
                predictions = model.predict(X)
                instance.MSE = mean_squared_error(y, predictions)
                instance.MAE = mean_absolute_error(y, predictions)
                instance.RSQUARE = r2_score(y, predictions)
            elif model_choice == 'decision_tree':
                model = DecisionTreeRegressor()
                model.fit(X, y)
                predictions = model.predict(X)
                instance.MSE = mean_squared_error(y, predictions)
                instance.MAE = mean_absolute_error(y, predictions)
                instance.RSQUARE = r2_score(y, predictions)
                binary_predictions = [1 if pred >= 0.5 else 0 for pred in predictions]
                instance.ACCURACY = accuracy_score(y, binary_predictions)
                instance.RECALL = recall_score(y, binary_predictions)
                instance.PRECISION = precision_score(y, binary_predictions)
                instance.F1 = f1_score(y, binary_predictions)

            instance.save()

            return redirect('success')
    else:
        form = UploadCsvForm()
    return render(request, 'index.html', {'form': form})
####################################


def success_result(request):
    return render(request, 'success_result.html')

def register(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=password)
            login(request, user)
            return redirect('VMLC')
    else:
        form = RegisterForm()
    return render(request, 'register.html', {'form': form})

def user_login(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('VMLC')
    else:
        form = LoginForm()
    return render(request, 'login.html', {'form': form})