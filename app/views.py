from django.shortcuts import render, redirect
from .forms import UploadCsvForm
from .models import UploadedData
import pandas as pd
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login
from django.contrib.auth import authenticate, login

# models
from sklearn import linear_model 
from sklearn.tree import DecisionTreeRegressor


def index(request):
    if request.method == 'POST':
        form = UploadCsvForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.cleaned_data['csv_file']
            model_choice = form.cleaned_data['model_choice']
            instance = UploadedData(csv_file=uploaded_file, user=request.user, model_type=model_choice)
            instance.save()

            file_path = instance.csv_file.path
            df = pd.read_csv(file_path)

            if model_choice == 'linear_regression':
                model = linear_model.LinearRegression()
            elif model_choice == 'decision_tree':
                model = DecisionTreeRegressor()


            return redirect('VMLC')
    else:
        form = UploadCsvForm()
    return render(request, 'index.html', {'form': form})

def success_result(request):
    return render(request, 'success_result.html')

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('index')
    else:
        form = UserCreationForm()
    return render(request, 'registration.html', {'form': form})

def user_login(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('название_страницы_после_входа')  # например, redirect('home')
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})