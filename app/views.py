from django.shortcuts import render, redirect
from .forms import UploadCsvForm
from .models import UploadedData
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from .forms import LoginForm, RegisterForm
from django.http import HttpResponse

# models
import pandas as pd
import matplotlib.pyplot as plt
import os
import zipfile
from io import StringIO
import numpy as np
from django.conf import settings

from sklearn import linear_model 
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.ensemble import GradientBoostingRegressor
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE

def preprocess_data(X, y, model_choice):
    X.fillna(X.mean(), inplace=True)
    y.fillna(y.mean(), inplace=True)

    # Обработка выбросов (здесь - пример с Z-score)
    # from scipy import stats
    # X = X[(np.abs(stats.zscore(X)) < 3).all(axis=1)]

    # Преобразование признаков
    # X = np.log1p(X)

    if model_choice in ['linear_regression', 'logistic_regression', 'svm', 'gradient_boosting']:
        # Кодирование категориальных переменных и стандартизация
        X = pd.get_dummies(X)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    elif model_choice in ['decision_tree', 'knn', 'kmeans']:
        # Кодирование категориальных переменных (если есть)
        X = pd.get_dummies(X)

    if model_choice in ['knn', 'kmeans', 'neural_network']:
        #Отбор признаков (например, с использованием SelectKBest)
        selector = SelectKBest(f_classif, k=10)
        X = selector.fit_transform(X, y)

    if model_choice in ['svm', 'neural_network']:
        # Балансировка классов
        smote = SMOTE()
        X, y = smote.fit_resample(X, y)

    return X, y
def handle_linear_regression(X, y, instance):
    model = linear_model.LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    instance.MSE = mean_squared_error(y, predictions)
    instance.MAE = mean_absolute_error(y, predictions)
    instance.RSQUARE = r2_score(y, predictions)
    # Генерация графика
    predictions = model.predict(X)
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue')
    plt.plot(X, predictions, color='red')
    plt.title('Linear Regression Result')
    plt.xlabel('Independent Variable')
    plt.ylabel('Dependent Variable')

    # Сохранение графика в файл
    graph_filename = f'linear_regression_{instance.id}.png'
    graph_path = os.path.join(settings.MEDIA_ROOT, 'graphs', graph_filename)
    plt.savefig(graph_path)

    # Обновление экземпляра модели с путем к файлу графика
    instance.graph_file.name = os.path.join('graphs', graph_filename)
    instance.save()
def handle_decision_tree(X, y, instance):
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
    feature_importance = model.feature_importances_
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_importance)), feature_importance)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance for Decision Tree')

    # Сохранение графика
    graph_filename = f'decision_tree_{instance.id}.png'
    graph_path = os.path.join(settings.MEDIA_ROOT, 'graphs', graph_filename)
    plt.savefig(graph_path)

    # Обновление экземпляра модели
    instance.graph_file.name = graph_path
    instance.save()
def handle_logistic_regression(X, y, instance):
    model = LogisticRegression()
    threshold = 0.5
    y_class = (y > threshold).astype(int)
    model.fit(X, y_class)
    predictions = model.predict(X)
    instance.ACCURACY = accuracy_score(y_class, predictions)
    instance.RECALL = recall_score(y_class, predictions)
    instance.PRECISION = precision_score(y_class, predictions)
    instance.F1 = f1_score(y_class, predictions)
    X_np = X.to_numpy() 
    plt.figure(figsize=(10, 6))
    plt.scatter(X_np[:, 0], X_np[:, 1], c=y)
    ax = plt.gca()
    x_vals = np.array(ax.get_xlim())
    y_vals = -(x_vals * model.coef_[0][0] + model.intercept_[0]) / model.coef_[0][1]
    plt.plot(x_vals, y_vals, '--')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Result')

    # Сохранение графика
    graph_filename = f'logistic_regression_{instance.id}.png'
    graph_path = os.path.join(settings.MEDIA_ROOT, 'graphs', graph_filename)
    plt.savefig(graph_path)

    # Обновление экземпляра модели
    instance.graph_file.name = graph_path
    instance.save()
def handle_svm(X, y, instance):
    model = SVC()
    threshold = 0.5
    y_class = (y > threshold).astype(int)
    model.fit(X, y_class)
    predictions = model.predict(X)
    instance.ACCURACY = accuracy_score(y, predictions)
    instance.RECALL = recall_score(y, predictions)
    instance.PRECISION = precision_score(y, predictions)
    instance.F1 = f1_score(y, predictions)
    # Визуализация
    X_np = X.to_numpy() if isinstance(X, pd.DataFrame) else X 
    plt.figure(figsize=(10, 6))
    plt.scatter(X_np[:, 0], X_np[:, 1], c=y)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM')

    # Сохранение графика
    graph_filename = f'svm_{instance.id}.png'
    graph_path = os.path.join(settings.MEDIA_ROOT, 'graphs', graph_filename)
    plt.savefig(graph_path)

    # Обновление экземпляра модели
    instance.graph_file.name = graph_path
def handle_knn(X, y, instance):
    model = KNeighborsClassifier()
    model.fit(X, y)
    predictions = model.predict(X)
    instance.ACCURACY = accuracy_score(y, predictions)
    instance.RECALL = recall_score(y, predictions)
    instance.PRECISION = precision_score(y, predictions)
    instance.F1 = f1_score(y, predictions)
    # Визуализация
    X_np = X.to_numpy() if isinstance(X, pd.DataFrame) else X 
    plt.figure(figsize=(10, 6))
    plt.scatter(X_np[:, 0], X_np[:, 1], c=y)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('KNN Classification')

    # Сохранение графика
    graph_filename = f'knn_{instance.id}.png'
    graph_path = os.path.join(settings.MEDIA_ROOT, 'graphs', graph_filename)
    plt.savefig(graph_path)

    # Обновление экземпляра модели
    instance.graph_file.name = graph_path
def handle_kmeans(X, y, instance):
    model = KMeans(n_clusters=3)
    model.fit(X)
    labels = model.labels_
    instance.SILHUETTE = silhouette_score(X, labels)
    instance.DAVIES = davies_bouldin_score(X, labels)

    X_np = X.to_numpy() if isinstance(X, pd.DataFrame) else X  # Преобразование в Numpy массив, если это DataFrame

    plt.figure(figsize=(10, 6))

    # Проверка, является ли X одномерным
    if X_np.shape[1] == 1:
        # Для одномерных данных создаем произвольные y-значения
        y_values = np.zeros(X_np.shape[0])
        plt.scatter(X_np[:, 0], y_values, c=labels)
        plt.scatter(model.cluster_centers_[:, 0], np.zeros(model.cluster_centers_.shape[0]), c='red', marker='x')
    else:
        # Для многомерных данных используем первые два признака
        plt.scatter(X_np[:, 0], X_np[:, 1], c=labels)
        plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], c='red', marker='x')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2' if X_np.shape[1] > 1 else 'Arbitrary Value')
    plt.title('K-Means Clustering')

    # Сохранение графика
    graph_filename = f'kmeans_{instance.id}.png'
    graph_path = os.path.join(settings.MEDIA_ROOT, 'graphs', graph_filename)
    plt.savefig(graph_path)

    # Обновление экземпляра модели
    instance.graph_file.name = graph_path
    instance.save()
def handle_gradient(X, y, instance):
    model = GradientBoostingRegressor()
    model.fit(X, y)
    predictions = model.predict(X)
    instance.MSE = mean_squared_error(y, predictions)
    instance.MAE = mean_absolute_error(y, predictions)
    instance.RSQUARE = r2_score(y, predictions)
    X_np = X.to_numpy() 
    plt.figure(figsize=(10, 6))
    plt.scatter(X_np[:, 0], X_np[:, 1], c=y)
    ax = plt.gca()
    x_vals = np.array(ax.get_xlim())
    y_vals = -(x_vals * model.coef_[0][0] + model.intercept_[0]) / model.coef_[0][1]
    plt.plot(x_vals, y_vals, '--')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Gradient boosting')

    # Сохранение графика
    graph_filename = f'gradient_{instance.id}.png'
    graph_path = os.path.join(settings.MEDIA_ROOT, 'graphs', graph_filename)
    plt.savefig(graph_path)

    # Обновление экземпляра модели
    instance.graph_file.name = graph_path
    instance.save()
def handle_neural(X, y, instance):
    # Определение и обучение нейронной сети
    class SimpleNN(nn.Module):
        def __init__(self, input_size):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_size, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, 1)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = torch.sigmoid(self.fc3(x))
            return x

    # Подготовка данных
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test.values, dtype=torch.float32)

    # Инициализация модели
    model = SimpleNN(X_train.shape[1])

    # Функция потерь и оптимизатор
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Обучение
    num_epochs = 100
    loss_list = []
    for epoch in range(num_epochs):
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Оценка
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_outputs = (test_outputs.squeeze() > 0.5).float()
        accuracy = (test_outputs == y_test).sum().item() / len(y_test)

    instance.ACCURACY = accuracy

    # Визуализация потерь
    plt.figure(figsize=(10, 6))
    plt.plot(range(num_epochs), loss_list)  # Используйте loss_list напрямую
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')

    # Сохранение графика
    graph_filename = f'neural_network_{instance.id}.png'
    graph_path = os.path.join(settings.MEDIA_ROOT, 'graphs', graph_filename)
    plt.savefig(graph_path)

    # Обновление экземпляра модели
    instance.graph_file.name = graph_path
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

            # определение целевой переменной и признаков
            y = df[target_variable]
            X = df.drop(columns=[target_variable])

            # выбор и обучение модели
            if model_choice == 'linear_regression':
                X, y = preprocess_data(X, y, model_choice)
                handle_linear_regression(X, y, instance)
            elif model_choice == 'decision_tree':
                X, y = preprocess_data(X, y, model_choice)
                handle_decision_tree(X, y, instance)
            elif model_choice == 'logistic_regression':
                
                handle_logistic_regression(X, y, instance)
            elif model_choice == 'svm':
                handle_svm(X, y, instance)
            elif model_choice == 'knn':
                handle_knn(X, y, instance)
            elif model_choice == 'kmeans':
                X = df
                handle_kmeans(X, y, instance)
            elif model_choice == 'gradient_boosting':
                handle_gradient(X, y, instance)
            elif model_choice == 'neural_network':
                handle_neural(X, y, instance)
            else:
                return redirect('index')
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


def resulations_models(request):
    # Получение последней модели пользователя
    last_model = UploadedData.objects.filter(user=request.user).order_by('-upload_date').first()

    if last_model:
        context = {
            'model_type': last_model.model_type,
            'target_variable': last_model.target_variable,
            'graph_url': last_model.graph_file.url if last_model.graph_file else None,
            'MSE': last_model.MSE,
            'MAE': last_model.MAE,
            'RSQUARE': last_model.RSQUARE,
            'ACCURACY': last_model.ACCURACY,
            'RECALL': last_model.RECALL,
            'PRECISION': last_model.PRECISION,
            'F1': last_model.F1,
            # Добавьте другие метрики по необходимости
        }
    else:
        context = {'error': 'Результаты не найдены.'}

    return render(request, 'result.html', context)

def get_model_metrics(instance):
    metrics = ""
    if instance.model_type == 'linear_regression':
        metrics = f"MSE: {instance.MSE}\nMAE: {instance.MAE}\nR-Squared: {instance.RSQUARE}\n"
    elif instance.model_type == 'decision_tree':
        metrics = (f"MSE: {instance.MSE}\nMAE: {instance.MAE}\nR-Squared: {instance.RSQUARE}\n"
                   f"Accuracy: {instance.ACCURACY}\nRecall: {instance.RECALL}\n"
                   f"Precision: {instance.PRECISION}\nF1 Score: {instance.F1}\n")
    elif instance.model_type == 'logistic_regression':
        metrics = (f"Accuracy: {instance.ACCURACY}\nRecall: {instance.RECALL}\n"
                   f"Precision: {instance.PRECISION}\nF1 Score: {instance.F1}\n")
    elif instance.model_type == 'svm':
        metrics = (f"Accuracy: {instance.ACCURACY}\nRecall: {instance.RECALL}\n"
                   f"Precision: {instance.PRECISION}\nF1 Score: {instance.F1}\n")
    elif instance.model_type == 'knn':
        metrics = (f"Accuracy: {instance.ACCURACY}\nRecall: {instance.RECALL}\n"
                   f"Precision: {instance.PRECISION}\nF1 Score: {instance.F1}\n")
    elif instance.model_type == 'kmeans':
        metrics = f"Silhouette Score: {instance.SILHUETTE}\nDavies-Bouldin Score: {instance.DAVIES}\n"
    elif instance.model_type == 'gradient_boosting':
        metrics = f"MSE: {instance.MSE}\nMAE: {instance.MAE}\nR-Squared: {instance.RSQUARE}\n"
    elif instance.model_type == 'neural_network':
        # Добавьте здесь метрики для нейронной сети, если они у вас есть
        metrics = f"Accuracy: {instance.ACCURACY}\n"

    return metrics
def download_current_results(request):
    # Получение данных последней модели пользователя
    last_model = UploadedData.objects.filter(user=request.user).order_by('-upload_date').first()

    if not last_model:
        return HttpResponse('Результаты не найдены.', status=404)

    # Формирование строки с метриками
    metrics = get_model_metrics(last_model)
    # Добавьте другие метрики, если они есть

    # Создание архива
    archive_name = 'model_results.zip'
    archive_path = os.path.join(settings.MEDIA_ROOT, archive_name)

    with zipfile.ZipFile(archive_path, 'w') as archive:
        # Добавление файла графика
        if last_model.graph_file:
            graph_path = last_model.graph_file.path
            archive.write(graph_path, os.path.basename(graph_path))

        # Добавление текстовых метрик
        archive.writestr('model_metrics.txt', metrics)

    # Отправка архива пользователю
    with open(archive_path, 'rb') as f:
        response = HttpResponse(f.read(), content_type='application/zip')
        response['Content-Disposition'] = f'attachment; filename={archive_name}'
        return response